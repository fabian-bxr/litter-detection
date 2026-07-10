from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from loguru import logger

from litter_agents.agents.reporter import add_llm_summary, build_report
from litter_agents.agents.search_area import build_search_area_agent, parse_search_area
from litter_agents.agents.validator import build_validation_agent, make_validate_fn
from litter_agents.config import AgentSettings
from litter_agents.debug_render import TrajectoryRenderer
from litter_agents.hunter.explore import explore
from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.planner import ExplorationPlanner
from litter_agents.interfaces.mission import MissionReport, SearchAreaSpec
from litter_agents.mapping.provider import FileMapProvider, MapProvider
from litter_agents.mapping.raster import rasterize_area
from litter_agents.mission.nav_client import ZenohNavClient
from litter_agents.mission.pose_tracker import ZenohPoseTracker
from litter_agents.validation.findings import FindingsRepository
from litter_agents.validation.worker import DetectionValidationWorker, decode_tracked
from litter_agents.zenoh_bridge import AsyncZenoh


class MissionController:
    """Wires one litter-search mission together.

    Sequence: wait for localization → load map → parse the search area
    (LLM, unless a spec is given) → start the detection-validation worker →
    run the deterministic exploration loop → halt, drain, report.
    """

    def __init__(
        self,
        settings: AgentSettings | None = None,
        *,
        map_provider: MapProvider | None = None,
    ) -> None:
        self.settings = settings or AgentSettings()
        self.params = HunterParams.from_settings(self.settings)
        self._map_provider = map_provider or FileMapProvider(self.settings.map_yaml_path)

    async def run(
        self,
        prompt: str,
        *,
        area_spec: SearchAreaSpec | None = None,
        confirm: bool = False,
        llm_summary: bool = True,
        enable_validation: bool = True,
    ) -> MissionReport:
        settings = self.settings
        mission_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        started = time.time()
        logger.info("Mission {} starting: {!r}", mission_id, prompt)

        az = AsyncZenoh(settings.zenoh_config())
        repo = FindingsRepository(settings.findings_db_path)
        worker: DetectionValidationWorker | None = None
        nav: ZenohNavClient | None = None
        try:
            # ── liveness ────────────────────────────────────────────────────
            pose_tracker = ZenohPoseTracker(az)
            try:
                start_pose = await pose_tracker.wait_first(10.0)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "No robot pose received within 10 s — is the robodog "
                    "localization stack (and the zenoh router) running?"
                ) from None
            logger.info(
                "Robot at ({:.2f}, {:.2f}, θ {:.2f})",
                start_pose.x,
                start_pose.y,
                start_pose.theta,
            )
            if enable_validation:
                detections_alive = az.subscribe_latest(
                    settings.topics().detection.tracked, decode_tracked
                )

            # ── map & search area ───────────────────────────────────────────
            grid = await self._map_provider.load()
            if area_spec is None:
                area_agent = build_search_area_agent(settings)
                area_spec = await parse_search_area(area_agent, prompt)
                logger.info(
                    "Search area: {} ({})", area_spec.model_dump_json(), area_spec.rationale
                )
            target = rasterize_area(area_spec, start_pose, grid)

            planner = ExplorationPlanner(grid, target, self.params, start_pose)
            target_m2 = float(target.sum()) * grid.resolution**2
            reachable_m2 = planner.coverage.denominator_m2()
            logger.info(
                "Target area {:.1f} m², reachable & observable {:.1f} m²",
                target_m2,
                reachable_m2,
            )
            if reachable_m2 <= 0.0:
                raise RuntimeError(
                    "No reachable free space inside the requested area — "
                    "check the area spec and the map metadata."
                )
            if confirm:
                await asyncio.to_thread(
                    input, "Press Enter to start the mission (Ctrl-C to abort)... "
                )

            # ── validation worker ───────────────────────────────────────────
            repo.start_mission(mission_id, prompt, area_spec, time.time_ns())
            if enable_validation:
                if detections_alive.latest is None:
                    await asyncio.sleep(3.0)
                if detections_alive.latest is None:
                    logger.warning(
                        "Nothing on {} yet — detector pipeline appears down; "
                        "searching anyway, no litter will be validated",
                        settings.topics().detection.tracked,
                    )
                validation_agent = build_validation_agent(settings)
                worker = DetectionValidationWorker(
                    az,
                    pose_tracker,
                    repo,
                    make_validate_fn(validation_agent, settings),
                    settings,
                    mission_id,
                )
                worker.start()

            # ── exploration ─────────────────────────────────────────────────
            nav = ZenohNavClient(az, pose_tracker, settings)
            renderer: TrajectoryRenderer | None = None
            if settings.debug_render:
                debug_dir = Path(settings.findings_dir) / mission_id / "debug"
                renderer = TrajectoryRenderer(grid, target, debug_dir)
                renderer.trajectory.append((start_pose.x, start_pose.y))
            planner.coverage.update(start_pose)
            coverage_task = asyncio.create_task(
                self._coverage_loop(planner, pose_tracker, renderer)
            )
            try:
                stats = await explore(
                    planner,
                    nav,
                    pose_tracker,
                    max_speed=settings.nav_max_speed,
                    max_waypoints=settings.mission_max_waypoints,
                    max_duration_s=settings.mission_max_duration_s,
                    no_gain_limit=settings.no_gain_replans_before_stop,
                    blocked_wait_s=settings.blocked_retreat_wait_s,
                )
            finally:
                coverage_task.cancel()
                await nav.halt()
                if renderer is not None:
                    final_pose = pose_tracker.latest or start_pose
                    path = renderer.save_frame(
                        planner.coverage.seen,
                        final_pose,
                        reachable=planner.coverage.denominator(),
                        obstacles=planner.dynamic.layer,
                        name="overview.png",
                    )
                    logger.info("Debug frames written to {}", path.parent)

            # ── wrap up ─────────────────────────────────────────────────────
            if worker is not None:
                logger.info("Draining validation queue…")
                await worker.stop(drain_timeout_s=90.0)
            report = build_report(
                mission_id=mission_id,
                prompt=prompt,
                area=area_spec,
                coverage_fraction=planner.coverage.fraction(),
                reachable_target_m2=planner.coverage.denominator_m2(),
                duration_s=time.time() - started,
                distance_traveled_m=pose_tracker.distance_traveled,
                n_waypoints=stats.n_waypoints,
                n_blocked=stats.n_blocked,
                validated=repo.findings(mission_id, status="validated"),
                status_counts=repo.status_counts(mission_id),
            )
            if llm_summary:
                await add_llm_summary(report, settings)
            self._persist_report(repo, report)
            return report
        finally:
            if worker is not None:
                await worker.stop(drain_timeout_s=5.0)  # no-op if already stopped
            repo.close()
            az.close()

    async def _coverage_loop(
        self,
        planner: ExplorationPlanner,
        pose_tracker: ZenohPoseTracker,
        renderer: TrajectoryRenderer | None = None,
    ) -> None:
        """Absorb live poses into the coverage grid, including while driving.

        Doubles as the debug-frame driver: accumulates the trajectory and
        periodically saves a frame, mirroring the offline sim's render loop.
        """
        interval = 1.0 / self.settings.coverage_update_hz
        render_every = self.settings.debug_render_interval_s
        last_render = -float("inf")
        while True:
            pose = pose_tracker.latest
            if pose is not None:
                planner.coverage.update(pose)
                if renderer is not None:
                    renderer.trajectory.append((pose.x, pose.y))
                    now = time.monotonic()
                    if render_every > 0 and now - last_render >= render_every:
                        renderer.save_frame(
                            planner.coverage.seen,
                            pose,
                            reachable=planner.coverage.denominator(),
                            obstacles=planner.dynamic.layer,
                        )
                        last_render = now
            await asyncio.sleep(interval)

    def _persist_report(
        self, repo: FindingsRepository, report: MissionReport
    ) -> None:
        report_json = report.model_dump_json(indent=2)
        repo.finish_mission(
            report.mission_id,
            finished_ns=time.time_ns(),
            coverage_fraction=report.coverage_fraction,
            distance_m=report.distance_traveled_m,
            n_waypoints=report.n_waypoints,
            n_blocked=report.n_blocked,
            report_json=report_json,
        )
        out_dir = Path(self.settings.findings_dir) / report.mission_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(report_json)
        logger.info("Report written to {}", out_dir / "report.json")
