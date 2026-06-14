"""MissionController — orchestrates the full litter-search mission.

Phase 5 will add the SearchArea agent and Reporter.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..config import AgentSettings
from ..hunter.coverage import CoverageTracker
from ..hunter.navigator import ZenohNavClient
from ..hunter.planner import ExplorationPlanner
from ..interfaces.robodog import NavigationState
from ..mapping.provider import FileMapProvider
from ..mapping.raster import AreaSpec, rasterize_area
from ..validation.findings_db import FindingRecord, FindingsDB
from ..validation.vision_agent import LitterValidationResult
from ..validation.worker import ValidationWorker
from ..zenoh_bridge import AsyncZenoh
from .pose_tracker import ZenohPoseTracker


@dataclass
class MissionResult:
    coverage_fraction: float
    waypoints_visited: int
    distance_m: float
    duration_s: float
    termination_reason: str
    findings: list[dict] = field(default_factory=list)


class MissionController:
    """Runs a full exploration mission and returns a MissionResult.

    Args:
        area_spec:  If provided, the area to search.  If None, a SearchArea
                    agent will be called (Phase 5).
        confirm:    If True, print plan and ask for Enter before moving.
    """

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self._cfg = settings or AgentSettings()

    async def run(
        self,
        area_spec: AreaSpec | None = None,
        prompt: str | None = None,
        confirm: bool = False,
        viz: bool = False,
    ) -> MissionResult:
        loop = asyncio.get_event_loop()
        bridge = AsyncZenoh(loop, self._cfg.zenoh_router_endpoint)

        pose_tracker = ZenohPoseTracker(bridge)
        pose_tracker.start()

        mission_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_ts = datetime.now(timezone.utc).isoformat()
        db = FindingsDB(self._cfg.findings_db_path)
        await db.init()

        validation_worker = ValidationWorker(
            bridge=bridge,
            db=db,
            mission_id=mission_id,
            run_ts=run_ts,
            pose_fn=pose_tracker.current,
            cfg=self._cfg,
            on_finding=self._on_finding,
        )
        validation_worker.start()

        try:
            result = await self._run_inner(
                bridge, pose_tracker, area_spec, prompt, confirm,
                mission_id=mission_id,
                viz=viz,
            )
        finally:
            await validation_worker.stop()
            await pose_tracker.stop()
            await bridge.close()

        # Attach findings from DB to result
        result.findings = await db.query_mission(mission_id)
        return result

    async def _on_finding(
        self,
        rec: FindingRecord,
        result: LitterValidationResult,
    ) -> None:
        status = "CONFIRMED" if rec.confirmed else "rejected"
        print(
            f"  [{status}] track #{rec.track_id}  "
            f"conf={rec.confidence:.2f}  cat={rec.category or '-'}  "
            f"→ {rec.description}"
        )

    async def _run_inner(
        self,
        bridge: AsyncZenoh,
        pose_tracker: ZenohPoseTracker,
        area_spec: AreaSpec | None,
        prompt: str | None,
        confirm: bool,
        mission_id: str = "",
        viz: bool = False,
    ) -> MissionResult:
        cfg = self._cfg

        # 2. Wait for first pose
        print("Waiting for robot pose …")
        start_pose = await pose_tracker.wait_first(timeout_s=15.0)
        print(f"  Robot at x={start_pose.x:.2f} y={start_pose.y:.2f} θ={start_pose.theta:.2f}")

        # 3. Load map
        grid = FileMapProvider(cfg.map_path).load()
        inflated = grid.inflate(cfg.robot_radius_m)
        print(f"  Map loaded: {grid.width}×{grid.height} px @ {grid.resolution:.3f} m/px")

        # 4. Resolve search area
        if area_spec is None:
            raise ValueError(
                "area_spec must be provided — call SearchAreaAgent first "
                "or use --area-circle / --area-rect."
            )
        area_mask = rasterize_area(area_spec, start_pose, grid)
        area_cells = int(area_mask.sum())
        area_m2 = area_cells * grid.resolution**2
        print(f"  Search area: {area_m2:.1f} m² ({area_cells} cells)")

        # 5. Coverage tracker
        coverage = CoverageTracker(
            grid,
            area_mask,
            fov_deg=cfg.fov_deg,
            range_m=cfg.seen_range_m,
            min_range_m=cfg.camera_min_range_m,
        )

        # 6. Planner + nav client
        planner = ExplorationPlanner(grid, inflated, coverage, cfg)
        nav = ZenohNavClient(bridge, pose_tracker.current)

        # 6b. Optional live map visualisation
        live_map = None
        stop_viz = asyncio.Event()
        viz_task = None
        if viz:
            from .live_map import LiveMap
            live_map = LiveMap(grid, coverage, pose_tracker, area_mask)
            viz_task = asyncio.create_task(live_map.run(stop_viz), name="live-map")

        if confirm:
            input("Plan ready — press Enter to start the mission …")

        # 7. Background coverage update loop (5 Hz)
        stop_cov = asyncio.Event()

        async def _cov_loop() -> None:
            while not stop_cov.is_set():
                coverage.update(pose_tracker.current())
                await asyncio.sleep(0.2)

        cov_task = asyncio.create_task(_cov_loop(), name="cov-update")

        # 8. Exploration loop
        t0 = time.monotonic()
        termination_reason = "unknown"
        no_candidate_streak = 0
        consecutive_limit = cfg.consecutive_low_gain_limit

        try:
            while not planner.done():
                pose = pose_tracker.current()
                candidate = planner.next_waypoint(pose)
                if candidate is None:
                    no_candidate_streak += 1
                    if no_candidate_streak >= consecutive_limit:
                        termination_reason = "no_candidates"
                        break
                    await asyncio.sleep(0.1)
                    continue

                no_candidate_streak = 0
                cov_frac = coverage.fraction()
                print(
                    f"  WP#{planner.n_waypoints:03d}  "
                    f"→ ({candidate.x:.2f}, {candidate.y:.2f})  "
                    f"gain={candidate.gain_m2:.2f} m²  "
                    f"cov={cov_frac*100:.1f}%"
                )

                if live_map:
                    live_map.set_waypoint(candidate.x, candidate.y)
                    live_map.update_stats(
                        planner.n_waypoints, pose_tracker.distance_traveled_m
                    )

                result = await nav.goto(candidate.pose)
                if result == NavigationState.BLOCKED:
                    planner.register_block(pose_tracker.current(), candidate)
                    if live_map:
                        live_map.set_waypoint(None, None)

            if planner.done():
                cov = coverage.fraction()
                if cov >= cfg.coverage_threshold:
                    termination_reason = "coverage_reached"
                elif planner.consecutive_low_gain >= consecutive_limit:
                    termination_reason = "low_gain"
                elif planner.n_waypoints >= cfg.mission_max_waypoints:
                    termination_reason = "max_waypoints"
                else:
                    termination_reason = "timeout"

        finally:
            stop_cov.set()
            await cov_task
            await nav.halt()
            if viz_task is not None:
                stop_viz.set()
                await viz_task

        duration_s = time.monotonic() - t0
        final_cov = coverage.fraction()

        result = MissionResult(
            coverage_fraction=final_cov,
            waypoints_visited=planner.n_waypoints,
            distance_m=pose_tracker.distance_traveled_m,
            duration_s=duration_s,
            termination_reason=termination_reason,
        )

        print(
            f"\nMission done — coverage={final_cov*100:.1f}%  "
            f"WPs={planner.n_waypoints}  "
            f"dist={pose_tracker.distance_traveled_m:.1f} m  "
            f"time={duration_s:.0f} s  "
            f"reason={termination_reason}  "
            f"(validation in progress)"
        )
        return result
