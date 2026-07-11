from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from litter_agents.agents.validator import build_validation_agent, make_validate_fn
from litter_agents.config import AgentSettings, repo_path
from litter_agents.debug_render import render_coverage_overlay
from litter_agents.hunter.explore import explore
from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.planner import ExplorationPlanner
from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.provider import build_map_provider
from litter_agents.mapping.raster import rasterize_area
from litter_agents.mission.orchestrator import MissionController
from litter_agents.sim.fake_nav import FakeNav, FakePoseSource
from litter_agents.sim.sim_main import default_start
from litter_agents.validation.findings import FindingsRepository
from litter_agents.validation.worker import DetectionValidationWorker
from litter_agents.zenoh_bridge import AsyncZenoh

router = APIRouter(tags=["mission"])

_REPO_ROOT = Path(__file__).parents[3]

# ── Module-scope mission state ────────────────────────────────────────────────
_running_mission: asyncio.Task[None] | None = None
_current_mission_id: str | None = None
_mission_log: deque[str] = deque(maxlen=200)
_log_subscribers: set[asyncio.Queue[str]] = set()
_loop: asyncio.AbstractEventLoop | None = None
_sink_id: int | None = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _loguru_sink(message: Any) -> None:
    """Loguru sink — may be called from any thread."""
    _emit_line(str(message).strip())


def _emit_line(line: str) -> None:
    _mission_log.append(line)
    if _loop is not None and _log_subscribers:
        for q in list(_log_subscribers):
            _loop.call_soon_threadsafe(_safe_put, q, line)


def _safe_put(q: asyncio.Queue[str], item: str) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass


def _setup_sink() -> int:
    return logger.add(
        _loguru_sink,
        format="{time:HH:mm:ss} {level.name:<8} {message}",
        colorize=False,
        level="INFO",
    )


def _remove_sink() -> None:
    global _sink_id
    if _sink_id is not None:
        try:
            logger.remove(_sink_id)
        except ValueError:
            pass
        _sink_id = None


def _new_mission_id(body: "StartBody") -> str:
    """Mint the id up front so the UI can show an empty board from tick zero."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if body.detection_test:
        return f"{stamp}-camtest-{uuid.uuid4().hex[:4]}"
    if body.sim_mode:
        return f"{stamp}-sim-{uuid.uuid4().hex[:4]}"
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _ui_area_spec(body: "StartBody") -> SearchAreaSpec | None:
    """The area we already know at start; None when the LLM still has to parse it."""
    if body.detection_test or body.circle_radius_m is None:
        return None
    return SearchAreaSpec(
        shape="circle", radius_m=body.circle_radius_m, rationale="Provided by UI"
    )


def _open_repo(settings: AgentSettings) -> FindingsRepository:
    return FindingsRepository(repo_path(settings.findings_db_path))


def _coverage_stats(planner: ExplorationPlanner) -> dict:
    """Small numeric summary shipped alongside the overlay for the UI badge."""
    return {
        "fraction": planner.coverage.fraction(),
        "reachable_m2": planner.coverage.denominator_m2(),
    }


def _push_coverage_overlay(planner: ExplorationPlanner, grid: Any, target: Any) -> None:
    """Render the current exploration state and publish it to /ws/state."""
    import litter_ui.zenoh_state as zstate

    png = render_coverage_overlay(
        grid,
        target,
        planner.coverage.seen,
        planner.coverage.denominator(),
        planner.dynamic.layer,
    )
    zstate.set_coverage_overlay(png, _coverage_stats(planner))


# ── Sim runner ────────────────────────────────────────────────────────────────

async def _run_sim(body: "StartBody", mission_id: str) -> None:
    """Exploration sim with FakePoseSource/FakeNav; broadcasts to /ws/state."""
    import litter_ui.zenoh_state as zstate

    settings = AgentSettings()
    params = HunterParams.from_settings(settings)

    # Explore the same map the UI displays (build_map_provider honours
    # map_source), so the live coverage overlay lines up with /api/map/image.
    grid = await build_map_provider(settings).load()
    start_pose = default_start(grid, params.robot_radius_m)

    radius = body.circle_radius_m or 5.0
    area_spec = SearchAreaSpec(shape="circle", radius_m=radius, rationale="sim")
    target = rasterize_area(area_spec, start_pose, grid)
    planner = ExplorationPlanner(grid, target, params, start_pose)

    pose_source = FakePoseSource(start_pose)

    # Reset map state so the sim starts from a clean slate
    zstate.path_history.clear()
    zstate.planned_path = []
    zstate.pose_latest = start_pose
    zstate.clear_coverage_overlay()

    tick_count = 0
    last_overlay = float("-inf")  # monotonic time of last overlay render

    def on_tick(pose: Pose2D) -> None:
        nonlocal tick_count, last_overlay
        planner.coverage.update(pose)  # accumulate FoV coverage from live poses
        zstate.pose_latest = pose
        zstate.path_history.append((pose.x, pose.y))
        tick_count += 1
        if tick_count % 5 == 0:  # throttle plain state broadcast
            zstate._broadcast()
        now = time.monotonic()
        if now - last_overlay >= 0.4:  # cap overlay re-render / re-encode rate
            last_overlay = now
            _push_coverage_overlay(planner, grid, target)

    nav = FakeNav(
        pose_source,
        on_tick=on_tick,
        grid=grid,
        blocked_inflated=planner.blocked_inflated(),
        skip_start_m=params.robot_radius_m,
    )

    repo = _open_repo(settings)

    logger.info("Sim {} gestartet — Radius {:.1f} m, Startpose ({:.2f}, {:.2f})",
                mission_id, radius, start_pose.x, start_pose.y)

    # Refines the row /start already created with the exact rasterized area.
    repo.start_mission(mission_id, body.prompt, area_spec, time.time_ns())
    planner.coverage.update(start_pose)
    _push_coverage_overlay(planner, grid, target)  # show the search area up front

    stats = await explore(
        planner, nav, pose_source,
        max_speed=0.6,
        max_waypoints=settings.mission_max_waypoints,
        max_duration_s=600.0,
        blocked_wait_s=0.0,
    )

    # Final render so the map shows the last position and full coverage
    _push_coverage_overlay(planner, grid, target)

    repo.finish_mission(
        mission_id,
        finished_ns=time.time_ns(),
        coverage_fraction=planner.coverage.fraction(),
        distance_m=pose_source.distance_traveled,
        n_waypoints=stats.n_waypoints,
        n_blocked=stats.n_blocked,
        report_json="{}",
    )
    repo.close()

    logger.info(
        "Sim beendet: {} | Coverage {:.0%} | {} Waypoints | {:.1f} m",
        stats.stop_reason,
        planner.coverage.fraction(),
        stats.n_waypoints,
        pose_source.distance_traveled,
    )


# ── Detection-test runner ─────────────────────────────────────────────────────

async def _run_detection_test(body: "StartBody", mission_id: str) -> None:
    """Validation worker only — no navigation, no robot needed.

    Subscribes to litter/tracked + litter/frame from the running detector,
    crops stable tracks, sends to Ollama for validation, and saves results.
    All findings land at robot position (0, 0) since there is no real pose.
    """
    import litter_ui.zenoh_state as zstate

    zstate.clear_coverage_overlay()  # no map coverage in the camera-only test

    settings = AgentSettings()
    repo = _open_repo(settings)

    # Dummy pose — robot sits at origin. Findings still get crop images saved.
    pose_source = FakePoseSource(Pose2D(x=0.0, y=0.0, theta=0.0))

    # Separate Zenoh session so the validation worker can manage its own subs.
    az = AsyncZenoh(settings.zenoh_config())

    if not settings.ollama_api_key:
        logger.warning(
            "OLLAMA_API_KEY nicht gesetzt — Findings werden als 'error' gespeichert, "
            "Bilder werden trotzdem gesichert."
        )

    validation_agent = build_validation_agent(settings)
    worker = DetectionValidationWorker(
        az,
        pose_source,
        repo,
        make_validate_fn(validation_agent, settings),
        settings,
        mission_id,
    )

    repo.start_mission(mission_id, body.prompt, None, time.time_ns())
    worker.start()
    logger.info(
        "Kamera-Test {} gestartet — halte Müll ≥{}× vor die Kamera und warte.",
        mission_id,
        settings.validation_min_observations,
    )

    try:
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Draining validation queue…")
        await worker.stop(drain_timeout_s=30.0)
        repo.finish_mission(
            mission_id,
            finished_ns=time.time_ns(),
            coverage_fraction=0.0,
            distance_m=0.0,
            n_waypoints=0,
            n_blocked=0,
            report_json="{}",
        )
        repo.close()
        az.close()


# ── Request/response models ───────────────────────────────────────────────────

class StartBody(BaseModel):
    prompt: str
    circle_radius_m: float | None = None
    sim_mode: bool = False
    detection_test: bool = False


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/start")
async def mission_start(body: StartBody) -> dict:
    global _running_mission, _current_mission_id, _sink_id, _loop

    if _running_mission is not None and not _running_mission.done():
        raise HTTPException(status_code=409, detail="Eine Mission läuft bereits")

    _loop = asyncio.get_running_loop()
    _mission_log.clear()
    _remove_sink()
    _sink_id = _setup_sink()

    # Open the mission's board *before* the run starts: mint the id, write the
    # row, publish it as the current mission. The UI switches to it immediately
    # and so drops the previous run's findings, instead of staring at stale
    # detections until this one finishes.
    mission_id = _new_mission_id(body)
    area_spec = _ui_area_spec(body)
    repo = _open_repo(AgentSettings())
    repo.start_mission(mission_id, body.prompt, area_spec, time.time_ns())
    repo.close()
    _current_mission_id = mission_id

    async def _run() -> None:
        global _sink_id
        try:
            if body.detection_test:
                await _run_detection_test(body, mission_id)
            elif body.sim_mode:
                await _run_sim(body, mission_id)
            else:
                import litter_ui.zenoh_state as zstate

                zstate.clear_coverage_overlay()

                def _on_coverage(planner: ExplorationPlanner, _pose: Pose2D) -> None:
                    # grid/target live on the planner; live pose+path already
                    # stream to the UI from zenoh_state's own subscriptions.
                    _push_coverage_overlay(planner, planner.grid, planner.target_mask)

                controller = MissionController()
                await controller.run(
                    body.prompt,
                    area_spec=area_spec,
                    mission_id=mission_id,
                    on_coverage=_on_coverage,
                )
        except asyncio.CancelledError:
            _emit_line("Mission wurde gestoppt")
        except Exception as exc:
            _emit_line(f"Mission-Fehler: {exc}")
        finally:
            _emit_line("__END__")
            _remove_sink()

    _running_mission = asyncio.create_task(_run())
    return {"status": "started", "sim": body.sim_mode, "mission_id": mission_id}


@router.post("/stop")
async def mission_stop() -> dict:
    if _running_mission is not None and not _running_mission.done():
        _running_mission.cancel()
        return {"status": "stopping"}
    return {"status": "not_running"}


@router.get("/status")
async def mission_status() -> dict:
    running = _running_mission is not None and not _running_mission.done()
    return {
        "running": running,
        "mission_id": _current_mission_id,
        "log_tail": list(_mission_log)[-50:],
    }


@router.get("/log")
async def mission_log_sse() -> StreamingResponse:
    async def _generate():
        if _running_mission is None or _running_mission.done():
            for line in list(_mission_log):
                yield f"data: {line}\n\n"
            yield "data: __END__\n\n"
            return

        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        _log_subscribers.add(q)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(q.get(), timeout=25.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {line}\n\n"
                if line == "__END__":
                    break
        finally:
            _log_subscribers.discard(q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
