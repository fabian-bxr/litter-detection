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
from litter_agents.config import AgentSettings
from litter_agents.hunter.explore import explore
from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.planner import ExplorationPlanner
from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.provider import FileMapProvider
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


# ── Sim runner ────────────────────────────────────────────────────────────────

async def _run_sim(body: "StartBody") -> None:
    """Exploration sim with FakePoseSource/FakeNav; broadcasts to /ws/state."""
    import litter_ui.zenoh_state as zstate

    global _current_mission_id

    settings = AgentSettings()
    mission_id = time.strftime("%Y%m%d-%H%M%S") + "-sim-" + uuid.uuid4().hex[:4]
    params = HunterParams.from_settings(settings)

    map_yaml = Path(settings.map_yaml_path)
    if not map_yaml.is_absolute():
        map_yaml = _REPO_ROOT / map_yaml

    grid = await FileMapProvider(str(map_yaml)).load()
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

    tick_count = 0

    def on_tick(pose: Pose2D) -> None:
        nonlocal tick_count
        zstate.pose_latest = pose
        zstate.path_history.append((pose.x, pose.y))
        tick_count += 1
        if tick_count % 5 == 0:  # throttle broadcast to every 5 ticks
            zstate._broadcast()

    nav = FakeNav(
        pose_source,
        on_tick=on_tick,
        grid=grid,
        blocked_inflated=planner.blocked_inflated(),
        skip_start_m=params.robot_radius_m,
    )

    db_path = Path(settings.findings_db_path)
    if not db_path.is_absolute():
        db_path = _REPO_ROOT / db_path
    repo = FindingsRepository(db_path)

    logger.info("Sim {} gestartet — Radius {:.1f} m, Startpose ({:.2f}, {:.2f})",
                mission_id, radius, start_pose.x, start_pose.y)

    repo.start_mission(mission_id, body.prompt, area_spec, time.time_ns())
    planner.coverage.update(start_pose)

    stats = await explore(
        planner, nav, pose_source,
        max_speed=0.6,
        max_waypoints=settings.mission_max_waypoints,
        max_duration_s=600.0,
        blocked_wait_s=0.0,
    )

    # Final broadcast so the map shows the last position
    zstate._broadcast()

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

    _current_mission_id = mission_id
    logger.info(
        "Sim beendet: {} | Coverage {:.0%} | {} Waypoints | {:.1f} m",
        stats.stop_reason,
        planner.coverage.fraction(),
        stats.n_waypoints,
        pose_source.distance_traveled,
    )


# ── Detection-test runner ─────────────────────────────────────────────────────

async def _run_detection_test(body: "StartBody") -> None:
    """Validation worker only — no navigation, no robot needed.

    Subscribes to litter/tracked + litter/frame from the running detector,
    crops stable tracks, sends to Ollama for validation, and saves results.
    All findings land at robot position (0, 0) since there is no real pose.
    """
    global _current_mission_id

    settings = AgentSettings()
    mission_id = time.strftime("%Y%m%d-%H%M%S") + "-camtest-" + uuid.uuid4().hex[:4]

    db_path = Path(settings.findings_db_path)
    if not db_path.is_absolute():
        db_path = _REPO_ROOT / db_path

    repo = FindingsRepository(db_path)

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
        _current_mission_id = mission_id


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

    async def _run() -> None:
        global _sink_id, _current_mission_id
        try:
            if body.detection_test:
                await _run_detection_test(body)
            elif body.sim_mode:
                await _run_sim(body)
            else:
                area_spec: SearchAreaSpec | None = None
                if body.circle_radius_m is not None:
                    area_spec = SearchAreaSpec(
                        shape="circle",
                        radius_m=body.circle_radius_m,
                        rationale="Provided by UI",
                    )
                controller = MissionController()
                report = await controller.run(body.prompt, area_spec=area_spec)
                _current_mission_id = report.mission_id
        except asyncio.CancelledError:
            _emit_line("Mission wurde gestoppt")
        except Exception as exc:
            _emit_line(f"Mission-Fehler: {exc}")
        finally:
            _emit_line("__END__")
            _remove_sink()

    _running_mission = asyncio.create_task(_run())
    return {"status": "started", "sim": body.sim_mode}


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
