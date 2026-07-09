"""Module-level Zenoh singleton for the UI backend.

Initialised once in the FastAPI lifespan; all WebSocket handlers read from
the shared queues/state it exposes. Startup failures (router unreachable,
bad config) are caught and logged — the app continues without live data.

Thread model: Zenoh callbacks fire on Zenoh runtime threads; AsyncZenoh
routes them into the asyncio event loop via call_soon_threadsafe.  Every
mutation below therefore runs single-threaded in the loop — no locks needed.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque

from loguru import logger

from litter_agents.config import (
    NAV_REQUEST_TOPIC,
    NAV_STATUS_TOPIC,
    ROBODOG_POSE_TOPIC,
    build_zenoh_config,
)
from litter_agents.interfaces.robodog import (
    NavigationRequest,
    NavigationStatus,
    OdometryState,
    Pose2D,
)
from litter_agents.zenoh_bridge import AsyncZenoh
from litter_detector.config import TOPICS

# ── Public state ──────────────────────────────────────────────────────────────

az: AsyncZenoh | None = None

# Camera: raw JPEG bytes, drop-oldest queue (4 frames ≈ 270 ms at 15 fps).
camera_queue: asyncio.Queue[bytes] | None = None

# Detection overlay: masked_frame from the detector (None when detector not running).
detection_queue: asyncio.Queue[bytes] | None = None

# Robot state (mutated only on the asyncio event loop).
pose_latest: Pose2D | None = None
path_history: deque[tuple[float, float]] = deque(maxlen=10_000)
nav_status_latest: dict | None = None
planned_path: list[tuple[float, float]] = []

# Active /ws/state connections — each holds pre-serialised JSON strings.
state_subscribers: set[asyncio.Queue[str]] = set()


# ── Snapshot & broadcast ──────────────────────────────────────────────────────


def state_snapshot() -> dict:
    """Current state as a plain dict, safe to JSON-serialise."""
    return {
        "pose": (
            {"x": pose_latest.x, "y": pose_latest.y, "theta": pose_latest.theta}
            if pose_latest is not None
            else None
        ),
        "path_history": list(path_history),
        "planned_path": planned_path,
        "nav_status": nav_status_latest,
    }


def _broadcast() -> None:
    """Serialise current state once and push to every subscriber queue."""
    if not state_subscribers:
        return
    data = json.dumps(state_snapshot())
    for q in state_subscribers:
        if q.full():
            try:
                q.get_nowait()  # drop oldest to make room
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass  # guard; shouldn't happen after get_nowait


# ── Zenoh event handlers (called on the asyncio event loop) ──────────────────


def _on_pose(odo: OdometryState) -> None:
    global pose_latest
    pose_latest = odo.to_pose2d()
    path_history.append((pose_latest.x, pose_latest.y))
    _broadcast()


def _on_nav_status(status: NavigationStatus) -> None:
    global nav_status_latest
    nav_status_latest = status.model_dump(mode="json")
    _broadcast()


def _on_nav_request(request: NavigationRequest) -> None:
    global planned_path
    planned_path = [(seg.target.x, seg.target.y) for seg in request.segments]
    _broadcast()


# ── Lifecycle ─────────────────────────────────────────────────────────────────


async def startup() -> None:
    """Open Zenoh session and register all subscriptions.

    Must be called from within a running event loop (FastAPI lifespan).
    Any exception is caught so the app starts even without a Zenoh router.
    """
    global az, camera_queue, detection_queue
    try:
        _az = AsyncZenoh(build_zenoh_config())

        # Camera frames — drop-oldest queue, subscriber kept alive in _az.
        camera_queue = _az.subscribe_queue(
            TOPICS.camera.frame,
            lambda s: s.payload.to_bytes(),
            maxsize=4,
        )

        # Detection overlay (litter/masked_frame) from the detector process.
        detection_queue = _az.subscribe_queue(
            TOPICS.detection.masked_frame,
            lambda s: s.payload.to_bytes(),
            maxsize=4,
        )

        # Robot pose → path history + state broadcast.
        _az.subscribe(
            ROBODOG_POSE_TOPIC,
            lambda s: OdometryState.model_validate_json(s.payload.to_bytes()),
            _on_pose,
        )

        # Navigation status (idle / following / blocked …).
        _az.subscribe(
            NAV_STATUS_TOPIC,
            lambda s: NavigationStatus.model_validate_json(s.payload.to_bytes()),
            _on_nav_status,
        )

        # Planned multi-leg path from the NBV planner.
        _az.subscribe(
            NAV_REQUEST_TOPIC,
            lambda s: NavigationRequest.model_validate_json(s.payload.to_bytes()),
            _on_nav_request,
        )

        az = _az
        logger.info(
            "Zenoh connected — subscribed to camera / pose / nav_status / nav_request"
        )
    except Exception as exc:
        logger.warning("Zenoh unavailable — live data disabled: {}", exc)
        az = None
        camera_queue = None
        detection_queue = None


def shutdown() -> None:
    global az
    if az is not None:
        try:
            az.close()
        except Exception:
            logger.opt(exception=True).debug("Error closing Zenoh session")
        az = None
