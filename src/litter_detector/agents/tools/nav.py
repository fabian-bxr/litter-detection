from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import zenoh
from loguru import logger

from litter_detector.agents.models import Pose
from litter_detector.config import Settings


@dataclass
class NavResult:
    """Terminal status for a navigation request."""

    request_id: str
    state: str  # "arrived_final" | "blocked" | "failed" | "aborted"
    final_pose: Pose | None
    distance_to_target: float | None


_TERMINAL = {"arrived_final", "blocked", "failed"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_single_segment_request(
    request_id: str,
    target: Pose,
    *,
    max_speed: float | None,
    allowed_deviation: float,
    allowed_orientation_deviation: float,
) -> dict:
    return {
        "request_id": request_id,
        "lookahead_segments": 1,
        "segments": [
            {
                "target": {"x": target.x, "y": target.y, "theta": target.theta},
                "max_speed": max_speed,
                "corridor": None,
                "allowed_deviation": allowed_deviation,
                "allowed_orientation_deviation": allowed_orientation_deviation,
                "must_stop": True,
                "orientation_at_target": target.theta,
                "rotation_allowed_on_segment": True,
            }
        ],
    }


class NavSink(Protocol):
    def submit(self, target: Pose, **kwargs) -> str: ...
    def wait_for_terminal(self, request_id: str, timeout: float) -> NavResult | None: ...


class NavClient:
    """Submits NavigationRequests over Zenoh and tracks terminal NavStatus."""

    def __init__(self, session: zenoh.Session | None = None) -> None:
        self._owns_session = session is None
        self._session = session if session is not None else zenoh.open(Settings.zenoh_config())
        topics = Settings.topics().robot
        self._pub = self._session.declare_publisher(
            topics.nav_request, encoding=zenoh.Encoding.APPLICATION_JSON
        )
        self._lock = threading.Lock()
        self._latest_by_id: dict[str, dict] = {}
        self._events: dict[str, threading.Event] = {}
        self._sub = self._session.declare_subscriber(topics.nav_status, self._on_status)
        logger.info(f"NavClient publishing to {topics.nav_request}, subscribing to {topics.nav_status}")

    def _on_status(self, sample: zenoh.Sample) -> None:
        try:
            data = json.loads(bytes(sample.payload).decode("utf-8"))
        except Exception as e:
            logger.warning(f"NavClient: failed to parse nav/status message: {e}")
            return
        rid = data.get("request_id")
        if not rid:
            return
        with self._lock:
            self._latest_by_id[rid] = data
            ev = self._events.get(rid)
        if data.get("state") in _TERMINAL and ev is not None:
            ev.set()

    def submit(
        self,
        target: Pose,
        *,
        max_speed: float | None = None,
        allowed_deviation: float = 0.15,
        allowed_orientation_deviation: float = 0.2,
    ) -> str:
        request_id = str(uuid.uuid4())
        msg = _build_single_segment_request(
            request_id,
            target,
            max_speed=max_speed,
            allowed_deviation=allowed_deviation,
            allowed_orientation_deviation=allowed_orientation_deviation,
        )
        with self._lock:
            self._events[request_id] = threading.Event()
        self._pub.put(json.dumps(msg))
        logger.debug(f"NavClient submitted request {request_id} → ({target.x:.2f},{target.y:.2f},θ={target.theta:.2f})")
        return request_id

    def wait_for_terminal(self, request_id: str, timeout: float) -> NavResult | None:
        with self._lock:
            ev = self._events.get(request_id)
        if ev is None:
            return None
        if not ev.wait(timeout):
            return None
        with self._lock:
            data = self._latest_by_id.get(request_id)
        if data is None:
            return None
        cp = data.get("current_pose") or {}
        final = (
            Pose(x=float(cp["x"]), y=float(cp["y"]), theta=float(cp.get("theta", 0.0)))
            if cp
            else None
        )
        return NavResult(
            request_id=request_id,
            state=str(data.get("state", "failed")),
            final_pose=final,
            distance_to_target=data.get("distance_to_target"),
        )

    def latest_status(self, request_id: str) -> dict | None:
        with self._lock:
            return self._latest_by_id.get(request_id)

    def close(self) -> None:
        self._sub.undeclare()
        self._pub.undeclare()
        if self._owns_session:
            self._session.close()


class FakeNavClient:
    """In-memory nav stub for tests.

    Each `submit()` returns immediately with a programmable terminal state.
    """

    def __init__(self, default_state: str = "arrived_final") -> None:
        self.default_state = default_state
        self.submitted: list[tuple[str, Pose]] = []
        self._results: dict[str, NavResult] = {}
        self.next_state: str | None = None

    def submit(self, target: Pose, **kwargs) -> str:
        rid = str(uuid.uuid4())
        state = self.next_state or self.default_state
        self.next_state = None
        self.submitted.append((rid, target))
        self._results[rid] = NavResult(
            request_id=rid,
            state=state,
            final_pose=target if state == "arrived_final" else None,
            distance_to_target=0.0 if state == "arrived_final" else None,
        )
        return rid

    def wait_for_terminal(self, request_id: str, timeout: float) -> NavResult | None:
        return self._results.get(request_id)

    def latest_status(self, request_id: str) -> dict | None:
        r = self._results.get(request_id)
        if r is None:
            return None
        return {"request_id": r.request_id, "state": r.state}

    def close(self) -> None:
        return
