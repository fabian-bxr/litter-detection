"""NavInterface Protocol + ZenohNavClient implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ..interfaces.robodog import (
    NavigationRequest,
    NavigationSegment,
    NavigationState,
    NavigationStatus,
    Pose2D,
)
from ..zenoh_bridge import AsyncZenoh

NAV_REQUEST_TOPIC = "nav/request"
NAV_STATUS_TOPIC = "nav/status"


@runtime_checkable
class NavInterface(Protocol):
    async def goto(
        self,
        target: Pose2D,
        max_speed: float = 0.4,
        must_stop: bool = True,
    ) -> NavigationState: ...

    async def halt(self) -> None: ...


class ZenohNavClient:
    """Sends NavigationRequest over Zenoh and waits for a terminal NavigationState.

    Timeout = max(20 s, 4 × estimated_distance / speed).  Both ARRIVED_FINAL
    and ARRIVED_SEGMENT are treated as success.  Silence > timeout → BLOCKED.
    """

    def __init__(
        self,
        bridge: AsyncZenoh,
        current_pose_fn: "Callable[[], Pose2D]",
        max_speed: float = 0.4,
    ) -> None:
        self._bridge = bridge
        self._current_pose_fn = current_pose_fn
        self._default_speed = max_speed
        self._status_q = bridge.subscribe_queue(NAV_STATUS_TOPIC, maxsize=200)
        self._req_counter = 0

    async def goto(
        self,
        target: Pose2D,
        max_speed: float | None = None,
        must_stop: bool = True,
    ) -> NavigationState:
        speed = max_speed if max_speed is not None else self._default_speed
        self._req_counter += 1
        request_id = f"lm-{int(time.time()*1000)}-{self._req_counter}"

        # Timeout based on estimated distance
        current = self._current_pose_fn()
        dist = current.distance_to(target)
        timeout_s = max(20.0, 4.0 * dist / max(speed, 0.05))

        request = NavigationRequest(
            request_id=request_id,
            segments=[
                NavigationSegment(
                    target=target,
                    max_speed=speed,
                    must_stop=must_stop,
                )
            ],
        )

        # Drain stale status messages before sending the request
        while not self._status_q.empty():
            try:
                self._status_q.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._bridge.publish_json(NAV_REQUEST_TOPIC, request)

        deadline = asyncio.get_event_loop().time() + timeout_s
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return NavigationState.BLOCKED
            try:
                raw = await asyncio.wait_for(
                    self._status_q.get(), timeout=min(remaining, 2.0)
                )
            except asyncio.TimeoutError:
                continue  # poll again until deadline expires

            try:
                status = NavigationStatus.model_validate_json(raw)
            except Exception:
                continue

            if status.request_id != request_id:
                continue

            if status.state in (
                NavigationState.ARRIVED_FINAL,
                NavigationState.ARRIVED_SEGMENT,
            ):
                return NavigationState.ARRIVED_FINAL
            if status.state in (NavigationState.BLOCKED, NavigationState.FAILED):
                return status.state

    async def halt(self) -> None:
        # Preempt the active goal by sending the robot's current position as
        # target — avoids crashing nav nodes that don't handle 0-segment paths.
        current = self._current_pose_fn()
        request = NavigationRequest(
            request_id=f"lm-halt-{int(time.time()*1000)}",
            segments=[
                NavigationSegment(
                    target=current,
                    max_speed=self._default_speed,
                    must_stop=True,
                )
            ],
        )
        self._bridge.publish_json(NAV_REQUEST_TOPIC, request)
