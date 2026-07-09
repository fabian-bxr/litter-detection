from __future__ import annotations

import time
import uuid
from asyncio import QueueEmpty, TimeoutError, wait_for

import zenoh
from loguru import logger

from litter_agents.config import NAV_REQUEST_TOPIC, NAV_STATUS_TOPIC, AgentSettings
from litter_agents.hunter.navigator import NavResult
from litter_agents.interfaces.robodog import (
    NavigationRequest,
    NavigationSegment,
    NavigationState,
    NavigationStatus,
    Pose2D,
)
from litter_agents.mission.pose_tracker import PoseSource
from litter_agents.zenoh_bridge import Bridge


def _decode_status(sample: zenoh.Sample) -> NavigationStatus:
    return NavigationStatus.model_validate_json(sample.payload.to_bytes())


def _path_length(start: Pose2D | None, path: list[Pose2D]) -> float:
    """Total polyline length from ``start`` through every leg (for the timeout)."""
    total = 0.0
    prev = start or (path[0] if path else None)
    for leg in path:
        if prev is not None:
            total += prev.distance_to(leg)
        prev = leg
    return total


class ZenohNavClient:
    """Waypoint execution against the robodog nav stack.

    One single-segment NavigationRequest per goto — exploration replans after
    every waypoint anyway, and a fresh request cleanly preempts whatever the
    executor is doing. ``orientation_at_target`` is always set to the
    candidate's travel heading: for moving targets pure pursuit arrives
    already facing it (the align step is a no-op), and it is what makes
    zero-distance rotation candidates work at all.
    """

    def __init__(
        self, az: Bridge, pose_source: PoseSource, settings: AgentSettings
    ) -> None:
        self._az = az
        self._pose_source = pose_source
        self._settings = settings
        self._status_queue = az.subscribe_queue(
            NAV_STATUS_TOPIC, _decode_status, maxsize=64
        )

    def _drain(self) -> None:
        while True:
            try:
                self._status_queue.get_nowait()
            except QueueEmpty:
                return

    async def goto(
        self, target: Pose2D, max_speed: float
    ) -> tuple[NavResult, Pose2D | None]:
        return await self.goto_path([target], max_speed)

    async def goto_path(
        self, path: list[Pose2D], max_speed: float
    ) -> tuple[NavResult, Pose2D | None]:
        if not path:
            return NavResult.ARRIVED, self._pose_source.latest
        target = path[-1]
        request_id = uuid.uuid4().hex
        # One request, multiple straight segments: the executor flows through
        # the intermediate corners (must_stop=False) and only halts at the
        # final viewpoint, facing its orientation.
        segments = [
            NavigationSegment(
                target=leg,
                max_speed=max_speed,
                allowed_deviation=self._settings.nav_allowed_deviation,
                must_stop=(i == len(path) - 1),
                orientation_at_target=leg.theta,
            )
            for i, leg in enumerate(path)
        ]
        self._drain()
        self._az.publish_json(
            NAV_REQUEST_TOPIC,
            NavigationRequest(request_id=request_id, segments=segments),
        )

        start_pose = self._pose_source.latest
        distance = _path_length(start_pose, path)
        deadline = time.monotonic() + max(
            20.0, self._settings.nav_goal_timeout_factor * distance / max(max_speed, 0.05)
        )
        last_pose: Pose2D | None = None
        while True:
            try:
                status = await wait_for(
                    self._status_queue.get(), self._settings.nav_status_timeout_s
                )
            except TimeoutError:
                logger.error("No nav/status for {} s — is the nav stack running?",
                             self._settings.nav_status_timeout_s)
                await self.halt()
                return NavResult.TIMEOUT, last_pose
            if status.request_id != request_id:
                continue  # stale status from a previous request
            if status.current_pose is not None:
                last_pose = status.current_pose
            if status.state is NavigationState.ARRIVED_FINAL:
                return NavResult.ARRIVED, last_pose
            if status.state is NavigationState.BLOCKED:
                return NavResult.BLOCKED, last_pose
            if status.state is NavigationState.FAILED:
                return NavResult.FAILED, last_pose
            if time.monotonic() > deadline:
                logger.warning("Goal ({:.2f}, {:.2f}) timed out", target.x, target.y)
                await self.halt()
                return NavResult.TIMEOUT, last_pose

    async def halt(self) -> None:
        """Preempt with a request targeting the current pose — stops the robot."""
        pose = self._pose_source.latest
        if pose is None:
            return
        self._az.publish_json(
            NAV_REQUEST_TOPIC,
            NavigationRequest(
                request_id=uuid.uuid4().hex,
                segments=[
                    NavigationSegment(
                        target=pose,
                        max_speed=0.1,
                        allowed_deviation=self._settings.nav_allowed_deviation,
                        must_stop=True,
                    )
                ],
            ),
        )
