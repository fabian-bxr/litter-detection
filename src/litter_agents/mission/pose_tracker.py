"""Tracks robot pose from Zenoh — latest value + ring-buffer history."""

from __future__ import annotations

import asyncio
import math
from collections import deque

from ..interfaces.robodog import OdometryState, Pose2D
from ..zenoh_bridge import AsyncZenoh

POSE_TOPIC = "robodog/localization/pose"


class ZenohPoseTracker:
    """Subscribes to robodog/localization/pose and maintains:

    - latest pose (always available after wait_first)
    - ring-buffer history for pose_at(ts_ns) nearest-match
    - distance integrator for mission reporting
    """

    def __init__(self, bridge: AsyncZenoh, history_size: int = 300) -> None:
        self._q = bridge.subscribe_queue(POSE_TOPIC, maxsize=200)
        self._latest: Pose2D | None = None
        self._latest_ts_ns: int = 0
        self._history: deque[tuple[int, Pose2D]] = deque(maxlen=history_size)
        self._distance_m: float = 0.0
        self._last_pose: Pose2D | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the background processing task — call from within an asyncio loop."""
        self._task = asyncio.create_task(self._process(), name="pose-tracker")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _process(self) -> None:
        while True:
            raw = await self._q.get()
            try:
                odom = OdometryState.model_validate_json(raw)
            except Exception:
                continue
            pose = odom.to_pose2d()
            ts_ns = int(odom.timestamp.timestamp() * 1e9)
            self._history.append((ts_ns, pose))
            if self._last_pose is not None:
                dx = pose.x - self._last_pose.x
                dy = pose.y - self._last_pose.y
                self._distance_m += math.sqrt(dx * dx + dy * dy)
            self._last_pose = pose
            self._latest = pose
            self._latest_ts_ns = ts_ns

    async def wait_first(self, timeout_s: float = 10.0) -> Pose2D:
        """Block until the first pose arrives. Raises RuntimeError on timeout."""
        deadline = asyncio.get_event_loop().time() + timeout_s
        while self._latest is None:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise RuntimeError(
                    "Timed out waiting for robot pose — is robodog running "
                    f"and publishing on {POSE_TOPIC}?"
                )
            await asyncio.sleep(0.1)
        return self._latest

    def current(self) -> Pose2D:
        return self._latest or Pose2D(x=0.0, y=0.0, theta=0.0)

    def pose_at(self, ts_ns: int) -> Pose2D:
        """Nearest-match pose from ring-buffer history."""
        if not self._history:
            return self.current()
        best, best_dt = self.current(), float("inf")
        for t, p in self._history:
            dt = abs(t - ts_ns)
            if dt < best_dt:
                best_dt, best = dt, p
        return best

    @property
    def distance_traveled_m(self) -> float:
        return self._distance_m
