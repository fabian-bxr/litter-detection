"""Offline-only navigation simulator — no Zenoh, no real robot."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Callable

from ..interfaces.robodog import Pose2D, NavigationState


class FakePoseSource:
    """Simulated pose source with a ring-buffer history for pose_at(ts_ns)."""

    def __init__(self, initial_pose: Pose2D | None = None) -> None:
        self._pose = initial_pose or Pose2D(x=0.0, y=0.0, theta=0.0)
        self._ts_ns: int = time.time_ns()
        self._history: deque[tuple[int, Pose2D]] = deque(maxlen=1000)
        self._history.append((self._ts_ns, self._pose))

    def _set(self, pose: Pose2D) -> None:
        self._pose = pose
        self._ts_ns = time.time_ns()
        self._history.append((self._ts_ns, pose))

    def current(self) -> Pose2D:
        return self._pose

    def current_ns(self) -> int:
        return self._ts_ns

    def pose_at(self, ts_ns: int) -> Pose2D:
        best, best_dt = self._pose, abs(self._ts_ns - ts_ns)
        for t, p in self._history:
            dt = abs(t - ts_ns)
            if dt < best_dt:
                best_dt, best = dt, p
        return best


class FakeNav:
    """Straight-line navigation simulator.

    blocked_discs: list of (x, y, radius_m) — entering triggers BLOCKED.
    on_step: optional callback(pose) called at each simulation step
             (use to feed CoverageTracker during traversal).
    """

    def __init__(
        self,
        pose_source: FakePoseSource,
        speed: float = 0.5,
        dt: float = 0.05,
        blocked_discs: list[tuple[float, float, float]] | None = None,
        on_step: Callable[[Pose2D], None] | None = None,
    ) -> None:
        self._pose = pose_source
        self._speed = speed
        self._dt = dt
        self._blocked_discs = blocked_discs or []
        self._on_step = on_step

    def _in_blocked_disc(self, x: float, y: float) -> bool:
        for bx, by, br in self._blocked_discs:
            if math.sqrt((x - bx) ** 2 + (y - by) ** 2) <= br:
                return True
        return False

    async def goto(
        self,
        target: Pose2D,
        max_speed: float = 0.4,
        must_stop: bool = True,
    ) -> NavigationState:
        speed = min(self._speed, max_speed)
        current = self._pose.current()

        dx = target.x - current.x
        dy = target.y - current.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 0.01:
            return NavigationState.ARRIVED_FINAL

        heading = math.atan2(dy, dx)
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        traveled = 0.0

        while traveled < dist:
            step = min(speed * self._dt, dist - traveled)
            traveled += step
            nx = current.x + cos_h * traveled
            ny = current.y + sin_h * traveled

            if self._in_blocked_disc(nx, ny):
                stall = Pose2D(x=nx, y=ny, theta=heading)
                self._pose._set(stall)
                if self._on_step:
                    self._on_step(stall)
                return NavigationState.BLOCKED

            new_pose = Pose2D(x=nx, y=ny, theta=heading)
            self._pose._set(new_pose)
            if self._on_step:
                self._on_step(new_pose)
            await asyncio.sleep(0)  # yield to event loop

        final = Pose2D(x=target.x, y=target.y, theta=heading)
        self._pose._set(final)
        if self._on_step:
            self._on_step(final)
        return NavigationState.ARRIVED_FINAL

    async def halt(self) -> None:
        pass
