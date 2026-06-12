from __future__ import annotations

import asyncio
import math
from collections.abc import Callable

from litter_agents.hunter.navigator import NavResult
from litter_agents.interfaces.robodog import Pose2D


class FakePoseSource:
    """In-memory stand-in for the live localization feed."""

    def __init__(self, start: Pose2D) -> None:
        self._pose = start
        self.distance_traveled = 0.0

    @property
    def latest(self) -> Pose2D | None:
        return self._pose

    def set(self, pose: Pose2D) -> None:
        self.distance_traveled += pose.distance_to(self._pose)
        self._pose = pose

    async def wait_first(self, timeout: float) -> Pose2D:
        return self._pose

    def pose_at(self, wall_ts_ns: int) -> Pose2D | None:
        return self._pose


class FakeNav:
    """Straight-line waypoint executor mirroring robodog semantics.

    Interpolates the pose toward the target at ``speed``, feeding every
    intermediate pose into the FakePoseSource (so coverage sees realistic
    en-route poses via ``on_tick``). Entering one of ``blocked_discs``
    (x, y, radius) stops the walk and returns BLOCKED at the stall pose —
    the hook for simulating "free on the map but not actually passable".
    """

    def __init__(
        self,
        pose_source: FakePoseSource,
        *,
        tick_s: float = 0.1,
        time_scale: float = 1000.0,
        blocked_discs: list[tuple[float, float, float]] | None = None,
        on_tick: Callable[[Pose2D], None] | None = None,
    ) -> None:
        self.pose_source = pose_source
        self.tick_s = tick_s
        self.time_scale = time_scale
        self.blocked_discs = blocked_discs or []
        self.on_tick = on_tick

    def _in_blocked_disc(self, pose: Pose2D) -> bool:
        return any(
            (pose.x - x) ** 2 + (pose.y - y) ** 2 <= r**2
            for x, y, r in self.blocked_discs
        )

    async def goto(
        self, target: Pose2D, max_speed: float
    ) -> tuple[NavResult, Pose2D | None]:
        pose = self.pose_source.latest
        assert pose is not None
        distance = pose.distance_to(target)
        if distance < 1e-6:
            # Rotation in place.
            rotated = Pose2D(x=pose.x, y=pose.y, theta=target.theta)
            self.pose_source.set(rotated)
            if self.on_tick:
                self.on_tick(rotated)
            await asyncio.sleep(0)
            return NavResult.ARRIVED, rotated
        heading = pose.bearing_to(target)
        n_ticks = max(1, math.ceil(distance / (max_speed * self.tick_s)))
        prev = Pose2D(x=pose.x, y=pose.y, theta=heading)
        for i in range(1, n_ticks + 1):
            frac = i / n_ticks
            step = Pose2D(
                x=pose.x + (target.x - pose.x) * frac,
                y=pose.y + (target.y - pose.y) * frac,
                theta=heading,
            )
            if self._in_blocked_disc(step):
                self.pose_source.set(prev)
                if self.on_tick:
                    self.on_tick(prev)
                return NavResult.BLOCKED, prev
            self.pose_source.set(step)
            if self.on_tick:
                self.on_tick(step)
            prev = step
            delay = self.tick_s / self.time_scale
            await asyncio.sleep(delay if delay > 1e-4 else 0)
        return NavResult.ARRIVED, prev

    async def halt(self) -> None:
        pass
