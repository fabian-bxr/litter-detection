from __future__ import annotations

import asyncio
import math
from collections.abc import Callable

import numpy as np

from litter_agents.hunter.navigator import NavResult
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap


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
    en-route poses via ``on_tick``). The walk stops and returns BLOCKED at the
    stall pose when a step enters one of ``blocked_discs`` (x, y, radius) — an
    *unmapped* obstacle the planner couldn't avoid — or, when ``grid`` /
    ``blocked_inflated`` are given, a known map obstacle. The latter makes the
    sim actually verify that planned paths are traversable instead of letting
    the robot glide through walls.
    """

    def __init__(
        self,
        pose_source: FakePoseSource,
        *,
        tick_s: float = 0.1,
        time_scale: float = 1000.0,
        blocked_discs: list[tuple[float, float, float]] | None = None,
        on_tick: Callable[[Pose2D], None] | None = None,
        grid: GridMap | None = None,
        blocked_inflated: np.ndarray | None = None,
        skip_start_m: float = 0.0,
    ) -> None:
        self.pose_source = pose_source
        self.tick_s = tick_s
        self.time_scale = time_scale
        self.blocked_discs = blocked_discs or []
        self.on_tick = on_tick
        self._grid = grid
        self._blocked_inflated = blocked_inflated
        # Ignore map-inflation collisions within this radius of a leg's start —
        # the robot legitimately stands in its own inflation overlap near walls,
        # and the planner skips the same band (pathing._straight_clear).
        self._skip_start_m = skip_start_m

    def _disc_blocked(self, pose: Pose2D) -> bool:
        return any(
            (pose.x - x) ** 2 + (pose.y - y) ** 2 <= r**2
            for x, y, r in self.blocked_discs
        )

    def _map_blocked(self, pose: Pose2D) -> bool:
        if self._grid is None or self._blocked_inflated is None:
            return False
        row, col = self._grid.world_to_grid(pose.x, pose.y)
        h, w = self._blocked_inflated.shape
        return not (0 <= row < h and 0 <= col < w) or bool(
            self._blocked_inflated[row, col]
        )

    async def goto(
        self, target: Pose2D, max_speed: float
    ) -> tuple[NavResult, Pose2D | None]:
        return await self.goto_path([target], max_speed)

    async def goto_path(
        self, path: list[Pose2D], max_speed: float
    ) -> tuple[NavResult, Pose2D | None]:
        last: Pose2D | None = self.pose_source.latest
        for leg in path:
            result, last = await self._walk_leg(leg, max_speed)
            if result is not NavResult.ARRIVED:
                return result, last
        return NavResult.ARRIVED, last

    async def _walk_leg(
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
            near_start = distance * frac < self._skip_start_m
            if self._disc_blocked(step) or (
                not near_start and self._map_blocked(step)
            ):
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
