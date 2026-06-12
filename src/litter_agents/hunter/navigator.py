from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from litter_agents.interfaces.robodog import Pose2D


class NavResult(StrEnum):
    ARRIVED = "arrived"
    BLOCKED = "blocked"
    FAILED = "failed"
    TIMEOUT = "timeout"


class NavInterface(Protocol):
    """Waypoint executor. Implemented by ZenohNavClient (real robot) and FakeNav (sim)."""

    async def goto(
        self, target: Pose2D, max_speed: float
    ) -> tuple[NavResult, Pose2D | None]:
        """Drive a straight line to ``target``; resolves on a terminal state.

        Returns the result and the last pose reported by the executor (the
        stall position when blocked, None if no pose was ever reported).
        """
        ...

    async def halt(self) -> None:
        """Stop the robot and cancel any active request."""
        ...


class PoseLatest(Protocol):
    """Minimal pose feed the exploration loop needs."""

    @property
    def latest(self) -> Pose2D | None: ...
