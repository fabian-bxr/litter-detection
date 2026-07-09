from __future__ import annotations

import asyncio
from collections import deque
from typing import Protocol

import zenoh

from litter_agents.config import ROBODOG_POSE_TOPIC
from litter_agents.interfaces.robodog import OdometryState, Pose2D
from litter_agents.zenoh_bridge import Bridge


class PoseSource(Protocol):
    """What the mission needs from localization (FakePoseSource implements it too)."""

    @property
    def latest(self) -> Pose2D | None: ...

    @property
    def distance_traveled(self) -> float: ...

    async def wait_first(self, timeout: float) -> Pose2D: ...

    def pose_at(self, wall_ts_ns: int) -> Pose2D | None: ...


class ZenohPoseTracker:
    """Live robot pose from ``robodog/localization/pose``.

    Keeps the latest pose, a ~10 s ring buffer for timestamp matching
    (detections carry wall-clock ``timestamp_ns``, as does OdometryState), and
    an integrated travel distance for the mission report.
    """

    # A detection more than this far from any buffered pose gets the latest
    # pose instead — better a slightly stale position than none.
    _MATCH_TOLERANCE_NS = 500_000_000

    def __init__(self, az: Bridge, topic: str = ROBODOG_POSE_TOPIC) -> None:
        self._buffer: deque[tuple[int, Pose2D]] = deque(maxlen=1024)
        self._latest: Pose2D | None = None
        self._distance = 0.0
        self._first = asyncio.Event()
        az.subscribe(topic, self._decode, self._on_pose)

    @staticmethod
    def _decode(sample: zenoh.Sample) -> OdometryState:
        return OdometryState.model_validate_json(sample.payload.to_bytes())

    def _on_pose(self, odo: OdometryState) -> None:
        pose = odo.to_pose2d()
        if self._latest is not None:
            self._distance += pose.distance_to(self._latest)
        self._latest = pose
        self._buffer.append((int(odo.timestamp.timestamp() * 1e9), pose))
        self._first.set()

    @property
    def latest(self) -> Pose2D | None:
        return self._latest

    @property
    def distance_traveled(self) -> float:
        return self._distance

    async def wait_first(self, timeout: float) -> Pose2D:
        await asyncio.wait_for(self._first.wait(), timeout)
        assert self._latest is not None
        return self._latest

    def pose_at(self, wall_ts_ns: int) -> Pose2D | None:
        if not self._buffer:
            return self._latest
        ts, pose = min(self._buffer, key=lambda e: abs(e[0] - wall_ts_ns))
        if abs(ts - wall_ts_ns) > self._MATCH_TOLERANCE_NS:
            return self._latest
        return pose
