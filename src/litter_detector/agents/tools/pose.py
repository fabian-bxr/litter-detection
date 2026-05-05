from __future__ import annotations

import json
import math
import threading
from datetime import datetime
from typing import Protocol

import zenoh
from loguru import logger

from litter_detector.agents.models import Pose
from litter_detector.config import Settings


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (Z-axis rotation) from a quaternion, radians."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class PoseSource(Protocol):
    def get(self, timeout: float = 1.0) -> Pose | None: ...


class PoseClient:
    """Subscribes to robot odometry and caches the latest pose."""

    def __init__(self, session: zenoh.Session | None = None) -> None:
        self._owns_session = session is None
        self._session = session if session is not None else zenoh.open(Settings.zenoh_config())
        topic = Settings.topics().robot.odometry
        self._lock = threading.Lock()
        self._latest: Pose | None = None
        self._event = threading.Event()
        self._sub = self._session.declare_subscriber(topic, self._on_msg)
        logger.info(f"PoseClient subscribed to {topic}")

    def _on_msg(self, sample: zenoh.Sample) -> None:
        try:
            data = json.loads(bytes(sample.payload).decode("utf-8"))
            qx, qy, qz, qw = data["quaternion"]
            stamp_str = data.get("timestamp")
            stamp = datetime.fromisoformat(stamp_str.replace("Z", "+00:00")) if stamp_str else None
            pose = Pose(
                x=float(data["x"]),
                y=float(data["y"]),
                theta=quaternion_to_yaw(qx, qy, qz, qw),
                stamp=stamp,
            )
        except Exception as e:
            logger.warning(f"PoseClient: failed to parse odometry message: {e}")
            return
        with self._lock:
            self._latest = pose
        self._event.set()

    def get(self, timeout: float = 1.0) -> Pose | None:
        """Return the latest pose, blocking up to `timeout` for the first message."""
        with self._lock:
            if self._latest is not None:
                return self._latest
        if not self._event.wait(timeout):
            return None
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._sub.undeclare()
        if self._owns_session:
            self._session.close()


class FakePoseClient:
    """In-memory pose source for tests."""

    def __init__(self, pose: Pose | None = None) -> None:
        self._pose = pose

    def set(self, pose: Pose) -> None:
        self._pose = pose

    def get(self, timeout: float = 1.0) -> Pose | None:
        return self._pose

    def close(self) -> None:
        return
