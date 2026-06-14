"""Wire contracts copied from robodog-digipro/src/interfaces/.

Source: C:\\Users\\Dominik\\PycharmProjects\\robodog-digipro
Do NOT import from that repo — copy only. Keep in sync manually.
Verified: 2026-06-13
"""

from __future__ import annotations

import base64
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation around Z) from a (qx, qy, qz, qw) quaternion."""
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

class Pose2D(BaseModel):
    x: float
    y: float
    theta: float

    def __add__(self, other: Pose2D) -> Pose2D:
        return Pose2D(
            x=self.x + other.x,
            y=self.y + other.y,
            theta=normalize_angle(self.theta + other.theta),
        )

    def __sub__(self, other: Pose2D) -> Pose2D:
        return Pose2D(
            x=self.x - other.x,
            y=self.y - other.y,
            theta=normalize_angle(self.theta - other.theta),
        )

    def __abs__(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def distance_to(self, other: Pose2D) -> float:
        return abs(self - other)

    def bearing_to(self, other: Pose2D) -> float:
        diff = other - self
        return math.atan2(diff.y, diff.x)


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------

class OdometryState(BaseModel):
    """Robot pose from onboard SLAM / odometry — topic: robodog/localization/pose."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    quaternion: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0],
        description="[qx, qy, qz, qw]",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_pose2d(self) -> Pose2D:
        qx, qy, qz, qw = self.quaternion
        return Pose2D(x=self.x, y=self.y, theta=quaternion_to_yaw(qx, qy, qz, qw))

    @classmethod
    def from_raw(cls, message: dict[str, Any]) -> OdometryState | None:
        try:
            data = message["data"]
            stamp = data["header"]["stamp"]
            pos = data["pose"]["position"]
            ori = data["pose"]["orientation"]
            return cls(
                x=pos["x"],
                y=pos["y"],
                z=pos["z"],
                quaternion=[ori["x"], ori["y"], ori["z"], ori["w"]],
                timestamp=datetime.fromtimestamp(
                    stamp["sec"] + stamp["nanosec"] / 1e9, tz=timezone.utc
                ),
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class NavigationSegment(BaseModel):
    """A single straight-line segment for the pure-pursuit executor."""

    target: Pose2D
    max_speed: float | None = None
    allowed_deviation: float = 0.15
    allowed_orientation_deviation: float = 0.1
    must_stop: bool = True
    orientation_at_target: float | None = None
    rotation_allowed_on_segment: bool = True


class NavigationRequest(BaseModel):
    """Published to nav/request — a new request preempts the running one."""

    request_id: str
    segments: list[NavigationSegment]
    lookahead_segments: int = 1


class NavigationState(str, Enum):
    IDLE = "idle"
    FOLLOWING = "following"
    ARRIVED_SEGMENT = "arrived_segment"
    ARRIVED_FINAL = "arrived_final"
    BLOCKED = "blocked"
    FAILED = "failed"


class NavigationStatus(BaseModel):
    """Received from nav/status at ~2 Hz."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: NavigationState
    current_pose: Pose2D | None = None
    distance_to_target: float | None = None
    distance_to_final: float | None = None
    current_segment_index: int | None = None
    request_id: str | None = None
    lookahead_point: Pose2D | None = None


# ---------------------------------------------------------------------------
# Occupancy grid  (topic: robodog/map/occupancy)
# ---------------------------------------------------------------------------

class OccupancyGrid(BaseModel):
    """ROS-style nav_msgs/OccupancyGrid. Values: -1 unknown / 0 free / 100 occupied.

    data is base64-encoded row-major int8 (height × width).
    Use from_array / to_array to convert to/from numpy.
    """

    width: int
    height: int
    resolution: float = Field(default=0.05, gt=0.0)
    origin_x: float = 0.0
    origin_y: float = 0.0
    frame_id: str = "world"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: str  # base64-encoded int8

    @classmethod
    def from_array(
        cls,
        grid: np.ndarray,
        *,
        resolution: float,
        origin_x: float,
        origin_y: float,
        frame_id: str = "world",
        timestamp: datetime | None = None,
    ) -> OccupancyGrid:
        if grid.dtype != np.int8:
            grid = grid.astype(np.int8)
        if grid.ndim != 2:
            raise ValueError(f"grid must be 2D, got shape {grid.shape}")
        height, width = grid.shape
        return cls(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            frame_id=frame_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            data=base64.b64encode(grid.tobytes()).decode("ascii"),
        )

    def to_array(self) -> np.ndarray:
        raw = base64.b64decode(self.data)
        return np.frombuffer(raw, dtype=np.int8).reshape(self.height, self.width)
