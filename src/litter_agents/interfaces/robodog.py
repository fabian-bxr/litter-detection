"""Zenoh wire contracts of the robodog-digipro stack.

Copied (not imported) from robodog-digipro/src/interfaces/{navigation,robot,
occupancy}.py — that repo is reference-only for this project. Only the models
that actually cross the wire between the two systems are kept; if robodog
changes a schema, update the copy here.
"""

from __future__ import annotations

import base64
import math
from datetime import datetime, timezone
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation around Z) from an (qx, qy, qz, qw) quaternion."""
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


class Pose2D(BaseModel):
    x: float
    y: float
    theta: float

    def __add__(self, other: "Pose2D") -> "Pose2D":
        return Pose2D(
            x=self.x + other.x,
            y=self.y + other.y,
            theta=normalize_angle(self.theta + other.theta),
        )

    def __sub__(self, other: "Pose2D") -> "Pose2D":
        return Pose2D(
            x=self.x - other.x,
            y=self.y - other.y,
            theta=normalize_angle(self.theta - other.theta),
        )

    def __abs__(self) -> float:
        """Euclidean distance from origin (ignores theta)."""
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def distance(self) -> float:
        return abs(self)

    @property
    def bearing(self) -> float:
        """Angle from origin to this point."""
        return math.atan2(self.y, self.x)

    def distance_to(self, other: "Pose2D") -> float:
        return abs(self - other)

    def bearing_to(self, other: "Pose2D") -> float:
        return (other - self).bearing


class Corridor(BaseModel):
    """Lateral bounds for path planning on a segment."""

    left_width: float  # meters, positive = left of travel direction
    right_width: float


class NavigationSegment(BaseModel):
    """A single segment to traverse (straight line to ``target``)."""

    target: Pose2D
    max_speed: float | None = None  # m/s, None = robodog default
    corridor: Corridor | None = None
    allowed_deviation: float = 0.15  # meters — how close counts as "arrived"
    allowed_orientation_deviation: float = 0.1  # radians
    must_stop: bool = True
    orientation_at_target: float | None = None  # required heading, radians
    rotation_allowed_on_segment: bool = True


class NavigationRequest(BaseModel):
    """Published to ``nav/request``; preempts any running request."""

    request_id: str  # correlation id echoed in NavigationStatus
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
    """Streamed on ``nav/status`` (~2 Hz and on state change)."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: NavigationState
    current_pose: Pose2D | None = None
    distance_to_target: float | None = None
    distance_to_final: float | None = None
    current_segment_index: int | None = None
    request_id: str | None = None
    lookahead_point: Pose2D | None = None


class OdometryState(BaseModel):
    """Robot pose streamed on ``robodog/localization/pose`` (world frame, meters)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    quaternion: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0],
        description="Orientation as [qx, qy, qz, qw]",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_pose2d(self) -> Pose2D:
        qx, qy, qz, qw = self.quaternion
        return Pose2D(x=self.x, y=self.y, theta=quaternion_to_yaw(qx, qy, qz, qw))


class OccupancyGrid(BaseModel):
    """ROS-style 2D occupancy grid (``nav_msgs/OccupancyGrid`` convention).

    Cell values: -1 unknown, 0 free, 100 occupied. Stored row-major
    (height × width) as base64-encoded int8 bytes; row index increases with
    +y, ``(origin_x, origin_y)`` is the corner of cell [0][0] (bottom-left).
    """

    width: int
    height: int
    resolution: float = Field(default=0.05, gt=0.0, description="Metres per cell.")
    origin_x: float = 0.0
    origin_y: float = 0.0
    frame_id: str = "world"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: str = Field(
        description="Base64-encoded row-major int8 grid (height × width)."
    )

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
