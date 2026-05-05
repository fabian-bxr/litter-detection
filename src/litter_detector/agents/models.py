from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Pose(BaseModel):
    """2D robot pose in the odometry frame."""

    x: float
    y: float
    theta: float  # yaw, radians
    stamp: datetime | None = None


class SearchArea(BaseModel):
    """Polygon in odometry frame defining the bounded search region.

    Anchored to the robot pose at the time the user's request was received.
    """

    polygon: list[tuple[float, float]] = Field(min_length=3)
    anchor_pose: Pose
    label: str = ""


class NBVParams(BaseModel):
    """Greedy next-best-view planner parameters."""

    h_fov_deg: float = 70.0
    fov_range_m: float = 4.0
    n_candidates: int = 16
    candidate_step_m: float = 1.5  # max distance from current pose for candidates
    candidate_min_separation_m: float = 0.5
    lambda_cost: float = 0.4  # score = gain - lambda_cost * cost_m
    coverage_target: float = 0.95
    max_iterations: int = 50
    obstacle_inflation_m: float = 0.3
    seen_mask_decay: float = 0.0  # 0 = never forget


class Candidate(BaseModel):
    """A scored next-best-view candidate pose."""

    pose: Pose
    gain: float  # newly-seen free cells if visited (normalized 0..1 of remaining unseen)
    cost_m: float  # euclidean distance from current pose (v1)
    score: float


MissionState = Literal[
    "idle",
    "planning",
    "navigating",
    "arrived",
    "completed",
    "blocked",
    "failed",
    "aborted",
]


class MissionStatus(BaseModel):
    """Live status of the active mission."""

    mission_id: str
    state: MissionState
    coverage: float = 0.0  # fraction of free cells in polygon that are now seen
    iteration: int = 0
    last_message: str = ""
    current_target: Pose | None = None
