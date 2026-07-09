"""Agent output schemas and mission-level data models.

These are plain pydantic models so that mapping/, validation/ and sim/ can use
them without importing pydantic-ai; the agents in litter_agents.agents build
their structured outputs around them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from litter_agents.interfaces.robodog import Pose2D

LitterCategory = Literal[
    "plastic",
    "paper",
    "cardboard",
    "metal",
    "glass",
    "organic",
    "cigarette",
    "textile",
    "other",
]


class SearchAreaSpec(BaseModel):
    """Structured search area, relative to the robot's pose at mission start.

    The robot sits at the origin facing +x; +y is to its left. ``center_dx_m``
    / ``center_dy_m`` offset the shape's center in that robot frame (the
    offset always rotates with the robot — "5 m in front of me" depends on
    where I face). ``rotate_with_robot`` controls only the *orientation* of
    rectangles/polygons: True aligns them with the robot's heading, False
    keeps them axis-aligned in the world frame.
    """

    shape: Literal["circle", "rectangle", "polygon"]
    radius_m: float | None = Field(default=None, gt=0, le=100)  # circle
    width_m: float | None = Field(default=None, gt=0, le=200)  # rectangle, lateral
    depth_m: float | None = Field(default=None, gt=0, le=200)  # rectangle, forward
    polygon_points: list[tuple[float, float]] | None = None  # robot frame, >= 3
    center_dx_m: float = 0.0  # forward offset from the robot
    center_dy_m: float = 0.0  # left offset from the robot
    rotate_with_robot: bool = True
    rationale: str = ""

    @model_validator(mode="after")
    def _check_shape_fields(self) -> "SearchAreaSpec":
        if self.shape == "circle" and self.radius_m is None:
            raise ValueError("shape 'circle' requires radius_m")
        if self.shape == "rectangle" and (self.width_m is None or self.depth_m is None):
            raise ValueError("shape 'rectangle' requires width_m and depth_m")
        if self.shape == "polygon" and (
            self.polygon_points is None or len(self.polygon_points) < 3
        ):
            raise ValueError("shape 'polygon' requires at least 3 polygon_points")
        return self


class LitterValidation(BaseModel):
    """Vision agent verdict on one cropped detection."""

    is_litter: bool
    category: LitterCategory | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    description: str

    @model_validator(mode="after")
    def _check_category(self) -> "LitterValidation":
        if self.is_litter and self.category is None:
            raise ValueError("category is required when is_litter is true")
        return self


class FindingSummary(BaseModel):
    """One validated litter finding, as it appears in the mission report."""

    track_id: int
    category: str
    confidence: float
    robot_pose: Pose2D  # where the robot stood when it saw the litter
    bearing_rad: float  # camera-frame bearing to the object (0 = straight ahead)
    image_path: str
    description: str
    possible_duplicate_of: int | None = None  # track_id of a nearby same-category find


class MissionReport(BaseModel):
    mission_id: str
    prompt: str
    area: SearchAreaSpec
    coverage_fraction: float
    reachable_target_m2: float
    duration_s: float
    distance_traveled_m: float
    n_waypoints: int
    n_blocked: int
    findings: list[FindingSummary]
    n_rejected: int = 0
    n_errors: int = 0
    summary_text: str = ""
