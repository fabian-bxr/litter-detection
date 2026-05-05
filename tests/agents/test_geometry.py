from __future__ import annotations

import math

import numpy as np

from litter_detector.agents.models import Pose
from litter_detector.agents.nbv.geometry import (
    point_in_polygon,
    polygon_mask,
    rect_ahead,
    rect_around,
)
from litter_detector.agents.tools.occupancy import OccupancyGrid


def test_rect_around_contains_center() -> None:
    poly = rect_around(2.0, 3.0, width=4.0, height=2.0)
    assert point_in_polygon(2.0, 3.0, poly)
    assert point_in_polygon(3.9, 3.9, poly)
    assert not point_in_polygon(4.1, 3.0, poly)


def test_rect_ahead_aligned_with_pose_heading() -> None:
    pose = Pose(x=0.0, y=0.0, theta=0.0)  # facing +x
    poly = rect_ahead(pose, depth=4.0, width=2.0)
    # Forward point should be inside
    assert point_in_polygon(2.0, 0.0, poly)
    # Behind the robot should NOT be inside
    assert not point_in_polygon(-1.0, 0.0, poly)


def test_rect_ahead_rotates_with_pose() -> None:
    pose = Pose(x=0.0, y=0.0, theta=math.pi / 2)  # facing +y
    poly = rect_ahead(pose, depth=4.0, width=2.0)
    assert point_in_polygon(0.0, 2.0, poly)
    assert not point_in_polygon(2.0, 0.0, poly)


def test_polygon_mask_counts_match_area() -> None:
    grid = OccupancyGrid(
        data=np.zeros((40, 40), dtype=np.int8),
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
    )
    poly = rect_around(2.0, 2.0, width=2.0, height=2.0)  # 2x2m → ~400 cells at 0.1m
    mask = polygon_mask(grid, poly)
    # Expect ~400 cells (allow ±20 for boundary discretization)
    assert 380 <= int(mask.sum()) <= 420
