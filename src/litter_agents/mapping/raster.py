"""Rasterize a search area specification onto a GridMap."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .grid import GridMap
from ..interfaces.robodog import Pose2D


@dataclass
class AreaSpec:
    """Search area in robot-relative frame (x = forward, y = left)."""

    shape: Literal["circle", "rectangle", "polygon"]
    radius_m: float | None = None
    width_m: float | None = None      # lateral extent (left+right)
    depth_m: float | None = None      # forward extent
    polygon_points: list[tuple[float, float]] | None = None
    center_dx_m: float = 0.0          # forward offset of centre from robot
    center_dy_m: float = 0.0          # lateral offset of centre from robot
    rotate_with_robot: bool = True


def _robot_to_world(
    dx: float,
    dy: float,
    robot_pose: Pose2D,
    rotate: bool,
) -> tuple[float, float]:
    if rotate:
        c, s = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
        wx = robot_pose.x + c * dx - s * dy
        wy = robot_pose.y + s * dx + c * dy
    else:
        wx = robot_pose.x + dx
        wy = robot_pose.y + dy
    return wx, wy


def rasterize_area(spec: AreaSpec, robot_pose: Pose2D, grid: GridMap) -> np.ndarray:
    """Return a bool mask (height, width) — True where cells are inside the area."""
    canvas = np.zeros((grid.height, grid.width), dtype=np.uint8)

    cx, cy = _robot_to_world(spec.center_dx_m, spec.center_dy_m, robot_pose, spec.rotate_with_robot)
    center_row, center_col = grid.world_to_grid(cx, cy)

    if spec.shape == "circle":
        assert spec.radius_m is not None, "radius_m required for circle"
        radius_px = max(1, int(round(spec.radius_m / grid.resolution)))
        cv2.circle(canvas, (center_col, center_row), radius_px, 255, -1)

    elif spec.shape == "rectangle":
        assert spec.width_m and spec.depth_m, "width_m and depth_m required for rectangle"
        half_w = spec.width_m / 2
        half_d = spec.depth_m / 2
        # Corners in robot local frame (x=forward, y=left)
        corners_local = np.array([
            [ half_d,  half_w],
            [ half_d, -half_w],
            [-half_d, -half_w],
            [-half_d,  half_w],
        ])
        angle = robot_pose.theta if spec.rotate_with_robot else 0.0
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        corners_world = corners_local @ rot.T + np.array([cx, cy])
        pts = []
        for wx, wy in corners_world:
            r, col = grid.world_to_grid(wx, wy)
            pts.append([col, r])
        cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], 255)

    elif spec.shape == "polygon":
        assert spec.polygon_points, "polygon_points required for polygon"
        pts = []
        for dx, dy in spec.polygon_points:
            wx, wy = _robot_to_world(dx, dy, robot_pose, spec.rotate_with_robot)
            r, col = grid.world_to_grid(wx, wy)
            pts.append([col, r])
        cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], 255)

    return canvas > 0
