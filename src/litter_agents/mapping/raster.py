from __future__ import annotations

import math

import cv2
import numpy as np
from loguru import logger

from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap


def rasterize_area(
    spec: SearchAreaSpec, robot_pose: Pose2D, grid: GridMap
) -> np.ndarray:
    """Rasterize a search-area spec onto the map grid.

    Returns a bool (height × width) mask of target cells. The shape's center
    offset is interpreted in the robot frame (+x forward, +y left) and always
    rotates with the robot's heading; ``rotate_with_robot`` additionally
    aligns rectangle/polygon orientation with the heading.
    """
    cos_t, sin_t = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
    center_x = robot_pose.x + cos_t * spec.center_dx_m - sin_t * spec.center_dy_m
    center_y = robot_pose.y + sin_t * spec.center_dx_m + cos_t * spec.center_dy_m
    shape_theta = robot_pose.theta if spec.rotate_with_robot else 0.0

    canvas = np.zeros((grid.height, grid.width), dtype=np.uint8)

    if spec.shape == "circle":
        assert spec.radius_m is not None
        row, col = grid.world_to_grid(center_x, center_y)
        radius_cells = max(1, round(spec.radius_m / grid.resolution))
        cv2.circle(canvas, (col, row), radius_cells, 1, thickness=-1)
        requested_m2 = math.pi * spec.radius_m**2
    elif spec.shape == "rectangle":
        assert spec.width_m is not None and spec.depth_m is not None
        # Local frame: x forward (depth), y left (width).
        half_d, half_w = spec.depth_m / 2.0, spec.width_m / 2.0
        local = [(-half_d, -half_w), (half_d, -half_w), (half_d, half_w), (-half_d, half_w)]
        _fill_polygon(canvas, local, center_x, center_y, shape_theta, grid)
        requested_m2 = spec.width_m * spec.depth_m
    else:  # polygon
        assert spec.polygon_points is not None
        _fill_polygon(canvas, spec.polygon_points, center_x, center_y, shape_theta, grid)
        requested_m2 = _polygon_area(spec.polygon_points)

    mask = canvas.astype(bool)
    drawn_m2 = float(mask.sum()) * grid.resolution**2
    if drawn_m2 < 0.95 * requested_m2:
        logger.warning(
            "Search area extends beyond the map: requested {:.1f} m², "
            "only {:.1f} m² fall inside",
            requested_m2,
            drawn_m2,
        )
    return mask


def _fill_polygon(
    canvas: np.ndarray,
    local_points: list[tuple[float, float]],
    center_x: float,
    center_y: float,
    theta: float,
    grid: GridMap,
) -> None:
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cells = []
    for px, py in local_points:
        wx = center_x + cos_t * px - sin_t * py
        wy = center_y + sin_t * px + cos_t * py
        row, col = grid.world_to_grid(wx, wy)
        cells.append((col, row))  # cv2 points are (x=col, y=row)
    cv2.fillPoly(canvas, [np.array(cells, dtype=np.int32)], 1)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    """Shoelace formula."""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
