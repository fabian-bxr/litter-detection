from __future__ import annotations

import math

import numpy as np

from litter_detector.agents.models import Pose
from litter_detector.agents.tools.occupancy import OccupancyGrid

Polygon = list[tuple[float, float]]


def polygon_area_m2(polygon: Polygon) -> float:
    """Signed-shoelace polygon area in m² (absolute value)."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def rect_around(center_x: float, center_y: float, width: float, height: float) -> Polygon:
    """Axis-aligned rectangle centered at (center_x, center_y).

    `width` is the X extent, `height` is the Y extent. Returned counter-clockwise.
    """
    hw, hh = width / 2.0, height / 2.0
    return [
        (center_x - hw, center_y - hh),
        (center_x + hw, center_y - hh),
        (center_x + hw, center_y + hh),
        (center_x - hw, center_y + hh),
    ]


def rect_ahead(pose: Pose, depth: float, width: float) -> Polygon:
    """Rectangle starting at `pose`, extending `depth` meters along pose.theta, `width` wide.

    Vertices counter-clockwise relative to the pose's local frame.
    """
    cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
    hw = width / 2.0
    # Local corners: (forward, lateral)
    locals_ = [(0.0, -hw), (depth, -hw), (depth, hw), (0.0, hw)]
    return [
        (pose.x + fx * cos_t - ly * sin_t, pose.y + fx * sin_t + ly * cos_t)
        for (fx, ly) in locals_
    ]


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def polygon_mask(grid: OccupancyGrid, polygon: Polygon) -> np.ndarray:
    """Boolean mask over grid cells whose centers fall inside the polygon."""
    h, w = grid.height, grid.width
    # Bounding box in grid coordinates (clamped).
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_row, min_col = grid.world_to_cell(min(xs), min(ys))
    max_row, max_col = grid.world_to_cell(max(xs), max(ys))
    min_row = max(0, min_row)
    min_col = max(0, min_col)
    max_row = min(h - 1, max_row)
    max_col = min(w - 1, max_col)
    mask = np.zeros((h, w), dtype=bool)
    if min_row > max_row or min_col > max_col:
        return mask
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            wx, wy = grid.cell_to_world(r, c)
            if point_in_polygon(wx, wy, polygon):
                mask[r, c] = True
    return mask
