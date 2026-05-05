from __future__ import annotations

import math

import numpy as np

from litter_detector.agents.models import NBVParams, Pose
from litter_detector.agents.tools.occupancy import OccupancyGrid


def _ray_step(
    grid: OccupancyGrid,
    occupied: np.ndarray,
    out: np.ndarray,
    start_row: float,
    start_col: float,
    angle_world: float,
    max_range_cells: float,
) -> None:
    """Walk a single ray in cell space, marking each traversed cell as seen.

    Stops at the first occupied cell or when `max_range_cells` is reached.
    Cells out of bounds terminate the ray.
    """
    # Direction in cell coordinates: world (cos, sin) maps directly to (col, row)
    # because world_to_cell is +x→col, +y→row with axis-aligned grid.
    dcol = math.cos(angle_world)
    drow = math.sin(angle_world)
    h, w = out.shape
    # Step in unit-length increments along the ray (cell units).
    n_steps = int(math.ceil(max_range_cells))
    for i in range(n_steps + 1):
        col = int(start_col + dcol * i)
        row = int(start_row + drow * i)
        if row < 0 or row >= h or col < 0 or col >= w:
            return
        if occupied[row, col]:
            return
        out[row, col] = True


def raycast_wedge(
    pose: Pose,
    params: NBVParams,
    grid: OccupancyGrid,
) -> np.ndarray:
    """Return a boolean mask of cells visible from `pose` within the FOV wedge.

    Rays span [theta - h_fov/2, theta + h_fov/2]; each terminates at the first
    occupied cell or when `fov_range_m` is exceeded. Unknown cells (-1) are
    traversed (we don't know they're blocked) — they will simply not be 'free'
    in the seen-coverage calculation.
    """
    occupied = grid.occupied_mask()
    seen = np.zeros((grid.height, grid.width), dtype=bool)
    start_row, start_col = grid.world_to_cell(pose.x, pose.y)
    if not grid.in_bounds(start_row, start_col):
        return seen
    half_fov = math.radians(params.h_fov_deg) / 2.0
    max_range_cells = params.fov_range_m / grid.resolution
    # Angular resolution: one ray per ~half cell at max range.
    arc_len_cells = 2.0 * half_fov * max_range_cells
    n_rays = max(8, int(math.ceil(arc_len_cells * 2.0)))
    for i in range(n_rays):
        t = i / max(1, n_rays - 1)
        angle = pose.theta - half_fov + t * 2.0 * half_fov
        _ray_step(grid, occupied, seen, start_row + 0.5, start_col + 0.5, angle, max_range_cells)
    return seen
