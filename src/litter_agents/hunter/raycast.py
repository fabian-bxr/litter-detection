"""DDA raycasting for visibility computation."""

from __future__ import annotations

import math

import numpy as np

from ..interfaces.robodog import Pose2D
from ..mapping.grid import GridMap


def visible_cells(
    pose: Pose2D,
    grid: GridMap,
    fov_deg: float = 70.0,
    range_m: float = 2.5,
    min_range_m: float = 0.3,
    n_rays: int = 90,
) -> np.ndarray:
    """Return bool mask (height, width) of cells visible from pose.

    Casts n_rays across fov_deg centred on pose.theta.
    Any cell that is not free (occupied or unknown) blocks the ray.
    Cells within min_range_m are excluded (camera blind spot).
    """
    half_fov = math.radians(fov_deg / 2.0)
    angles = np.linspace(pose.theta - half_fov, pose.theta + half_fov, n_rays)

    step = grid.resolution * 0.5          # sub-pixel sampling
    n_steps = int(range_m / step) + 1
    min_steps = max(0, int(min_range_m / step))
    t = np.arange(n_steps, dtype=np.float32) * step  # (n_steps,)

    # World positions for all rays and steps: (n_rays, n_steps)
    xs = pose.x + np.outer(np.cos(angles).astype(np.float32), t)
    ys = pose.y + np.outer(np.sin(angles).astype(np.float32), t)

    # Grid indices (integer)
    cols = ((xs - grid.origin_x) / grid.resolution).astype(np.int32)
    rows = ((ys - grid.origin_y) / grid.resolution).astype(np.int32)

    valid = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    rows_s = np.clip(rows, 0, grid.height - 1)
    cols_s = np.clip(cols, 0, grid.width - 1)
    cell_vals = grid.data[rows_s, cols_s]   # (n_rays, n_steps)

    # Blocking: non-free cell that is within grid bounds
    blocked = (cell_vals != 0) & valid

    mask = np.zeros((grid.height, grid.width), dtype=bool)

    for i in range(n_rays):
        ray_blocked = blocked[i]
        ray_valid = valid[i]

        # Find first blocking step at or after min_range
        after_min = ray_blocked.copy()
        after_min[:min_steps] = False

        stop = int(np.argmax(after_min)) if after_min.any() else n_steps

        # Mark cells from min_steps up to (not including) the blocking cell
        idx = np.arange(min_steps, stop)
        if idx.size == 0:
            continue
        ok = ray_valid[idx]
        mask[rows[i, idx[ok]], cols[i, idx[ok]]] = True

    return mask
