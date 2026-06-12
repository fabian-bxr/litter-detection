from __future__ import annotations

import numpy as np


def visible_cells(
    blocked: np.ndarray,
    origin_rc: tuple[float, float],
    heading_rad: float,
    fov_rad: float,
    max_range_cells: float,
    min_range_cells: float,
    n_rays: int,
    step_cells: float = 0.5,
) -> np.ndarray:
    """Cells visible from ``origin_rc`` through a FoV wedge, as a bool (H, W) mask.

    Vectorized DDA: ``n_rays`` rays fan across ``[heading - fov/2, heading +
    fov/2]``, sampled every ``step_cells`` out to ``max_range_cells``. A ray
    stops at the first blocked or out-of-map cell; the blocking cell itself is
    NOT marked visible (a wall surface is never "inspected for litter").
    Samples closer than ``min_range_cells`` are excluded (near blind spot), so
    the visible region is an annular wedge.

    Works in grid space: row = y, col = x, both increasing with the world
    axes, so a world heading can be used directly.
    """
    h, w = blocked.shape
    angles = np.linspace(
        heading_rad - fov_rad / 2.0, heading_rad + fov_rad / 2.0, n_rays
    )
    t = np.arange(step_cells, max_range_cells + 1e-9, step_cells)
    if t.size == 0:
        return np.zeros((h, w), dtype=bool)

    rows = origin_rc[0] + np.sin(angles)[:, None] * t[None, :]
    cols = origin_rc[1] + np.cos(angles)[:, None] * t[None, :]
    ri = np.floor(rows).astype(np.intp)
    ci = np.floor(cols).astype(np.intp)

    oob = (ri < 0) | (ri >= h) | (ci < 0) | (ci >= w)
    stop = blocked[np.clip(ri, 0, h - 1), np.clip(ci, 0, w - 1)] | oob
    # Index of the first stop per ray; rays without one run the full range.
    first_stop = np.where(stop.any(axis=1), np.argmax(stop, axis=1), t.size)

    step_idx = np.arange(t.size)[None, :]
    vis = (step_idx < first_stop[:, None]) & (t[None, :] >= min_range_cells)

    out = np.zeros((h, w), dtype=bool)
    out[ri[vis], ci[vis]] = True
    return out


def ray_clearance_cells(
    blocked: np.ndarray,
    origin_rc: tuple[float, float],
    heading_rad: float,
    max_range_cells: float,
    skip_cells: float = 0.0,
    step_cells: float = 0.5,
) -> float:
    """Free distance (in cells) along one ray before the first blocked cell.

    ``skip_cells`` ignores blockage near the origin — used to let the robot
    escape when its own cell lies inside the inflated obstacle layer.
    """
    h, w = blocked.shape
    t = np.arange(step_cells, max_range_cells + 1e-9, step_cells)
    if t.size == 0:
        return 0.0
    rows = origin_rc[0] + np.sin(heading_rad) * t
    cols = origin_rc[1] + np.cos(heading_rad) * t
    ri = np.floor(rows).astype(np.intp)
    ci = np.floor(cols).astype(np.intp)
    oob = (ri < 0) | (ri >= h) | (ci < 0) | (ci >= w)
    stop = (blocked[np.clip(ri, 0, h - 1), np.clip(ci, 0, w - 1)] | oob) & (
        t > skip_cells
    )
    if not stop.any():
        return float(max_range_cells)
    return float(t[np.argmax(stop)] - step_cells)
