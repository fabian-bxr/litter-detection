"""Standoff-viewpoint sampling for the NBV planner.

Instead of proposing tiny straight-line steps from the current pose (which
crawl in clutter), this reject-samples genuine *viewpoints*: positions a
stand-off distance behind a frontier cell, so the FOV wedge actually sweeps
unseen area on arrival. Biasing the seeds toward the committed cluster's cells
keeps candidates where the remaining work is. Mask-based (uses the rasterized
search-area mask), so no polygon geometry is needed.

Ported from feature/agent-setup ``nbv/candidates.py``.
"""

from __future__ import annotations

import math

import numpy as np

from litter_agents.hunter.params import HunterParams
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap


def sample_standoff_viewpoints(
    current: Pose2D,
    *,
    grid: GridMap,
    area_mask: np.ndarray,
    safe_mask: np.ndarray,
    unseen_mask: np.ndarray,
    params: HunterParams,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Reject-sample candidate (x, y) viewpoints in world coordinates.

    With probability ``frontier_bias`` a seed is a random unseen cell offset
    back by ``[standoff_frac_min, standoff_frac_max] * camera_range_m`` in a
    random direction; otherwise a uniform sample over the search area's
    bounding box (keeps exploring when no unseen sits nearby). Accepted iff the
    cell is inside the search area, on ``safe_mask`` (reachable + inflated-free),
    a minimum step from the current pose, and ``candidate_min_separation_m``
    from already-accepted viewpoints.
    """
    h, w = safe_mask.shape
    if not safe_mask.any() or not area_mask.any():
        return []

    rows_a, cols_a = np.where(area_mask)
    bx_min, bx_max = grid.grid_to_world(0, int(cols_a.min()))[0], grid.grid_to_world(
        0, int(cols_a.max())
    )[0]
    by_min, by_max = grid.grid_to_world(int(rows_a.min()), 0)[1], grid.grid_to_world(
        int(rows_a.max()), 0
    )[1]

    frontier_xy: tuple[np.ndarray, np.ndarray] | None = None
    if unseen_mask.any():
        rs, cs = np.where(unseen_mask)
        fxs = grid.origin_x + (cs.astype(np.float64) + 0.5) * grid.resolution
        fys = grid.origin_y + (rs.astype(np.float64) + 0.5) * grid.resolution
        frontier_xy = (fxs, fys)

    min_sep_sq = params.candidate_min_separation_m**2
    min_step = max(grid.resolution * 4, params.candidate_min_step_m)
    min_step_sq = min_step**2
    fov_r = params.camera_range_m

    positions: list[tuple[float, float]] = []
    for _ in range(max(1, params.n_candidates) * 40):
        if len(positions) >= params.n_candidates:
            break

        if frontier_xy is not None and rng.random() < params.frontier_bias:
            fxs, fys = frontier_xy
            i = int(rng.integers(0, len(fxs)))
            d = rng.uniform(params.standoff_frac_min * fov_r, params.standoff_frac_max * fov_r)
            a = rng.uniform(0.0, 2.0 * math.pi)
            x = float(fxs[i]) + d * math.cos(a)
            y = float(fys[i]) + d * math.sin(a)
        else:
            x = rng.uniform(bx_min, bx_max)
            y = rng.uniform(by_min, by_max)

        row, col = grid.world_to_grid(x, y)
        if not (0 <= row < h and 0 <= col < w):
            continue
        if not area_mask[row, col] or not safe_mask[row, col]:
            continue
        if (x - current.x) ** 2 + (y - current.y) ** 2 < min_step_sq:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < min_sep_sq for px, py in positions):
            continue
        positions.append((x, y))
    return positions
