from __future__ import annotations

import math

import numpy as np

from litter_detector.agents.models import NBVParams, Pose
from litter_detector.agents.nbv.geometry import Polygon, point_in_polygon
from litter_detector.agents.tools.occupancy import OccupancyGrid


_FRONTIER_BIAS = 0.8  # fraction of attempts seeded from unseen frontier cells


def sample_candidate_positions(
    current: Pose,
    grid: OccupancyGrid,
    polygon: Polygon,
    params: NBVParams,
    rng: np.random.Generator,
    safe_mask: np.ndarray,
    unseen_mask: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Reject-sample candidate (x,y) positions in world coordinates.

    Sampling strategy:
      - With probability `_FRONTIER_BIAS` (when `unseen_mask` is non-empty),
        pick a random unseen target cell and offset back by a random distance
        in [0.4, 0.9] x fov_range_m in a random direction. The viewpoint then
        sits at a stand-off from the frontier so the FOV wedge actually
        sweeps it on the next pose. This avoids the previous behavior of
        clustering candidates in a 1.5 m disc around the robot — which kept
        sampling already-seen ground.
      - Otherwise, fall back to a uniform sample over the polygon's bounding
        box (so we still explore when no unseen cells exist nearby yet).

    A candidate is accepted iff:
      - inside `polygon`
      - lands on a `safe_mask` cell (free + obstacle-inflated)
      - is at least max(0.5 * candidate_step_m, 4 * resolution) from current
        pose (avoids near-zero motion that wastes a nav cycle)
      - has min separation `candidate_min_separation_m` from accepted ones
    """
    h, w = safe_mask.shape
    if not safe_mask.any():
        return []
    min_sep_sq = params.candidate_min_separation_m ** 2
    min_step = max(grid.resolution * 4, 0.5 * params.candidate_step_m)
    min_step_sq = min_step ** 2

    frontier_xy: tuple[np.ndarray, np.ndarray] | None = None
    if unseen_mask is not None and unseen_mask.any():
        rs, cs = np.where(unseen_mask)
        fxs = grid.origin_x + (cs.astype(np.float64) + 0.5) * grid.resolution
        fys = grid.origin_y + (rs.astype(np.float64) + 0.5) * grid.resolution
        frontier_xy = (fxs, fys)

    xs_poly = [p[0] for p in polygon]
    ys_poly = [p[1] for p in polygon]
    bx_min, bx_max = min(xs_poly), max(xs_poly)
    by_min, by_max = min(ys_poly), max(ys_poly)

    fov_r = params.fov_range_m
    positions: list[tuple[float, float]] = []
    max_attempts = max(1, params.n_candidates) * 40

    for _ in range(max_attempts):
        if len(positions) >= params.n_candidates:
            break

        if frontier_xy is not None and rng.random() < _FRONTIER_BIAS:
            fxs, fys = frontier_xy
            i = int(rng.integers(0, len(fxs)))
            d = rng.uniform(0.4 * fov_r, 0.9 * fov_r)
            a = rng.uniform(0.0, 2.0 * math.pi)
            x = float(fxs[i]) + d * math.cos(a)
            y = float(fys[i]) + d * math.sin(a)
        else:
            x = rng.uniform(bx_min, bx_max)
            y = rng.uniform(by_min, by_max)

        if not point_in_polygon(x, y, polygon):
            continue
        row, col = grid.world_to_cell(x, y)
        if row < 0 or row >= h or col < 0 or col >= w:
            continue
        if not safe_mask[row, col]:
            continue
        if (x - current.x) ** 2 + (y - current.y) ** 2 < min_step_sq:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < min_sep_sq for px, py in positions):
            continue
        positions.append((x, y))
    return positions
