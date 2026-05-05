from __future__ import annotations

import math

import numpy as np

from litter_detector.agents.models import NBVParams, Pose
from litter_detector.agents.nbv.geometry import Polygon, point_in_polygon
from litter_detector.agents.tools.occupancy import OccupancyGrid


def sample_candidate_positions(
    current: Pose,
    grid: OccupancyGrid,
    polygon: Polygon,
    params: NBVParams,
    rng: np.random.Generator,
    safe_mask: np.ndarray,
) -> list[tuple[float, float]]:
    """Reject-sample candidate (x,y) positions in world coordinates.

    A candidate is valid iff:
      - within `candidate_step_m` of current pose (annulus from 0.2m for movement)
      - inside the search polygon
      - lands on a `safe_mask` cell (free + obstacle-inflated)
      - has min separation `candidate_min_separation_m` from already-accepted candidates
    """
    positions: list[tuple[float, float]] = []
    min_sep_sq = params.candidate_min_separation_m ** 2
    max_attempts = params.n_candidates * 20
    min_radius = max(0.2, grid.resolution * 2)
    for _ in range(max_attempts):
        if len(positions) >= params.n_candidates:
            break
        # Uniform in disc around current pose.
        r = math.sqrt(rng.uniform(min_radius ** 2, params.candidate_step_m ** 2))
        a = rng.uniform(0.0, 2.0 * math.pi)
        x = current.x + r * math.cos(a)
        y = current.y + r * math.sin(a)
        if not point_in_polygon(x, y, polygon):
            continue
        row, col = grid.world_to_cell(x, y)
        if not grid.in_bounds(row, col) or not safe_mask[row, col]:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < min_sep_sq for px, py in positions):
            continue
        positions.append((x, y))
    return positions
