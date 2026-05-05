from __future__ import annotations

import math

import numpy as np

from litter_detector.agents.models import Candidate, NBVParams, Pose
from litter_detector.agents.nbv.fov import raycast_wedge
from litter_detector.agents.tools.occupancy import OccupancyGrid


N_ORIENTATIONS = 8


def _best_orientation(
    x: float,
    y: float,
    grid: OccupancyGrid,
    params: NBVParams,
    unseen_target: np.ndarray,
) -> tuple[float, int]:
    """Return (best_theta, best_new_cells) by trying N_ORIENTATIONS at this position."""
    best_theta = 0.0
    best_count = -1
    for k in range(N_ORIENTATIONS):
        theta = (2.0 * math.pi * k) / N_ORIENTATIONS
        wedge = raycast_wedge(Pose(x=x, y=y, theta=theta), params, grid)
        new = int((wedge & unseen_target).sum())
        if new > best_count:
            best_count = new
            best_theta = theta
    return best_theta, max(0, best_count)


def score_candidates(
    current: Pose,
    positions: list[tuple[float, float]],
    grid: OccupancyGrid,
    params: NBVParams,
    unseen_target: np.ndarray,
) -> list[Candidate]:
    """Score each candidate position. Orientation is chosen to maximize gain.

    `unseen_target` = (polygon_mask & ~occupied_mask) & ~seen — the cells we still
    care about covering. Gain is normalized by its total at scoring time, so
    score = gain - lambda * cost is unitless.
    """
    total_unseen = max(1, int(unseen_target.sum()))
    out: list[Candidate] = []
    for x, y in positions:
        theta, new_cells = _best_orientation(x, y, grid, params, unseen_target)
        gain = new_cells / total_unseen
        cost = math.hypot(x - current.x, y - current.y)
        score = gain - params.lambda_cost * cost
        out.append(
            Candidate(
                pose=Pose(x=x, y=y, theta=theta),
                gain=gain,
                cost_m=cost,
                score=score,
            )
        )
    return out


def best_candidate(candidates: list[Candidate]) -> Candidate | None:
    """Return the highest-scoring candidate with strictly positive gain."""
    feasible = [c for c in candidates if c.gain > 0]
    if not feasible:
        return None
    return max(feasible, key=lambda c: c.score)
