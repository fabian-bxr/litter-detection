from __future__ import annotations

import math

import numpy as np

from litter_detector.agents.models import Candidate, NBVParams, Pose
from litter_detector.agents.nbv.fov import raycast_wedge
from litter_detector.agents.tools.occupancy import OccupancyGrid


def _orient_toward_unseen(
    x: float,
    y: float,
    grid: OccupancyGrid,
    params: NBVParams,
    unseen_target: np.ndarray,
) -> float:
    """Aim the camera at the centroid of unseen cells within FOV range.

    Replaces the previous brute 8-orientation scan (one raycast per candidate
    instead of eight). The chosen theta is informative-driven and continuous,
    not quantized to k*pi/4 — heading sequences are smoother and the robot
    stops needlessly rotating between similar info-density angles.
    """
    rs, cs = np.where(unseen_target)
    if len(rs) == 0:
        return 0.0
    xs = grid.origin_x + (cs.astype(np.float64) + 0.5) * grid.resolution
    ys = grid.origin_y + (rs.astype(np.float64) + 0.5) * grid.resolution
    dx = xs - x
    dy = ys - y
    r_max_sq = (params.fov_range_m + grid.resolution) ** 2
    in_range = (dx * dx + dy * dy) <= r_max_sq
    if in_range.any():
        cx = float(xs[in_range].mean())
        cy = float(ys[in_range].mean())
    else:
        # No unseen in FOV reach; aim at the global unseen centroid so the
        # robot turns toward where work remains.
        cx = float(xs.mean())
        cy = float(ys.mean())
    return math.atan2(cy - y, cx - x)


def score_candidates(
    current: Pose,
    positions: list[tuple[float, float]],
    grid: OccupancyGrid,
    params: NBVParams,
    unseen_target: np.ndarray,
) -> list[Candidate]:
    """Score viewpoints with cost-utility:

        score = new_cells * exp(-lambda_cost * dist) * (1 + gamma_heading * cos(dtheta))

    where dtheta = (bearing from robot to candidate) - robot heading.

    - Multiplicative distance discount (never goes negative, doesn't blow up
      late in the mission like the old `gain/total_unseen - lambda*d` form).
    - cos(dtheta) is the directional-consistency / hysteresis term from
      Holz et al. 2010 and González-Baños & Latombe 2002 (see IEEE 2020
      review of frontier utility functions). It rewards committing to a
      direction over zig-zagging.

    Per-candidate orientation is chosen by aiming at the centroid of unseen
    cells in FOV reach, not by an 8-way scan.
    """
    out: list[Candidate] = []
    total_target = max(1, int(unseen_target.sum()))
    for x, y in positions:
        theta = _orient_toward_unseen(x, y, grid, params, unseen_target)
        wedge = raycast_wedge(Pose(x=x, y=y, theta=theta), params, grid)
        new_cells = int((wedge & unseen_target).sum())
        cost = math.hypot(x - current.x, y - current.y)
        if cost > 1e-9:
            bearing = math.atan2(y - current.y, x - current.x)
            d_theta = math.atan2(
                math.sin(bearing - current.theta),
                math.cos(bearing - current.theta),
            )
            heading_factor = 1.0 + params.gamma_heading * math.cos(d_theta)
        else:
            heading_factor = 1.0
        distance_factor = math.exp(-params.lambda_cost * cost)
        score = float(new_cells) * distance_factor * heading_factor
        out.append(
            Candidate(
                pose=Pose(x=x, y=y, theta=theta),
                gain=new_cells / total_target,
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
