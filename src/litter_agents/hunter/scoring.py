"""Candidate waypoint generation and info-gain scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..interfaces.robodog import Pose2D
from ..mapping.grid import GridMap
from .raycast import visible_cells
from .reachability import Blacklist


@dataclass
class Candidate:
    x: float
    y: float
    heading: float      # direction of travel = arrival heading (radians)
    distance_m: float
    gain_m2: float
    score: float

    @property
    def pose(self) -> Pose2D:
        return Pose2D(x=self.x, y=self.y, theta=self.heading)


def _ray_clear_distance(
    inflated_grid: GridMap,
    ox: float,
    oy: float,
    heading: float,
    max_m: float,
) -> float:
    """How far we can travel along heading on inflated_grid before hitting a blocked cell."""
    step = inflated_grid.resolution * 0.5
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    n = int(max_m / step) + 1
    for i in range(1, n):
        t = i * step
        r, c = inflated_grid.world_to_grid(ox + cos_h * t, oy + sin_h * t)
        if not inflated_grid.in_bounds(r, c) or inflated_grid.data[r, c] != 0:
            return max(0.0, (i - 1) * step)
    return max_m


def generate_candidates(
    pose: Pose2D,
    raw_grid: GridMap,
    inflated_grid: GridMap,
    unseen_target: np.ndarray,
    blacklist: Blacklist,
    *,
    n_directions: int = 36,
    sample_start_m: float = 0.75,
    sample_step_m: float = 0.5,
    max_range_m: float = 8.0,
    fov_deg: float = 70.0,
    seen_range_m: float = 2.5,
    min_range_m: float = 0.3,
    n_rays: int = 90,
    w_gain: float = 1.0,
    w_dist: float = 0.25,
    w_turn: float = 0.3,
) -> list[Candidate]:
    """Generate and score waypoint candidates in all directions.

    Gain is cumulative per direction: each further sample adds what it sees
    that wasn't already seen by closer samples along the same heading.
    """
    candidates: list[Candidate] = []
    res = raw_grid.resolution

    for i in range(n_directions):
        heading = i * (2 * math.pi / n_directions)
        clear_m = _ray_clear_distance(inflated_grid, pose.x, pose.y, heading, max_range_m)
        max_sample = min(clear_m, max_range_m)
        if max_sample < sample_start_m:
            continue

        cos_h, sin_h = math.cos(heading), math.sin(heading)
        cum_seen = np.zeros((raw_grid.height, raw_grid.width), dtype=bool)
        d = sample_start_m

        while d <= max_sample + 1e-9:
            wx = pose.x + cos_h * d
            wy = pose.y + sin_h * d
            if not blacklist.is_blacklisted(wx, wy):
                sample_pose = Pose2D(x=wx, y=wy, theta=heading)
                cum_seen |= visible_cells(
                    sample_pose, raw_grid, fov_deg, seen_range_m, min_range_m, n_rays
                )
                gain_m2 = float((cum_seen & unseen_target).sum()) * (res ** 2)
                delta_h = math.atan2(
                    math.sin(heading - pose.theta),
                    math.cos(heading - pose.theta),
                )
                score = w_gain * gain_m2 - w_dist * d - w_turn * abs(delta_h)
                candidates.append(Candidate(
                    x=wx, y=wy, heading=heading,
                    distance_m=d, gain_m2=gain_m2, score=score,
                ))
            d += sample_step_m

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
