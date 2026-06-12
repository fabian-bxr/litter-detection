from __future__ import annotations

import math

import numpy as np
from loguru import logger

from litter_agents.hunter.coverage import CoverageTracker
from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.reachability import (
    Blacklist,
    DynamicObstacles,
    reachable_mask,
)
from litter_agents.hunter.scoring import Candidate, generate_candidates
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap


class ExplorationPlanner:
    """Owns the exploration state: coverage, obstacles, blacklist, reachability."""

    def __init__(
        self,
        grid: GridMap,
        target_mask: np.ndarray,
        params: HunterParams,
        start_pose: Pose2D,
    ) -> None:
        self.grid = grid
        self.params = params
        self._blocked_raw = grid.blocked_mask()
        self._static_inflated = grid.inflated_blocked(params.robot_radius_m)
        self.dynamic = DynamicObstacles(grid, params.robot_radius_m)
        self.blacklist = Blacklist(params.blacklist_radius_m)
        self.n_blocked = 0
        reachable = self._compute_reachable(start_pose)
        self.coverage = CoverageTracker(grid, target_mask, reachable, params)

    def blocked_inflated(self) -> np.ndarray:
        return self._static_inflated | self.dynamic.layer

    def _compute_reachable(self, pose: Pose2D) -> np.ndarray:
        return reachable_mask(
            ~self.blocked_inflated(), self.grid.world_to_grid(pose.x, pose.y)
        )

    def next_waypoint(self, pose: Pose2D) -> Candidate | None:
        """Best straight-line waypoint, or None when nothing gains enough."""
        candidates = generate_candidates(
            pose,
            grid=self.grid,
            blocked_inflated=self.blocked_inflated(),
            blocked_raw=self._blocked_raw,
            unseen_target=self.coverage.unseen_target(),
            blacklist=self.blacklist,
            params=self.params,
        )
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.score)
        if best.gain_m2 < self.params.min_gain_m2:
            return None
        return best

    def register_block(
        self, stall_pose: Pose2D, failed_goal: Pose2D, robot_pose: Pose2D
    ) -> None:
        """Record a failed goto: obstacle disc, goal blacklist, reachability redo.

        The actual obstruction sits in front of where the robot stalled, so
        the disc is placed one and a half robot radii beyond the stall pose
        toward the goal — not on the stall pose itself, which the robot
        demonstrably occupied.
        """
        bearing = stall_pose.bearing_to(failed_goal)
        offset = 1.5 * self.params.robot_radius_m
        self.dynamic.add_disc(
            stall_pose.x + math.cos(bearing) * offset,
            stall_pose.y + math.sin(bearing) * offset,
            self.params.robot_radius_m,
        )
        self.blacklist.add(failed_goal.x, failed_goal.y)
        self.n_blocked += 1
        self.coverage.set_reachable(self._compute_reachable(robot_pose))
        logger.warning(
            "Registered blocked goal ({:.2f}, {:.2f}); reachable target now {:.1f} m²",
            failed_goal.x,
            failed_goal.y,
            self.coverage.denominator_m2(),
        )

    def done(self) -> bool:
        return self.coverage.fraction() >= self.params.coverage_target_fraction
