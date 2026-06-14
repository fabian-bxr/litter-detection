"""Stateful exploration planner: generates next waypoint, handles blocked goals."""

from __future__ import annotations

import time

import numpy as np

from ..config import AgentSettings
from ..interfaces.robodog import Pose2D
from ..mapping.grid import GridMap
from .coverage import CoverageTracker
from .reachability import DynamicObstacles, Blacklist, reachable_mask
from .scoring import Candidate, generate_candidates


class ExplorationPlanner:
    def __init__(
        self,
        raw_grid: GridMap,
        inflated_grid: GridMap,
        coverage: CoverageTracker,
        settings: AgentSettings,
    ) -> None:
        self._raw = raw_grid
        self._inflated = inflated_grid
        self._coverage = coverage
        self._cfg = settings
        self._dynamic_obs = DynamicObstacles()
        self._blacklist = Blacklist(radius_m=1.0)
        self._n_waypoints = 0
        self._start_time = time.monotonic()
        self._consecutive_low_gain = 0

    def _snap_to_free(self, effective: GridMap, r: int, c: int) -> tuple[int, int]:
        """If (r,c) is occupied in the inflated grid, return the nearest free cell."""
        if effective.is_free(r, c):
            return r, c
        free_cells = np.argwhere(effective.data == 0)
        if len(free_cells) == 0:
            return r, c
        dists = (free_cells[:, 0] - r) ** 2 + (free_cells[:, 1] - c) ** 2
        best = free_cells[int(np.argmin(dists))]
        return int(best[0]), int(best[1])

    def _refresh_reachable(self, pose: Pose2D) -> None:
        effective = self._dynamic_obs.apply_to(self._inflated)
        r, c = effective.world_to_grid(pose.x, pose.y)
        r, c = self._snap_to_free(effective, r, c)
        reach = reachable_mask(effective, r, c)
        self._coverage.set_reachable(reach)

    def _effective_pose(self, pose: Pose2D) -> Pose2D:
        """Return pose snapped to the nearest navigable cell (for candidate generation)."""
        effective = self._dynamic_obs.apply_to(self._inflated)
        r, c = effective.world_to_grid(pose.x, pose.y)
        r, c = self._snap_to_free(effective, r, c)
        x, y = effective.grid_to_world(r, c)
        return Pose2D(x=x, y=y, theta=pose.theta)

    def next_waypoint(self, pose: Pose2D) -> Candidate | None:
        """Return the best next waypoint, or None if no good candidate exists."""
        self._refresh_reachable(pose)
        effective = self._dynamic_obs.apply_to(self._inflated)
        unseen_target = self._coverage.denominator_mask() & ~self._coverage.seen

        # Use snapped pose so candidate rays start from navigable space
        nav_pose = self._effective_pose(pose)

        candidates = generate_candidates(
            pose=nav_pose,
            raw_grid=self._raw,
            inflated_grid=effective,
            unseen_target=unseen_target,
            blacklist=self._blacklist,
            n_directions=36,
            sample_start_m=self._cfg.sample_start_m,
            sample_step_m=self._cfg.sample_step_m,
            max_range_m=8.0,
            fov_deg=self._cfg.fov_deg,
            seen_range_m=self._cfg.seen_range_m,
            min_range_m=self._cfg.camera_min_range_m,
            w_gain=self._cfg.w_gain,
            w_dist=self._cfg.w_dist,
            w_turn=self._cfg.w_turn,
        )

        if not candidates or candidates[0].gain_m2 < self._cfg.min_gain_m2:
            self._consecutive_low_gain += 1
            return None

        self._consecutive_low_gain = 0
        self._n_waypoints += 1
        return candidates[0]

    def register_block(self, stall_pose: Pose2D, goal: Candidate) -> None:
        """Call when nav returns BLOCKED — burns a disc and blacklists the goal."""
        self._dynamic_obs.add_disc(stall_pose.x, stall_pose.y, radius_m=0.5)
        self._blacklist.add(goal.x, goal.y)
        self._consecutive_low_gain += 1

    def done(self) -> bool:
        if self._coverage.fraction() >= self._cfg.coverage_threshold:
            return True
        if self._consecutive_low_gain >= self._cfg.consecutive_low_gain_limit:
            return True
        if self._n_waypoints >= self._cfg.mission_max_waypoints:
            return True
        if time.monotonic() - self._start_time >= self._cfg.mission_max_duration_s:
            return True
        return False

    @property
    def n_waypoints(self) -> int:
        return self._n_waypoints

    @property
    def consecutive_low_gain(self) -> int:
        return self._consecutive_low_gain
