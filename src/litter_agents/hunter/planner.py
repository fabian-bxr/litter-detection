from __future__ import annotations

import math

import numpy as np
from loguru import logger

from litter_agents.hunter.candidates import sample_standoff_viewpoints
from litter_agents.hunter.clusters import find_frontier_clusters, pick_active_cluster
from litter_agents.hunter.coverage import CoverageTracker
from litter_agents.hunter.frontier import frontier_waypoint
from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.pathing import plan_path
from litter_agents.hunter.reachability import (
    Blacklist,
    DynamicObstacles,
    reachable_mask,
)
from litter_agents.hunter.scoring import (
    Candidate,
    generate_candidates,
    score_viewpoints,
)
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
        rng_seed: int = 0,
    ) -> None:
        self.grid = grid
        self.params = params
        self.target_mask = target_mask
        self._blocked_raw = grid.blocked_mask()
        self._static_inflated = grid.inflated_blocked(params.robot_radius_m)
        self.dynamic = DynamicObstacles(grid, params.robot_radius_m)
        self.blacklist = Blacklist(params.blacklist_radius_m)
        self.frontier_blacklist = Blacklist(params.frontier_blacklist_radius_m)
        self.n_blocked = 0
        self._rng = np.random.default_rng(rng_seed)
        self._active_cluster_centroid: tuple[float, float] | None = None
        # Collision-free polyline to the last returned candidate (set by
        # next_waypoint); the executor follows it leg by leg.
        self.last_path: list[Pose2D] = []
        reachable = self._compute_reachable(start_pose)
        self.coverage = CoverageTracker(grid, target_mask, reachable, params)

    def blocked_inflated(self) -> np.ndarray:
        return self._static_inflated | self.dynamic.layer

    def _compute_reachable(self, pose: Pose2D) -> np.ndarray:
        return reachable_mask(
            ~self.blocked_inflated(), self.grid.world_to_grid(pose.x, pose.y)
        )

    def next_waypoint(self, pose: Pose2D) -> Candidate | None:
        """Next viewpoint to drive to, or None when nothing gainful remains.

        Side effect: sets ``self.last_path`` to the collision-free polyline the
        executor should follow to reach the returned candidate.
        """
        self.last_path = []
        if self.params.planner_mode == "nbv":
            return self._nbv_waypoint(pose)
        best = self._greedy_waypoint(pose)
        if best is not None:
            # Greedy/frontier candidates are already straight-line clear.
            self.last_path = [best.target]
            return best
        if not self.params.enable_frontier_fallback:
            return None
        wp = self._frontier_waypoint(pose)
        if wp is not None:
            self.last_path = [wp.target]
        return wp

    def plan_path_to(self, pose: Pose2D, target: Pose2D) -> list[Pose2D] | None:
        """Collision-free straight-leg polyline to ``target``, or None."""
        return plan_path(
            pose,
            target,
            grid=self.grid,
            blocked_inflated=self.blocked_inflated(),
            params=self.params,
        )

    def _nbv_waypoint(self, pose: Pose2D) -> Candidate | None:
        """Cluster-commit next-best-view: pick the highest cost-utility standoff
        viewpoint of the committed cluster that is actually reachable by a
        collision-free path (the robodog nav drives straight segments only)."""
        unseen = self.coverage.unseen_target()
        if not unseen.any():
            return None

        clusters = find_frontier_clusters(
            unseen, self.grid, self.params.min_cluster_cells
        )
        active, self._active_cluster_centroid = pick_active_cluster(
            clusters, pose, self._active_cluster_centroid, self.params
        )
        # Sample biased toward the committed cluster's cells; fall back to all
        # unseen when no cluster cleared the min-size floor.
        sampling_mask = active.cells if active is not None else unseen
        safe = ~self.blocked_inflated() & self.coverage.reachable
        positions = sample_standoff_viewpoints(
            pose,
            grid=self.grid,
            area_mask=self.target_mask,
            safe_mask=safe,
            unseen_mask=sampling_mask,
            params=self.params,
            rng=self._rng,
        )
        # Optionally include "stay and rotate": a zero-travel look-around wins
        # over a small forward hop whenever turning reveals as much, so the
        # robot doesn't nibble forward just to change heading.
        if self.params.nbv_rotate_in_place:
            positions = [(pose.x, pose.y), *positions]
        if not positions:
            return None
        candidates = score_viewpoints(
            pose,
            positions,
            grid=self.grid,
            blocked_raw=self._blocked_raw,
            unseen_target=unseen,
            params=self.params,
        )
        # Best-first, but only commit to a viewpoint we can actually path to.
        for cand in sorted(
            (c for c in candidates if c.gain_m2 > 0.0),
            key=lambda c: c.score,
            reverse=True,
        ):
            path = self.plan_path_to(pose, cand.target)
            if path is None:
                continue
            self.last_path = path
            logger.info(
                "NBV viewpoint: ({:.2f}, {:.2f}) θ {:+.0f}° gain {:.2f} m² "
                "via {} leg(s), {:.2f} m ({} clusters, active size {})",
                cand.target.x, cand.target.y, math.degrees(cand.target.theta),
                cand.gain_m2, len(path), cand.distance_m, len(clusters),
                active.size if active is not None else 0,
            )
            return cand
        return None

    def _greedy_waypoint(self, pose: Pose2D) -> Candidate | None:
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

    def _frontier_waypoint(self, pose: Pose2D) -> Candidate | None:
        """Reposition toward the nearest unseen reachable cell (around corners)."""
        wp = frontier_waypoint(
            pose,
            grid=self.grid,
            blocked_inflated=self.blocked_inflated(),
            reachable=self.coverage.reachable,
            unseen_target=self.coverage.unseen_target(),
            frontier_blacklist=self.frontier_blacklist,
            params=self.params,
        )
        if wp is not None:
            logger.info(
                "Frontier reposition: ({:.2f}, {:.2f}) dist {:.2f} m "
                "(greedy stalled, {:.1f} m² still unseen)",
                wp.target.x, wp.target.y, wp.distance_m,
                self.coverage.unseen_target().sum() * self.grid.resolution**2,
            )
        return wp

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
        # The committed cluster may be behind the obstacle; let NBV re-pick.
        self._active_cluster_centroid = None
        self.coverage.set_reachable(self._compute_reachable(robot_pose))
        logger.warning(
            "Registered blocked goal ({:.2f}, {:.2f}); reachable target now {:.1f} m²",
            failed_goal.x,
            failed_goal.y,
            self.coverage.denominator_m2(),
        )

    def done(self) -> bool:
        return self.coverage.fraction() >= self.params.coverage_target_fraction
