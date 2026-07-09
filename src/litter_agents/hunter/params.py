from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from litter_agents.config import AgentSettings


@dataclass(frozen=True)
class HunterParams:
    """Tunables of the exploration algorithm, decoupled from AgentSettings so
    the hunter package stays importable without the zenoh-flavored config."""

    robot_radius_m: float = 0.25  # Go2 half-width ~0.16 m + margin
    fov_rad: float = math.radians(70.0)
    camera_range_m: float = 3.0
    camera_min_range_m: float = 0.3
    n_fov_rays: int = 90
    n_scoring_rays: int = 45
    n_candidate_directions: int = 36
    candidate_step_m: float = 0.5
    candidate_min_dist_m: float = 0.5
    candidate_max_dist_m: float = 8.0
    w_gain: float = 1.0
    w_dist: float = 0.25
    w_turn: float = 0.3
    min_gain_m2: float = 0.15
    blacklist_radius_m: float = 0.5
    coverage_target_fraction: float = 0.95
    # When the greedy scorer finds no gainful straight-line move, reposition
    # toward the nearest unseen reachable cell instead of stopping immediately.
    enable_frontier_fallback: bool = True
    frontier_blacklist_radius_m: float = 0.5

    # ── Next-best-view planner (cluster-commit, standoff-viewpoint sampling) ──
    # "nbv" samples standoff viewpoints biased toward a committed frontier
    # cluster and ranks them by a multiplicative cost-utility; "greedy" is the
    # legacy ray-scorer + frontier fallback. See hunter/clusters.py, candidates.py.
    planner_mode: Literal["greedy", "nbv"] = "nbv"
    n_candidates: int = 16
    candidate_min_separation_m: float = 0.5
    # Smallest translation worth a nav cycle — viewpoints closer than this to
    # the robot are rejected, so the planner rotates in place (a stay candidate
    # is always scored) rather than nibbling forward.
    candidate_min_step_m: float = 0.4
    # Score a zero-travel "stay and rotate" candidate each step. Fewer/larger
    # moves (good for hardware stop-go) at the cost of more total travel.
    nbv_rotate_in_place: bool = True
    # score = new_cells * exp(-lambda_cost * d) * (1 + gamma_heading * cos(dtheta))
    lambda_cost: float = 0.4  # per-meter distance discount
    gamma_heading: float = 0.3  # directional-consistency (anti-zigzag) weight
    # Commit to one frontier cluster; switch only if a rival's utility beats
    # the active cluster's by (1 + cluster_hysteresis).
    cluster_hysteresis: float = 0.25
    min_cluster_cells: int = 5
    # Stand-off range as a fraction of camera_range_m when seeding a viewpoint
    # behind a frontier cell, and the share of samples seeded that way.
    standoff_frac_min: float = 0.4
    standoff_frac_max: float = 0.9
    frontier_bias: float = 0.8

    @classmethod
    def from_settings(cls, settings: "AgentSettings") -> "HunterParams":
        return cls(
            robot_radius_m=settings.robot_radius_m,
            fov_rad=math.radians(settings.camera_fov_deg),
            camera_range_m=settings.camera_range_m,
            camera_min_range_m=settings.camera_min_range_m,
            n_fov_rays=settings.n_fov_rays,
            n_scoring_rays=settings.n_scoring_rays,
            n_candidate_directions=settings.n_candidate_directions,
            candidate_step_m=settings.candidate_step_m,
            candidate_min_dist_m=settings.candidate_min_dist_m,
            candidate_max_dist_m=settings.candidate_max_dist_m,
            w_gain=settings.w_gain,
            w_dist=settings.w_dist,
            w_turn=settings.w_turn,
            min_gain_m2=settings.min_gain_m2,
            blacklist_radius_m=settings.blacklist_radius_m,
            coverage_target_fraction=settings.coverage_target_fraction,
            enable_frontier_fallback=settings.enable_frontier_fallback,
            frontier_blacklist_radius_m=settings.frontier_blacklist_radius_m,
            planner_mode=settings.planner_mode,
            n_candidates=settings.n_candidates,
            candidate_min_separation_m=settings.candidate_min_separation_m,
            candidate_min_step_m=settings.candidate_min_step_m,
            nbv_rotate_in_place=settings.nbv_rotate_in_place,
            lambda_cost=settings.lambda_cost,
            gamma_heading=settings.gamma_heading,
            cluster_hysteresis=settings.cluster_hysteresis,
            min_cluster_cells=settings.min_cluster_cells,
            standoff_frac_min=settings.standoff_frac_min,
            standoff_frac_max=settings.standoff_frac_max,
            frontier_bias=settings.frontier_bias,
        )
