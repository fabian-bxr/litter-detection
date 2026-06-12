from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litter_agents.config import AgentSettings


@dataclass(frozen=True)
class HunterParams:
    """Tunables of the exploration algorithm, decoupled from AgentSettings so
    the hunter package stays importable without the zenoh-flavored config."""

    robot_radius_m: float = 0.30  # Go2 half-width ~0.16 m + margin
    fov_rad: float = math.radians(70.0)
    camera_range_m: float = 2.5
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
        )
