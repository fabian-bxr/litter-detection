from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.raycast import ray_clearance_cells, visible_cells
from litter_agents.hunter.reachability import Blacklist
from litter_agents.interfaces.robodog import Pose2D, normalize_angle
from litter_agents.mapping.grid import GridMap


@dataclass(frozen=True)
class Candidate:
    target: Pose2D  # theta = travel direction
    distance_m: float
    gain_m2: float  # est. new unseen-target area seen en route + at arrival
    turn_rad: float
    score: float


def generate_candidates(
    pose: Pose2D,
    *,
    grid: GridMap,
    blocked_inflated: np.ndarray,
    blocked_raw: np.ndarray,
    unseen_target: np.ndarray,
    blacklist: Blacklist,
    params: HunterParams,
) -> list[Candidate]:
    """Score straight-line-reachable waypoints by information gain.

    Candidates lie on rays from the current position (the robot only walks
    straight lines). Travel feasibility uses the *inflated* grid — a clear ray
    there doubles as a collision-free corridor check — while visibility uses
    the *raw* grid (sight isn't robot-sized). Because every sample pose along
    a ray shares the travel heading, the visible set is cumulative along the
    ray: one wedge raycast per sample, OR-ed up, gives the en-route + arrival
    gain for each candidate distance.

    Every direction also yields a zero-distance candidate — rotate in place to
    that heading. Rotation needs no corridor, so tight spots where no travel
    candidate survives inflation can still be swept with the camera.
    """
    res = grid.resolution
    origin_rc = grid.world_to_grid_f(pose.x, pose.y)
    robot_radius_cells = params.robot_radius_m / res
    travel_range_cells = (params.candidate_max_dist_m + params.robot_radius_m) / res
    view_range_cells = params.camera_range_m / res
    view_min_cells = params.camera_min_range_m / res

    candidates: list[Candidate] = []
    for phi in np.linspace(0.0, 2.0 * math.pi, params.n_candidate_directions, endpoint=False):
        turn = abs(normalize_angle(phi - pose.theta))
        cos_p, sin_p = math.cos(phi), math.sin(phi)

        # Rotate-in-place candidate: look along phi from where we stand.
        cum = visible_cells(
            blocked_raw,
            origin_rc,
            phi,
            params.fov_rad,
            view_range_cells,
            view_min_cells,
            params.n_scoring_rays,
        )
        gain0_m2 = float((cum & unseen_target).sum()) * res**2
        candidates.append(
            Candidate(
                target=Pose2D(x=pose.x, y=pose.y, theta=float(phi)),
                distance_m=0.0,
                gain_m2=gain0_m2,
                turn_rad=float(turn),
                score=float(params.w_gain * gain0_m2 - params.w_turn * turn),
            )
        )

        clear_cells = ray_clearance_cells(
            blocked_inflated,
            origin_rc,
            phi,
            travel_range_cells,
            # Ignore blockage under the robot itself so it can leave a spot
            # that inflation swallowed (e.g. after stopping near a wall).
            skip_cells=robot_radius_cells,
        )
        d_max = min(clear_cells * res - params.robot_radius_m, params.candidate_max_dist_m)
        if d_max < params.candidate_min_dist_m:
            continue

        for d in np.arange(params.candidate_min_dist_m, d_max + 1e-9, params.candidate_step_m):
            x = pose.x + cos_p * d
            y = pose.y + sin_p * d
            vis = visible_cells(
                blocked_raw,
                grid.world_to_grid_f(x, y),
                phi,
                params.fov_rad,
                view_range_cells,
                view_min_cells,
                params.n_scoring_rays,
            )
            cum = vis if cum is None else (cum | vis)
            if blacklist.contains(x, y):
                continue
            gain_m2 = float((cum & unseen_target).sum()) * res**2
            score = params.w_gain * gain_m2 - params.w_dist * d - params.w_turn * turn
            candidates.append(
                Candidate(
                    target=Pose2D(x=x, y=y, theta=float(phi)),
                    distance_m=float(d),
                    gain_m2=gain_m2,
                    turn_rad=float(turn),
                    score=float(score),
                )
            )
    return candidates
