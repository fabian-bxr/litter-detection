import math

import numpy as np

from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.reachability import Blacklist
from litter_agents.hunter.scoring import generate_candidates
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import FREE, OCCUPIED, GridMap

PARAMS = HunterParams()


def make_grid(h=200, w=200, res=0.1):
    occ = np.full((h, w), FREE, dtype=np.int8)
    return GridMap(occ=occ, resolution=res, origin_x=-10.0, origin_y=-10.0)


def run(grid, pose, unseen, params=PARAMS, blacklist=None):
    return generate_candidates(
        pose,
        grid=grid,
        blocked_inflated=grid.inflated_blocked(params.robot_radius_m),
        blocked_raw=grid.blocked_mask(),
        unseen_target=unseen,
        blacklist=blacklist or Blacklist(params.blacklist_radius_m),
        params=params,
    )


def test_best_candidate_faces_unseen_region():
    grid = make_grid()
    unseen = np.zeros((200, 200), dtype=bool)
    unseen[:, 100:] = True  # everything at x > 0 is unseen
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    cands = run(grid, pose, unseen)
    best = max(cands, key=lambda c: c.score)
    assert abs(best.target.theta) < math.pi / 2


def test_distance_penalty_orders_equal_gain():
    grid = make_grid()
    unseen = np.zeros((200, 200), dtype=bool)  # nothing to gain anywhere
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    cands = run(grid, pose, unseen)
    same_dir = sorted(
        (c for c in cands if abs(c.target.theta) < 1e-9),
        key=lambda c: c.distance_m,
    )
    assert len(same_dir) >= 2
    # Zero gain everywhere → score strictly decreases with distance.
    scores = [c.score for c in same_dir]
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_turn_penalty_breaks_symmetry():
    grid = make_grid()
    unseen = np.ones((200, 200), dtype=bool)  # symmetric gain everywhere
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    cands = run(grid, pose, unseen)
    best = max(cands, key=lambda c: c.score)
    assert best.turn_rad < math.pi / 2


def test_candidates_stop_short_of_walls():
    grid = make_grid()
    # Wall crossing the +x direction 1 m ahead of the robot.
    wall_col = grid.world_to_grid(1.0, 0.0)[1]
    grid.occ[:, wall_col] = OCCUPIED
    unseen = np.ones((200, 200), dtype=bool)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    cands = run(grid, pose, unseen)
    moving = [c for c in cands if c.distance_m > 0]
    assert moving  # other directions are open
    # No travel target on or beyond the wall, and straight-ahead targets keep
    # a robot radius of clearance.
    assert all(c.target.x < 1.0 for c in moving)
    forward = [c for c in moving if abs(c.target.theta) < 0.1]
    assert all(c.target.x <= 1.0 - PARAMS.robot_radius_m + 1e-6 for c in forward)
    # Rotation candidates exist even when travel is limited.
    assert any(c.distance_m == 0.0 for c in cands)


def test_blacklist_vetoes_targets():
    grid = make_grid()
    unseen = np.ones((200, 200), dtype=bool)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    bl = Blacklist(PARAMS.blacklist_radius_m)
    bl.add(0.75, 0.0)
    cands = run(grid, pose, unseen, blacklist=bl)
    assert all(
        (c.target.x - 0.75) ** 2 + c.target.y**2 > PARAMS.blacklist_radius_m**2
        for c in cands
    )


def test_deterministic():
    grid = make_grid()
    unseen = np.ones((200, 200), dtype=bool)
    pose = Pose2D(x=1.3, y=-0.7, theta=0.4)
    a = run(grid, pose, unseen)
    b = run(grid, pose, unseen)
    assert [(c.target.x, c.target.y, c.score) for c in a] == [
        (c.target.x, c.target.y, c.score) for c in b
    ]
