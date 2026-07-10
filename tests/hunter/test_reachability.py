import numpy as np

from litter_agents.hunter.reachability import (
    Blacklist,
    DynamicObstacles,
    reachable_mask,
)
from litter_agents.mapping.grid import FREE, GridMap


def test_pocket_behind_wall_excluded():
    free = np.ones((20, 20), dtype=bool)
    free[:, 10] = False  # full vertical wall
    mask = reachable_mask(free, (5, 5))
    assert mask[5, 5]
    assert mask[15, 3]
    assert not mask[5, 15]  # other side of the wall


def test_blocked_start_snaps_to_nearest_free():
    free = np.ones((20, 20), dtype=bool)
    free[4:7, 4:7] = False  # robot stands inside an inflated blob
    mask = reachable_mask(free, (5, 5))
    assert mask.sum() > 0
    assert mask[0, 0]


def test_start_too_far_from_free_space():
    free = np.zeros((50, 50), dtype=bool)
    free[45:, 45:] = True
    mask = reachable_mask(free, (0, 0), max_snap_dist_cells=10.0)
    assert mask.sum() == 0


def test_dynamic_obstacles_inflate_and_split():
    occ = np.full((40, 40), FREE, dtype=np.int8)
    grid = GridMap(occ=occ, resolution=0.1, origin_x=0.0, origin_y=0.0)
    dyn = DynamicObstacles(grid, inflate_radius_m=0.2)
    assert len(dyn) == 0
    dyn.add_disc(2.0, 2.0, radius_m=0.1)
    layer = dyn.layer
    assert layer[20, 20]
    # 0.1 m disc + 0.2 m inflation = 3 cells radius.
    assert layer[20, 22]
    assert not layer[20, 25]


def test_blacklist():
    bl = Blacklist(radius_m=0.5)
    assert not bl.contains(1.0, 1.0)
    bl.add(1.0, 1.0)
    assert bl.contains(1.2, 1.2)
    assert not bl.contains(2.0, 2.0)
    assert len(bl) == 1
