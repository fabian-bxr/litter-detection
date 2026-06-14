"""Tests for hunter/reachability.py"""

import numpy as np
from litter_agents.mapping.grid import GridMap
from litter_agents.hunter.reachability import reachable_mask, DynamicObstacles, Blacklist


def _grid(data: np.ndarray, res: float = 0.05) -> GridMap:
    return GridMap(data=data.astype(np.int8), resolution=res, origin_x=0.0, origin_y=0.0)


def test_all_free_fully_reachable():
    data = np.zeros((20, 20), dtype=np.int8)
    g = _grid(data)
    reach = reachable_mask(g, 10, 10)
    assert reach.all(), "All-free grid must be fully reachable"


def test_wall_splits_regions():
    data = np.zeros((20, 20), dtype=np.int8)
    data[:, 10] = 100  # vertical wall
    g = _grid(data)
    reach = reachable_mask(g, 5, 5)    # start left of wall
    assert reach[5, 5], "Start cell reachable"
    assert not reach[5, 15], "Right side unreachable from left start"


def test_start_on_obstacle_returns_empty():
    data = np.zeros((20, 20), dtype=np.int8)
    data[10, 10] = 100
    g = _grid(data)
    reach = reachable_mask(g, 10, 10)
    assert not reach.any(), "Start on obstacle → empty mask"


def test_dynamic_obstacle_disc():
    data = np.zeros((100, 100), dtype=np.int8)
    g = _grid(data, res=0.05)
    dyn = DynamicObstacles()
    dyn.add_disc(x=2.5, y=2.5, radius_m=0.5)   # world (2.5, 2.5), r=0.5m
    g2 = dyn.apply_to(g)
    r, c = g2.world_to_grid(2.5, 2.5)
    assert g2.data[r, c] == 100, "Centre of disc must be occupied"


def test_dynamic_obstacle_shrinks_reachable():
    data = np.zeros((100, 100), dtype=np.int8)
    g = _grid(data, res=0.05)
    inflated = g  # already no-obstacle
    dyn = DynamicObstacles()
    # Block the path to the right side (col 80+)
    dyn.add_disc(x=4.0, y=2.5, radius_m=3.0)
    g2 = dyn.apply_to(inflated)
    reach_before = reachable_mask(inflated, 50, 5).sum()
    reach_after = reachable_mask(g2, 50, 5).sum()
    assert reach_after < reach_before, "Dynamic obstacle should reduce reachable area"


def test_blacklist_radius():
    bl = Blacklist(radius_m=1.0)
    bl.add(5.0, 5.0)
    assert bl.is_blacklisted(5.0, 5.0)
    assert bl.is_blacklisted(5.4, 5.4)
    assert not bl.is_blacklisted(6.5, 5.0)
