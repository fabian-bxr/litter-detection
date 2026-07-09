import math

import numpy as np
import pytest

from litter_agents.hunter.raycast import ray_clearance_cells, visible_cells

FOV = math.radians(70.0)


def empty(h=200, w=200) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def test_wedge_area_matches_analytic():
    blocked = empty()
    vis = visible_cells(
        blocked, (100.0, 100.0), heading_rad=0.0, fov_rad=FOV,
        max_range_cells=50.0, min_range_cells=6.0, n_rays=90,
    )
    expected = 0.5 * FOV * (50.0**2 - 6.0**2)
    assert vis.sum() == pytest.approx(expected, rel=0.10)


def test_heading_rotates_wedge():
    blocked = empty()
    vis = visible_cells(
        blocked, (100.0, 100.0), heading_rad=math.pi / 2, fov_rad=FOV,
        max_range_cells=30.0, min_range_cells=2.0, n_rays=60,
    )
    rows, cols = np.nonzero(vis)
    # Facing +y (= +row): everything visible lies above the origin row.
    assert rows.min() > 100
    assert abs(cols.mean() - 100) < 3


def test_wall_shadows_cells_behind():
    blocked = empty(100, 100)
    blocked[40:61, 30] = True  # vertical wall ahead
    vis = visible_cells(
        blocked, (50.0, 10.0), heading_rad=0.0, fov_rad=FOV,
        max_range_cells=40.0, min_range_cells=1.0, n_rays=90,
    )
    assert vis[50, 25]  # in front of the wall
    assert not vis[50, 30]  # the wall cell itself is not "seen"
    assert not vis[50, 35]  # shadowed
    assert not vis[50, 45]


def test_min_range_blind_spot():
    blocked = empty(50, 50)
    vis = visible_cells(
        blocked, (25.0, 25.0), heading_rad=0.0, fov_rad=FOV,
        max_range_cells=20.0, min_range_cells=5.0, n_rays=60,
    )
    assert not vis[25, 26]  # right next to the robot
    assert vis[25, 35]


def test_out_of_bounds_clipping():
    blocked = empty(20, 20)
    # Looking off the map edge: no error, nothing marked outside.
    vis = visible_cells(
        blocked, (10.0, 18.0), heading_rad=0.0, fov_rad=FOV,
        max_range_cells=30.0, min_range_cells=0.5, n_rays=30,
    )
    assert vis.shape == (20, 20)
    assert vis.sum() > 0  # the sliver before the edge is still seen


def test_clearance_empty_and_walled():
    blocked = empty(50, 50)
    assert ray_clearance_cells(blocked, (25.0, 25.0), 0.0, 20.0) == pytest.approx(20.0)
    blocked[25, 35] = True  # 10 cells ahead
    d = ray_clearance_cells(blocked, (25.0, 25.0), 0.0, 20.0)
    assert 8.5 <= d <= 10.0


def test_clearance_skip_lets_robot_escape():
    blocked = empty(50, 50)
    blocked[25, 26] = True  # blockage right at the robot (inflation artifact)
    d_no_skip = ray_clearance_cells(blocked, (25.0, 25.0), 0.0, 20.0)
    d_skip = ray_clearance_cells(blocked, (25.0, 25.0), 0.0, 20.0, skip_cells=3.0)
    assert d_no_skip < 2.0
    assert d_skip == pytest.approx(20.0)
