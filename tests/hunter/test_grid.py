import numpy as np
import pytest

from litter_agents.mapping.grid import FREE, OCCUPIED, UNKNOWN, GridMap


def make_grid(h=20, w=30, resolution=0.1, origin=(-1.0, -0.5)) -> GridMap:
    occ = np.full((h, w), FREE, dtype=np.int8)
    return GridMap(occ=occ, resolution=resolution, origin_x=origin[0], origin_y=origin[1])


def test_world_grid_roundtrip():
    grid = make_grid()
    x, y = 0.73, 0.42
    row, col = grid.world_to_grid(x, y)
    cx, cy = grid.grid_to_world(row, col)
    assert abs(cx - x) <= grid.resolution / 2
    assert abs(cy - y) <= grid.resolution / 2


def test_origin_is_cell_zero():
    grid = make_grid()
    # A point just inside the bottom-left corner lands in cell (0, 0).
    row, col = grid.world_to_grid(grid.origin_x + 0.01, grid.origin_y + 0.01)
    assert (row, col) == (0, 0)


def test_masks_partition():
    grid = make_grid()
    grid.occ[2, 3] = OCCUPIED
    grid.occ[4, 5] = UNKNOWN
    total = grid.free_mask() | grid.occupied_mask() | grid.unknown_mask()
    assert total.all()
    assert grid.blocked_mask().sum() == 2
    assert grid.blocked_mask()[2, 3] and grid.blocked_mask()[4, 5]


def test_inflation_radius():
    grid = make_grid(h=21, w=21, resolution=0.1, origin=(0.0, 0.0))
    grid.occ[10, 10] = OCCUPIED
    blocked = grid.inflated_blocked(radius_m=0.3)
    # 0.3 m = 3 cells: cells within 3 of the obstacle are blocked.
    assert blocked[10, 13]
    assert blocked[13, 10]
    assert not blocked[10, 15]
    # Unknown inflates too.
    grid2 = make_grid(h=21, w=21, resolution=0.1, origin=(0.0, 0.0))
    grid2.occ[10, 10] = UNKNOWN
    assert grid2.inflated_blocked(radius_m=0.3)[10, 12]


def test_inflation_zero_radius_is_blocked_mask():
    grid = make_grid()
    grid.occ[1, 1] = OCCUPIED
    assert (grid.inflated_blocked(0.0) == grid.blocked_mask()).all()


def test_occupancy_grid_roundtrip():
    grid = make_grid(h=5, w=7, resolution=0.2, origin=(1.5, -2.0))
    grid.occ[0, 0] = OCCUPIED
    grid.occ[4, 6] = UNKNOWN
    og = grid.to_occupancy_grid()
    back = GridMap.from_occupancy_grid(og)
    assert (back.occ == grid.occ).all()
    assert back.resolution == pytest.approx(grid.resolution)
    assert back.origin_x == pytest.approx(grid.origin_x)
    assert back.origin_y == pytest.approx(grid.origin_y)
