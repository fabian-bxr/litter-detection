"""Tests for hunter/coverage.py"""

import numpy as np
import pytest
from litter_agents.mapping.grid import GridMap
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.hunter.coverage import CoverageTracker


def _make_grid(free_mask: np.ndarray, resolution: float = 0.05) -> GridMap:
    h, w = free_mask.shape
    data = np.where(free_mask, np.int8(0), np.int8(100)).astype(np.int8)
    return GridMap(data=data, resolution=resolution, origin_x=0.0, origin_y=0.0)


def test_fraction_starts_zero():
    free = np.ones((100, 100), dtype=bool)
    grid = _make_grid(free)
    target = np.zeros((100, 100), dtype=bool)
    target[40:60, 40:60] = True
    cov = CoverageTracker(grid, target)
    assert cov.fraction() == 0.0


def test_fraction_reaches_one_when_all_seen():
    free = np.ones((60, 60), dtype=bool)
    grid = _make_grid(free)
    target = np.ones((60, 60), dtype=bool)
    cov = CoverageTracker(grid, target)
    # Manually mark all cells as seen
    cov._seen[:] = True
    assert cov.fraction() == pytest.approx(1.0, abs=1e-6)


def test_denominator_excludes_occupied():
    data = np.zeros((50, 50), dtype=np.int8)
    data[20:30, 20:30] = np.int8(100)  # occupied block
    grid = GridMap(data=data, resolution=0.05, origin_x=0.0, origin_y=0.0)
    target = np.ones((50, 50), dtype=bool)
    cov = CoverageTracker(grid, target)
    denom = cov.denominator_mask()
    assert not denom[25, 25], "Occupied cell must not be in denominator"
    assert denom[5, 5], "Free cell must be in denominator"


def test_denominator_excludes_unknown():
    data = np.full((50, 50), np.int8(-1), dtype=np.int8)
    data[10:40, 10:40] = np.int8(0)  # free patch
    grid = GridMap(data=data, resolution=0.05, origin_x=0.0, origin_y=0.0)
    target = np.ones((50, 50), dtype=bool)
    cov = CoverageTracker(grid, target)
    denom = cov.denominator_mask()
    assert not denom[0, 0], "Unknown cell must not be in denominator"
    assert denom[25, 25], "Free cell must be in denominator"


def test_denominator_excludes_unreachable():
    free = np.ones((50, 50), dtype=bool)
    grid = _make_grid(free)
    target = np.ones((50, 50), dtype=bool)
    cov = CoverageTracker(grid, target)
    reachable = np.zeros((50, 50), dtype=bool)
    reachable[0:25, :] = True
    cov.set_reachable(reachable)
    denom = cov.denominator_mask()
    assert not denom[40, 25], "Unreachable cell excluded"
    assert denom[10, 25], "Reachable cell included"


def test_update_increases_fraction():
    free = np.ones((200, 200), dtype=bool)
    grid = _make_grid(free, resolution=0.05)
    # 5x5 m target centred at (2.5, 2.5)
    target = np.zeros((200, 200), dtype=bool)
    target[50:150, 50:150] = True
    cov = CoverageTracker(grid, target, fov_deg=70.0, range_m=2.0, min_range_m=0.1)
    assert cov.fraction() == 0.0
    cov.update(Pose2D(x=2.5, y=2.5, theta=0.0))
    assert cov.fraction() > 0.0, "Coverage should increase after update"
