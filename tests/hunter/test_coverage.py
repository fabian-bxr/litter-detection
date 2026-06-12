import numpy as np

from litter_agents.hunter.coverage import CoverageTracker
from litter_agents.hunter.params import HunterParams
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import FREE, OCCUPIED, UNKNOWN, GridMap

PARAMS = HunterParams(camera_range_m=2.0, camera_min_range_m=0.2)


def make_tracker(h=100, w=100, res=0.1):
    occ = np.full((h, w), FREE, dtype=np.int8)
    grid = GridMap(occ=occ, resolution=res, origin_x=0.0, origin_y=0.0)
    target = np.ones((h, w), dtype=bool)
    reachable = np.ones((h, w), dtype=bool)
    return grid, CoverageTracker(grid, target, reachable, PARAMS)


def test_update_marks_cells_and_short_circuits():
    _, cov = make_tracker()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    n1 = cov.update(pose)
    assert n1 > 0
    # Identical pose: visibility unchanged, raycast skipped.
    assert cov.update(pose) == 0
    # Turning re-raycasts and sees new cells.
    assert cov.update(Pose2D(x=5.0, y=5.0, theta=2.0)) > 0


def test_denominator_excludes_opaque_cells():
    grid, _ = make_tracker()
    grid.occ[10, 10] = OCCUPIED
    grid.occ[20, 20] = UNKNOWN
    target = np.ones_like(grid.occ, dtype=bool)
    reachable = np.ones_like(target)
    cov = CoverageTracker(grid, target, reachable, PARAMS)
    denom = cov.denominator()
    assert not denom[10, 10]
    assert not denom[20, 20]
    assert denom[50, 50]


def test_shrinking_reachability_raises_fraction():
    _, cov = make_tracker()
    cov.update(Pose2D(x=5.0, y=5.0, theta=0.0))
    before = cov.fraction()
    smaller = np.zeros_like(cov.seen)
    smaller[40:70, 40:70] = True
    cov.set_reachable(smaller)
    assert cov.denominator().sum() == 30 * 30
    assert cov.fraction() >= before


def test_empty_denominator_is_complete():
    grid, _ = make_tracker()
    target = np.zeros_like(grid.occ, dtype=bool)
    cov = CoverageTracker(grid, target, np.ones_like(target), PARAMS)
    assert cov.fraction() == 1.0


def test_unseen_target_shrinks_with_updates():
    _, cov = make_tracker()
    before = int(cov.unseen_target().sum())
    cov.update(Pose2D(x=5.0, y=5.0, theta=0.0))
    assert int(cov.unseen_target().sum()) < before
