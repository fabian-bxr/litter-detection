from __future__ import annotations

import math

import numpy as np

from litter_detector.agents.models import NBVParams, Pose
from litter_detector.agents.nbv.candidates import sample_candidate_positions
from litter_detector.agents.nbv.fov import raycast_wedge
from litter_detector.agents.nbv.geometry import polygon_mask, rect_around
from litter_detector.agents.nbv.scoring import best_candidate, score_candidates
from litter_detector.agents.nbv.seen_mask import SeenMask
from litter_detector.agents.tools.occupancy import OccupancyGrid


def _free_grid(h: int = 100, w: int = 100, res: float = 0.1) -> OccupancyGrid:
    return OccupancyGrid(
        data=np.zeros((h, w), dtype=np.int8),
        resolution=res,
        origin_x=0.0,
        origin_y=0.0,
    )


def test_raycast_wedge_marks_cells_in_front() -> None:
    grid = _free_grid()
    params = NBVParams(h_fov_deg=60.0, fov_range_m=2.0)
    pose = Pose(x=5.0, y=5.0, theta=0.0)
    seen = raycast_wedge(pose, params, grid)
    # Cell directly in front should be seen
    r, c = grid.world_to_cell(6.0, 5.0)
    assert seen[r, c]
    # Cell directly behind should NOT
    r, c = grid.world_to_cell(4.0, 5.0)
    assert not seen[r, c]


def test_raycast_wedge_blocked_by_obstacle() -> None:
    grid = _free_grid()
    # Wall at x=5.5 (col 55), y from 4 to 6
    arr = grid.data.copy()
    r0, c_wall = grid.world_to_cell(5.5, 4.5)
    r1, _ = grid.world_to_cell(5.5, 5.5)
    arr[r0:r1 + 1, c_wall] = 100
    blocked = OccupancyGrid(data=arr, resolution=grid.resolution, origin_x=0.0, origin_y=0.0)
    params = NBVParams(h_fov_deg=20.0, fov_range_m=4.0)
    pose = Pose(x=5.0, y=5.0, theta=0.0)
    seen = raycast_wedge(pose, params, blocked)
    # Past the wall must not be seen
    r, c = grid.world_to_cell(8.0, 5.0)
    assert not seen[r, c]


def test_seen_mask_rebind_preserves_overlap_and_drops_out_of_bounds() -> None:
    grid_a = _free_grid(h=20, w=20, res=0.1)  # covers world (0..2, 0..2)
    sm = SeenMask(grid_a)
    # Mark a small block as seen, around world (1.0, 1.0)
    r, c = grid_a.world_to_cell(1.0, 1.0)
    sm.mask[r - 1: r + 2, c - 1: c + 2] = True

    # New grid: same resolution, shifted origin so half the seen block falls outside.
    grid_b = OccupancyGrid(
        data=np.zeros((20, 20), dtype=np.int8),
        resolution=0.1,
        origin_x=1.0,  # was 0.0; world (1..3, 1..3)
        origin_y=1.0,
        frame_id="world",
    )
    sm.rebind(grid_b)
    assert sm.shape == (20, 20)
    assert sm.origin_x == 1.0
    # Cell at world (1.0,1.0) should still be seen on the new grid (it lies on its corner).
    nr, nc = grid_b.world_to_cell(1.0, 1.0)
    if grid_b.in_bounds(nr, nc):
        assert sm.mask[nr, nc]


def test_seen_mask_rebind_higher_resolution() -> None:
    grid_a = _free_grid(h=10, w=10, res=0.2)
    sm = SeenMask(grid_a)
    sm.mask[5, 5] = True  # world ~(1.1, 1.1)

    grid_b = OccupancyGrid(
        data=np.zeros((40, 40), dtype=np.int8),
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        frame_id="world",
    )
    sm.rebind(grid_b)
    assert sm.shape == (40, 40)
    nr, nc = grid_b.world_to_cell(1.1, 1.1)
    assert sm.mask[nr, nc]


def test_seen_mask_coverage_increases_after_update() -> None:
    grid = _free_grid()
    sm = SeenMask(grid)
    poly = rect_around(5.0, 5.0, width=4.0, height=4.0)
    target = polygon_mask(grid, poly) & grid.free_mask()
    assert sm.coverage_inside(target) == 0.0
    sm.update(Pose(x=5.0, y=5.0, theta=0.0), grid, NBVParams(fov_range_m=3.0))
    cov = sm.coverage_inside(target)
    assert 0.0 < cov < 1.0


def test_candidates_stay_in_polygon_and_safe_region() -> None:
    grid = _free_grid()
    params = NBVParams(n_candidates=8, candidate_step_m=1.0)
    poly = rect_around(5.0, 5.0, width=4.0, height=4.0)
    safe = grid.inflate_obstacles(0.0)
    rng = np.random.default_rng(42)
    positions = sample_candidate_positions(
        Pose(x=5.0, y=5.0, theta=0.0), grid, poly, params, rng, safe
    )
    assert len(positions) > 0
    for x, y in positions:
        assert 3.0 <= x <= 7.0
        assert 3.0 <= y <= 7.0
        assert math.hypot(x - 5.0, y - 5.0) <= 1.0 + 1e-6


def test_score_picks_orientation_into_unseen() -> None:
    grid = _free_grid()
    params = NBVParams(h_fov_deg=60.0, fov_range_m=2.0, lambda_cost=0.0)
    poly = rect_around(5.0, 5.0, width=4.0, height=4.0)
    target = polygon_mask(grid, poly) & grid.free_mask()
    # Position the candidate at (5,3) — unseen mass is mostly to the +y side.
    candidates = score_candidates(
        Pose(x=5.0, y=3.0, theta=0.0),
        [(5.0, 3.0)],
        grid,
        params,
        unseen_target=target,
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c.gain > 0.0
    # Best orientation should point roughly into +y (theta ~ pi/2)
    # Allow any of the 8 sampled angles closest to pi/2.
    assert math.cos(c.pose.theta - math.pi / 2) > 0.5


def test_best_candidate_returns_none_when_no_gain() -> None:
    candidates = []
    assert best_candidate(candidates) is None
