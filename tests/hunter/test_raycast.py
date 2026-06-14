"""Tests for hunter/raycast.py"""

import math
import numpy as np
from litter_agents.mapping.grid import GridMap
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.hunter.raycast import visible_cells


def _open_grid(size: int = 200, resolution: float = 0.05) -> GridMap:
    """All-free grid centred at world origin."""
    half = size * resolution / 2
    return GridMap(
        data=np.zeros((size, size), dtype=np.int8),
        resolution=resolution,
        origin_x=-half,
        origin_y=-half,
    )


def test_wedge_area_approx_analytic():
    """Seen area should be within 10% of the analytic annular sector."""
    grid = _open_grid(400)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    fov, rng, min_r = 70.0, 2.5, 0.3
    mask = visible_cells(pose, grid, fov_deg=fov, range_m=rng, min_range_m=min_r, n_rays=180)

    seen_m2 = mask.sum() * grid.resolution ** 2
    # Annular sector: (π*R² - π*r²) * fov/360
    analytic = (math.pi * rng ** 2 - math.pi * min_r ** 2) * (fov / 360.0)
    assert abs(seen_m2 - analytic) / analytic < 0.10, (
        f"seen={seen_m2:.2f}m² analytic={analytic:.2f}m² diff={abs(seen_m2-analytic)/analytic*100:.1f}%"
    )


def test_wall_shadow():
    """Cells behind a wall should NOT be visible."""
    grid = _open_grid(200)
    # Place a vertical wall at x=1.0 (col ≈ 120 in a 200-cell grid centred at 0)
    wall_col = int((1.0 - grid.origin_x) / grid.resolution)
    grid.data[:, wall_col] = np.int8(100)

    pose = Pose2D(x=0.0, y=0.0, theta=0.0)  # looking right (+x)
    mask = visible_cells(pose, grid, fov_deg=70.0, range_m=2.5, min_range_m=0.1, n_rays=180)

    # Cells 0.5m beyond the wall in the forward direction should be invisible
    check_x, check_y = 1.5, 0.0
    r, c = grid.world_to_grid(check_x, check_y)
    assert grid.in_bounds(r, c), "check cell is out of bounds"
    assert not mask[r, c], f"Cell at ({check_x},{check_y}) behind wall should be invisible"


def test_unknown_blocks_ray():
    """Unknown (-1) cells should block raycasting just like occupied cells."""
    grid = _open_grid(200)
    # Place unknown column at x=1.0
    wall_col = int((1.0 - grid.origin_x) / grid.resolution)
    grid.data[:, wall_col] = np.int8(-1)

    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    mask = visible_cells(pose, grid, fov_deg=70.0, range_m=2.5, min_range_m=0.1, n_rays=180)

    r, c = grid.world_to_grid(1.5, 0.0)
    assert not mask[r, c], "Cell behind unknown wall should be invisible"


def test_min_range_blind_spot():
    """Cells closer than min_range should not appear in the visibility mask."""
    grid = _open_grid(200)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    min_r = 0.5
    mask = visible_cells(pose, grid, fov_deg=70.0, range_m=2.5, min_range_m=min_r, n_rays=90)

    # The robot's own cell should not be visible
    r, c = grid.world_to_grid(0.0, 0.0)
    assert not mask[r, c], "Robot's own cell is inside blind spot"

    # A cell at min_range-0.1 should not be visible
    r2, c2 = grid.world_to_grid(min_r * 0.5, 0.0)
    assert not mask[r2, c2], "Cell inside blind spot should not be visible"


def test_fov_limits_lateral():
    """Cells far outside the FoV should not be visible."""
    grid = _open_grid(400)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)  # facing +x
    mask = visible_cells(pose, grid, fov_deg=70.0, range_m=2.5, min_range_m=0.1, n_rays=90)

    # 90° to the side (perpendicular) should NOT be in the FoV
    r, c = grid.world_to_grid(0.0, 2.0)
    if grid.in_bounds(r, c):
        assert not mask[r, c], "Cell 90° off-axis should be outside FoV"
