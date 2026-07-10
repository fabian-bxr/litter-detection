import math

import numpy as np
import pytest

from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import FREE, GridMap
from litter_agents.mapping.raster import rasterize_area


@pytest.fixture
def grid() -> GridMap:
    # 20 m x 20 m at 5 cm, origin so the world origin sits at the map center.
    occ = np.full((400, 400), FREE, dtype=np.int8)
    return GridMap(occ=occ, resolution=0.05, origin_x=-10.0, origin_y=-10.0)


def mask_centroid_world(mask: np.ndarray, grid: GridMap) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    x = grid.origin_x + (cols.mean() + 0.5) * grid.resolution
    y = grid.origin_y + (rows.mean() + 0.5) * grid.resolution
    return x, y


def test_circle_area(grid):
    spec = SearchAreaSpec(shape="circle", radius_m=3.0)
    mask = rasterize_area(spec, Pose2D(x=0, y=0, theta=0), grid)
    area = mask.sum() * grid.resolution**2
    assert area == pytest.approx(math.pi * 9.0, rel=0.05)


def test_circle_centered_on_robot(grid):
    spec = SearchAreaSpec(shape="circle", radius_m=2.0)
    mask = rasterize_area(spec, Pose2D(x=1.5, y=-2.0, theta=0.7), grid)
    cx, cy = mask_centroid_world(mask, grid)
    assert cx == pytest.approx(1.5, abs=0.1)
    assert cy == pytest.approx(-2.0, abs=0.1)


def test_offset_rotates_with_robot(grid):
    # Robot faces +y; "3 m in front of me" must land at (0, 3).
    spec = SearchAreaSpec(shape="circle", radius_m=1.0, center_dx_m=3.0)
    mask = rasterize_area(spec, Pose2D(x=0, y=0, theta=math.pi / 2), grid)
    cx, cy = mask_centroid_world(mask, grid)
    assert cx == pytest.approx(0.0, abs=0.1)
    assert cy == pytest.approx(3.0, abs=0.1)


def test_rectangle_area_and_orientation(grid):
    spec = SearchAreaSpec(shape="rectangle", width_m=2.0, depth_m=6.0)
    # Robot facing +x: depth extends along x.
    mask = rasterize_area(spec, Pose2D(x=0, y=0, theta=0), grid)
    area = mask.sum() * grid.resolution**2
    assert area == pytest.approx(12.0, rel=0.05)
    rows, cols = np.nonzero(mask)
    x_extent = (cols.max() - cols.min()) * grid.resolution
    y_extent = (rows.max() - rows.min()) * grid.resolution
    assert x_extent == pytest.approx(6.0, abs=0.2)
    assert y_extent == pytest.approx(2.0, abs=0.2)


def test_rectangle_axis_aligned_when_not_rotating(grid):
    spec = SearchAreaSpec(
        shape="rectangle", width_m=2.0, depth_m=6.0, rotate_with_robot=False
    )
    # Heading +y, but the rectangle stays axis-aligned (depth along world x).
    mask = rasterize_area(spec, Pose2D(x=0, y=0, theta=math.pi / 2), grid)
    rows, cols = np.nonzero(mask)
    x_extent = (cols.max() - cols.min()) * grid.resolution
    assert x_extent == pytest.approx(6.0, abs=0.2)


def test_rectangle_rotates_with_robot(grid):
    spec = SearchAreaSpec(shape="rectangle", width_m=2.0, depth_m=6.0)
    mask = rasterize_area(spec, Pose2D(x=0, y=0, theta=math.pi / 2), grid)
    rows, cols = np.nonzero(mask)
    y_extent = (rows.max() - rows.min()) * grid.resolution
    assert y_extent == pytest.approx(6.0, abs=0.2)


def test_polygon_area(grid):
    # Right triangle, legs 4 m: area 8 m².
    spec = SearchAreaSpec(
        shape="polygon", polygon_points=[(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
    )
    mask = rasterize_area(spec, Pose2D(x=0, y=0, theta=0), grid)
    area = mask.sum() * grid.resolution**2
    assert area == pytest.approx(8.0, rel=0.07)


def test_area_clipped_at_map_border(grid):
    # Circle centered near the map edge: roughly half falls outside.
    spec = SearchAreaSpec(shape="circle", radius_m=3.0)
    mask = rasterize_area(spec, Pose2D(x=9.9, y=0, theta=0), grid)
    area = mask.sum() * grid.resolution**2
    assert area < 0.6 * math.pi * 9.0


def test_spec_validation():
    with pytest.raises(ValueError):
        SearchAreaSpec(shape="circle")
    with pytest.raises(ValueError):
        SearchAreaSpec(shape="rectangle", width_m=2.0)
    with pytest.raises(ValueError):
        SearchAreaSpec(shape="polygon", polygon_points=[(0, 0), (1, 1)])
