from __future__ import annotations

import numpy as np

from litter_detector.agents.tools.occupancy import FakeOccupancyClient, OccupancyGrid


def _grid(data: np.ndarray, resolution: float = 0.1) -> OccupancyGrid:
    return OccupancyGrid(
        data=data.astype(np.int8),
        resolution=resolution,
        origin_x=0.0,
        origin_y=0.0,
    )


def test_world_cell_round_trip() -> None:
    g = _grid(np.zeros((10, 10), dtype=np.int8), resolution=0.5)
    row, col = g.world_to_cell(1.25, 2.75)
    assert (row, col) == (5, 2)
    x, y = g.cell_to_world(row, col)
    # cell center
    assert abs(x - 1.25) < 1e-9
    assert abs(y - 2.75) < 1e-9


def test_free_and_occupied_masks() -> None:
    arr = np.array([[0, -1, 100], [0, 0, 100]], dtype=np.int8)
    g = _grid(arr)
    assert g.free_mask().sum() == 3
    assert g.occupied_mask().sum() == 2


def test_inflate_obstacles_grows_blocked_region() -> None:
    arr = np.zeros((7, 7), dtype=np.int8)
    arr[3, 3] = 100
    g = _grid(arr, resolution=0.1)
    safe_no_inflate = g.inflate_obstacles(0.0)
    safe_inflated = g.inflate_obstacles(0.2)  # 2 cells
    # Inflation should remove more cells than no inflation
    assert safe_inflated.sum() < safe_no_inflate.sum()
    # The obstacle cell itself is never safe
    assert not safe_inflated[3, 3]
    # A cell 2 away on cardinal axis should be excluded
    assert not safe_inflated[3, 5]


def test_fake_occupancy_client() -> None:
    fake = FakeOccupancyClient()
    assert fake.get(timeout=0.0) is None
    g = _grid(np.zeros((4, 4), dtype=np.int8))
    fake.set(g)
    assert fake.get(timeout=0.0) is g
