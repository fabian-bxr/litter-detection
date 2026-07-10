import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from litter_agents.interfaces.robodog import OccupancyGrid
from litter_agents.mapping.grid import FREE, OCCUPIED, UNKNOWN, GridMap
from litter_agents.mapping.provider import FileMapProvider, ZenohMapProvider

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_map(tmp_path: Path, img: np.ndarray, **meta_overrides) -> Path:
    cv2.imwrite(str(tmp_path / "map.png"), img)
    meta = {
        "image": "map.png",
        "resolution": 0.1,
        "origin": [1.0, 2.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    meta.update(meta_overrides)
    yaml_path = tmp_path / "map.yaml"
    lines = []
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    yaml_path.write_text("\n".join(lines))
    return yaml_path


def test_trinary_thresholds_and_flip(tmp_path):
    # Image (top row first): black / white / gray in the top row.
    img = np.full((3, 3), 205, dtype=np.uint8)
    img[0, 0] = 0  # occupied
    img[0, 1] = 254  # free
    img[0, 2] = 127  # unknown (mid gray)
    yaml_path = write_map(tmp_path, img)

    grid = asyncio.run(FileMapProvider(yaml_path).load())
    assert grid.height == 3 and grid.width == 3
    # The image's top row is the max-y edge → row index 2 after the flip.
    assert grid.occ[2, 0] == OCCUPIED
    assert grid.occ[2, 1] == FREE
    assert grid.occ[2, 2] == UNKNOWN
    assert grid.origin_x == pytest.approx(1.0)
    assert grid.origin_y == pytest.approx(2.0)
    assert grid.resolution == pytest.approx(0.1)


def test_negate_flag(tmp_path):
    img = np.zeros((2, 2), dtype=np.uint8)  # black
    yaml_path = write_map(tmp_path, img, negate=1)
    grid = asyncio.run(FileMapProvider(yaml_path).load())
    # With negate, black means p_occ = 0 → free.
    assert (grid.occ == FREE).all()


def test_origin_yaw_unsupported(tmp_path):
    img = np.zeros((2, 2), dtype=np.uint8)
    yaml_path = write_map(tmp_path, img, origin=[0.0, 0.0, 0.5])
    with pytest.raises(NotImplementedError):
        asyncio.run(FileMapProvider(yaml_path).load())


def test_loads_real_lab_grid():
    yaml_path = REPO_ROOT / "my_lab_grid.yaml"
    grid = asyncio.run(FileMapProvider(yaml_path).load())
    assert (grid.height, grid.width) == (225, 405)
    n_free = int(grid.free_mask().sum())
    n_occupied = int(grid.occupied_mask().sum())
    # Sanity: the lab map has a usable amount of free space and real walls.
    assert 3000 < n_free < 30000
    assert n_occupied > 500
    assert grid.unknown_mask().sum() > n_free  # gray background dominates


def test_zenoh_map_provider_uses_fetch():
    occ = np.full((4, 5), FREE, dtype=np.int8)
    occ[1, 1] = OCCUPIED
    og = OccupancyGrid.from_array(occ, resolution=0.2, origin_x=0.0, origin_y=0.0)

    async def fetch() -> OccupancyGrid:
        return og

    grid = asyncio.run(ZenohMapProvider(fetch).load())
    assert isinstance(grid, GridMap)
    assert grid.occ[1, 1] == OCCUPIED
    assert grid.resolution == pytest.approx(0.2)
