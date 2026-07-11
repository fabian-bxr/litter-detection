import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from litter_agents.config import AgentSettings
from litter_agents.interfaces.robodog import OccupancyGrid
from litter_agents.mapping.grid import FREE, OCCUPIED, UNKNOWN, GridMap
from litter_agents.mapping.provider import (
    FileMapProvider,
    ZenohMapProvider,
    build_map_provider,
)

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
    assert (grid.height, grid.width) == (594, 336)
    n_free = int(grid.free_mask().sum())
    n_occupied = int(grid.occupied_mask().sum())
    # Sanity: the lab map has a usable amount of free space and real walls.
    assert 30000 < n_free < 50000
    assert n_occupied > 500
    assert grid.unknown_mask().sum() > n_free  # gray background dominates


def test_file_provider_path_is_cwd_independent(tmp_path, monkeypatch):
    """A relative map_yaml_path hangs off the repo root, not the CWD.

    Regression: launching the UI/mission from anywhere but the repo root used to
    resolve 'my_lab_grid.yaml' against the CWD and 404 with "no such file".
    """
    monkeypatch.chdir(tmp_path)
    settings = AgentSettings(map_source="file", map_yaml_path="my_lab_grid.yaml")
    provider = build_map_provider(settings)
    assert isinstance(provider, FileMapProvider)
    assert provider.yaml_path == REPO_ROOT / "my_lab_grid.yaml"
    assert asyncio.run(provider.load()).width > 0


def test_settings_env_file_is_cwd_independent(tmp_path, monkeypatch):
    """AgentSettings reads the repo-root .env wherever it is constructed from.

    Regression: a CWD-relative env_file silently dropped MAP_SOURCE and
    OLLAMA_API_KEY back to their defaults when launched from a subdirectory.
    """
    monkeypatch.chdir(tmp_path)
    env_file = AgentSettings.model_config["env_file"]
    assert Path(env_file).is_absolute()
    assert Path(env_file) == REPO_ROOT / ".env"


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
