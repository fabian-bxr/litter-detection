from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

from litter_agents.interfaces.robodog import OccupancyGrid
from litter_agents.mapping.grid import FREE, OCCUPIED, UNKNOWN, GridMap


class MapProvider(ABC):
    """Source of the static map.

    The file-based provider is the current default; the same interface lets a
    Zenoh- or REST-served map replace it without touching any consumer.
    """

    @abstractmethod
    async def load(self) -> GridMap: ...


class FileMapProvider(MapProvider):
    """Loads a ROS map_server-style map: grayscale PNG + YAML metadata.

    This is the format MOLA's mm2grid emits. Thresholding follows map_server
    semantics: occupancy probability p = (255 - v) / 255 (or v / 255 when
    ``negate``), occupied if p > occupied_thresh, free if p < free_thresh,
    else unknown. The PNG's top row is the map's max-y edge, so the image is
    flipped vertically to get row index increasing with +y.
    """

    def __init__(self, yaml_path: str | Path) -> None:
        self.yaml_path = Path(yaml_path)

    async def load(self) -> GridMap:
        meta = yaml.safe_load(self.yaml_path.read_text())
        image_path = Path(meta["image"])
        if not image_path.is_absolute():
            image_path = self.yaml_path.parent / image_path

        origin = meta.get("origin", [0.0, 0.0, 0.0])
        if len(origin) > 2 and abs(float(origin[2])) > 1e-9:
            raise NotImplementedError(
                f"map origin yaw {origin[2]} != 0 is not supported"
            )
        mode = meta.get("mode", "trinary")
        if mode != "trinary":
            raise NotImplementedError(f"map mode {mode!r} not supported (trinary only)")

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read map image {image_path}")

        occupied_thresh = float(meta.get("occupied_thresh", 0.65))
        free_thresh = float(meta.get("free_thresh", 0.196))
        if int(meta.get("negate", 0)):
            p_occ = img.astype(np.float32) / 255.0
        else:
            p_occ = (255.0 - img.astype(np.float32)) / 255.0

        occ = np.full(img.shape, UNKNOWN, dtype=np.int8)
        occ[p_occ > occupied_thresh] = OCCUPIED
        occ[p_occ < free_thresh] = FREE
        occ = np.flipud(occ).copy()

        grid = GridMap(
            occ=occ,
            resolution=float(meta["resolution"]),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
        )
        logger.info(
            "Loaded map {} ({}x{} cells @ {} m, {} free / {} occupied / {} unknown)",
            image_path.name,
            grid.width,
            grid.height,
            grid.resolution,
            int(grid.free_mask().sum()),
            int(grid.occupied_mask().sum()),
            int(grid.unknown_mask().sum()),
        )
        return grid


class ZenohMapProvider(MapProvider):
    """Map served as a robodog OccupancyGrid message.

    Takes the fetch coroutine instead of a Zenoh session so the class stays
    transport-agnostic; the mission wiring supplies e.g. a one-shot subscriber
    on ``robodog/map/occupancy``.
    """

    def __init__(self, fetch: Callable[[], Awaitable[OccupancyGrid]]) -> None:
        self._fetch = fetch

    async def load(self) -> GridMap:
        return GridMap.from_occupancy_grid(await self._fetch())
