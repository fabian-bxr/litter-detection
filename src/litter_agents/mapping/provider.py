from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import yaml

from .grid import GridMap


class MapProvider(ABC):
    @abstractmethod
    def load(self) -> GridMap: ...


class FileMapProvider(MapProvider):
    """Loads a ROS map_server PNG+YAML pair (MOLA mm2grid output).

    Pixel convention (negate=0):
      occ = (255 - pixel) / 255
      occ < free_thresh   → free  (0)
      occ > occ_thresh    → occupied (100)
      else                → unknown (-1)
    """

    def __init__(self, yaml_path: str | Path) -> None:
        self._yaml_path = Path(yaml_path)

    def load(self) -> GridMap:
        with open(self._yaml_path) as f:
            meta = yaml.safe_load(f)

        img_path = self._yaml_path.parent / meta["image"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Map image not found: {img_path}")

        # PNG origin is top-left; ROS map_server origin is bottom-left
        img = np.flipud(img)

        resolution = float(meta["resolution"])
        origin = meta["origin"]          # [x, y, yaw]
        origin_x = float(origin[0])
        origin_y = float(origin[1])
        # yaw != 0 is not supported yet — raise early rather than silently wrong
        if abs(float(origin[2])) > 1e-6:
            raise NotImplementedError("Map origin yaw != 0 is not supported")

        free_thresh = float(meta.get("free_thresh", 0.196))
        occ_thresh = float(meta.get("occupied_thresh", 0.65))
        negate = int(meta.get("negate", 0))

        pixels = img.astype(np.float32)
        if negate:
            occ = pixels / 255.0
        else:
            occ = (255.0 - pixels) / 255.0

        data = np.full(img.shape, np.int8(-1), dtype=np.int8)
        data[occ < free_thresh] = np.int8(0)
        data[occ > occ_thresh] = np.int8(100)

        return GridMap(
            data=data,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
        )
