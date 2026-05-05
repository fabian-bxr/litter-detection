from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import zenoh
from loguru import logger

from litter_detector.config import Settings


@dataclass(frozen=True)
class OccupancyGrid:
    """Snapshot of the occupancy grid in world frame.

    Cell values: -1 unknown, 0 free, 100 occupied.
    """

    data: np.ndarray  # shape (height, width), dtype int8
    resolution: float  # meters per cell
    origin_x: float  # world X of cell column 0
    origin_y: float  # world Y of cell row 0
    frame_id: str = "world"

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """World (x,y) → (row, col). May be out of bounds."""
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Cell center in world coordinates."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def free_mask(self) -> np.ndarray:
        return self.data == 0

    def occupied_mask(self) -> np.ndarray:
        return self.data == 100

    def inflate_obstacles(self, radius_m: float) -> np.ndarray:
        """Boolean mask: True where a cell is free AND outside an inflated obstacle.

        Used as the candidate-sampling region.
        """
        occ = self.occupied_mask()
        if radius_m <= 0:
            return self.free_mask() & ~occ
        r = max(1, int(round(radius_m / self.resolution)))
        # Square dilation via cumulative max — sufficient for v1 obstacle inflation.
        h, w = occ.shape
        dilated = occ.copy()
        for _ in range(r):
            shifted = np.zeros_like(dilated)
            shifted[:-1, :] |= dilated[1:, :]
            shifted[1:, :] |= dilated[:-1, :]
            shifted[:, :-1] |= dilated[:, 1:]
            shifted[:, 1:] |= dilated[:, :-1]
            dilated |= shifted
        return self.free_mask() & ~dilated


class OccupancySource(Protocol):
    def get(self, timeout: float = 1.0) -> OccupancyGrid | None: ...


def _decode_grid(data: dict) -> OccupancyGrid:
    raw = base64.b64decode(data["data"])
    arr = np.frombuffer(raw, dtype=np.int8).reshape(int(data["height"]), int(data["width"]))
    return OccupancyGrid(
        data=arr,
        resolution=float(data["resolution"]),
        origin_x=float(data["origin_x"]),
        origin_y=float(data["origin_y"]),
        frame_id=str(data.get("frame_id", "world")),
    )


class OccupancyClient:
    """Subscribes to the occupancy grid and caches the latest snapshot."""

    def __init__(self, session: zenoh.Session | None = None) -> None:
        self._owns_session = session is None
        self._session = session if session is not None else zenoh.open(Settings.zenoh_config())
        topic = Settings.topics().robot.occupancy
        self._lock = threading.Lock()
        self._latest: OccupancyGrid | None = None
        self._event = threading.Event()
        self._sub = self._session.declare_subscriber(topic, self._on_msg)
        logger.info(f"OccupancyClient subscribed to {topic}")

    def _on_msg(self, sample: zenoh.Sample) -> None:
        try:
            data = json.loads(bytes(sample.payload).decode("utf-8"))
            grid = _decode_grid(data)
        except Exception as e:
            logger.warning(f"OccupancyClient: failed to parse occupancy message: {e}")
            return
        with self._lock:
            self._latest = grid
        self._event.set()

    def get(self, timeout: float = 1.0) -> OccupancyGrid | None:
        with self._lock:
            if self._latest is not None:
                return self._latest
        if not self._event.wait(timeout):
            return None
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._sub.undeclare()
        if self._owns_session:
            self._session.close()


class FakeOccupancyClient:
    """In-memory occupancy grid for tests."""

    def __init__(self, grid: OccupancyGrid | None = None) -> None:
        self._grid = grid

    def set(self, grid: OccupancyGrid) -> None:
        self._grid = grid

    def get(self, timeout: float = 1.0) -> OccupancyGrid | None:
        return self._grid

    def close(self) -> None:
        return
