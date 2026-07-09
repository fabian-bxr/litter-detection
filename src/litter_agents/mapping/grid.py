from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from litter_agents.interfaces.robodog import OccupancyGrid

UNKNOWN: int = -1
FREE: int = 0
OCCUPIED: int = 100


@dataclass(frozen=True)
class GridMap:
    """2D occupancy grid with world anchoring.

    ``occ`` is int8 (height × width) using the nav_msgs convention
    (-1 unknown / 0 free / 100 occupied). Row index increases with +y;
    ``(origin_x, origin_y)`` is the world position of the corner of cell
    [0][0] (bottom-left).
    """

    occ: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int:
        return self.occ.shape[0]

    @property
    def width(self) -> int:
        return self.occ.shape[1]

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """World coordinates → (row, col). May be out of bounds; see in_bounds."""
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """(row, col) → world coordinates of the cell center."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def world_to_grid_f(self, x: float, y: float) -> tuple[float, float]:
        """World coordinates → continuous (row, col), for raycasting."""
        return (
            (y - self.origin_y) / self.resolution,
            (x - self.origin_x) / self.resolution,
        )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def free_mask(self) -> np.ndarray:
        return self.occ == FREE

    def occupied_mask(self) -> np.ndarray:
        return self.occ == OCCUPIED

    def unknown_mask(self) -> np.ndarray:
        return self.occ == UNKNOWN

    def blocked_mask(self) -> np.ndarray:
        """Cells that block both travel and sight: occupied or unknown."""
        return self.occ != FREE

    def inflated_blocked(self, radius_m: float) -> np.ndarray:
        """Blocked mask dilated by ``radius_m`` (robot radius).

        Unknown counts as blocked — the robot must never be commanded through
        unobserved space. The complement of the result is the configuration
        space for straight-line travel.
        """
        if radius_m <= 0:
            return self.blocked_mask()
        k = 2 * math.ceil(radius_m / self.resolution) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(self.blocked_mask().astype(np.uint8), kernel)
        return dilated.astype(bool)

    @classmethod
    def from_occupancy_grid(cls, og: OccupancyGrid) -> "GridMap":
        return cls(
            occ=og.to_array().copy(),
            resolution=og.resolution,
            origin_x=og.origin_x,
            origin_y=og.origin_y,
        )

    def to_occupancy_grid(self, frame_id: str = "world") -> OccupancyGrid:
        return OccupancyGrid.from_array(
            self.occ,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            frame_id=frame_id,
        )
