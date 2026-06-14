from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GridMap:
    """2D occupancy grid.

    data: int8 numpy array (height, width)
        -1 = unknown, 0 = free, 100 = occupied
    resolution: metres per pixel
    origin_x/y: world coords of the bottom-left corner (row=0, col=0)
    """

    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and int(self.data[row, col]) == 0

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) → (row, col). Row 0 = bottom of the map."""
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """(row, col) → world centre of that cell."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def inflate(self, radius_m: float) -> GridMap:
        """Return a new GridMap where every obstacle/unknown cell is expanded by
        radius_m.  The result uses only 0 (navigable) and 100 (not navigable) —
        unknown cells are treated as occupied for navigation purposes.
        """
        radius_px = max(1, int(np.ceil(radius_m / self.resolution)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
        )
        obstacle = (self.data != 0).astype(np.uint8)
        inflated = cv2.dilate(obstacle, kernel)
        new_data = np.where(inflated, np.int8(100), np.int8(0)).astype(np.int8)
        return GridMap(
            data=new_data,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
        )
