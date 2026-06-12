from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage

from litter_agents.mapping.grid import GridMap


def reachable_mask(
    free: np.ndarray,
    start_rc: tuple[int, int],
    max_snap_dist_cells: float = 20.0,
) -> np.ndarray:
    """Connected component of ``free`` containing ``start_rc`` (4-connectivity).

    If the start cell itself is not free (robot hugging a wall after
    inflation), it snaps to the nearest free cell within
    ``max_snap_dist_cells``; beyond that everything counts as unreachable.
    """
    labels, _ = ndimage.label(free)
    row = int(np.clip(start_rc[0], 0, free.shape[0] - 1))
    col = int(np.clip(start_rc[1], 0, free.shape[1] - 1))
    if not free[row, col]:
        free_cells = np.argwhere(free)
        if free_cells.size == 0:
            return np.zeros_like(free)
        d2 = ((free_cells - np.array([row, col])) ** 2).sum(axis=1)
        nearest = int(np.argmin(d2))
        if d2[nearest] > max_snap_dist_cells**2:
            return np.zeros_like(free)
        row, col = free_cells[nearest]
    return labels == labels[row, col]


class DynamicObstacles:
    """Obstacles discovered at runtime (BLOCKED navigation goals).

    Discs are stored already inflated by the robot radius, so
    ``static_inflated | layer`` stays a valid configuration-space mask.
    """

    def __init__(self, grid: GridMap, inflate_radius_m: float) -> None:
        self._grid = grid
        self._inflate_radius_m = inflate_radius_m
        self._layer = np.zeros((grid.height, grid.width), dtype=np.uint8)

    def add_disc(self, x: float, y: float, radius_m: float) -> None:
        row, col = self._grid.world_to_grid(x, y)
        r_cells = max(
            1,
            math.ceil((radius_m + self._inflate_radius_m) / self._grid.resolution),
        )
        cv2.circle(self._layer, (col, row), r_cells, 1, thickness=-1)

    @property
    def layer(self) -> np.ndarray:
        return self._layer.astype(bool)

    def __len__(self) -> int:
        return int(self._layer.any())


class Blacklist:
    """Navigation goals that repeatedly failed; candidates near them are vetoed."""

    def __init__(self, radius_m: float) -> None:
        self._radius2 = radius_m**2
        self._points: list[tuple[float, float]] = []

    def add(self, x: float, y: float) -> None:
        self._points.append((x, y))

    def contains(self, x: float, y: float) -> bool:
        return any(
            (x - px) ** 2 + (y - py) ** 2 <= self._radius2 for px, py in self._points
        )

    def __len__(self) -> int:
        return len(self._points)
