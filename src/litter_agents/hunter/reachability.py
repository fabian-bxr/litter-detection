"""Flood-fill reachability and dynamic obstacles."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import label

from ..mapping.grid import GridMap


def reachable_mask(inflated_grid: GridMap, start_row: int, start_col: int) -> np.ndarray:
    """Return bool mask of cells reachable from (start_row, start_col) via flood-fill
    on the free space of inflated_grid.
    """
    free = (inflated_grid.data == 0).astype(np.int32)
    if (
        not inflated_grid.in_bounds(start_row, start_col)
        or free[start_row, start_col] == 0
    ):
        return np.zeros((inflated_grid.height, inflated_grid.width), dtype=bool)

    labeled, _ = label(free)
    start_label = int(labeled[start_row, start_col])
    if start_label == 0:
        return np.zeros((inflated_grid.height, inflated_grid.width), dtype=bool)
    return labeled == start_label


class DynamicObstacles:
    """Circular obstacle discs burned in after BLOCKED events."""

    def __init__(self) -> None:
        self._discs: list[tuple[float, float, float]] = []  # (x, y, radius_m)

    def add_disc(self, x: float, y: float, radius_m: float = 0.5) -> None:
        self._discs.append((x, y, radius_m))

    def apply_to(self, grid: GridMap) -> GridMap:
        """Return a new GridMap with all discs burned in as occupied cells."""
        if not self._discs:
            return grid
        data = grid.data.copy()
        for ox, oy, r in self._discs:
            radius_px = max(1, int(np.ceil(r / grid.resolution)))
            cr, cc = grid.world_to_grid(ox, oy)
            rr = np.arange(
                max(0, cr - radius_px), min(grid.height, cr + radius_px + 1)
            )
            cc_range = np.arange(
                max(0, cc - radius_px), min(grid.width, cc + radius_px + 1)
            )
            rg, cg = np.meshgrid(rr, cc_range, indexing="ij")
            inside = (rg - cr) ** 2 + (cg - cc) ** 2 <= radius_px ** 2
            data[rg[inside], cg[inside]] = np.int8(100)
        return GridMap(
            data=data,
            resolution=grid.resolution,
            origin_x=grid.origin_x,
            origin_y=grid.origin_y,
        )

    def __len__(self) -> int:
        return len(self._discs)


class Blacklist:
    """Waypoint positions the planner must avoid."""

    def __init__(self, radius_m: float = 1.0) -> None:
        self._entries: list[tuple[float, float]] = []
        self.radius_m = radius_m

    def add(self, x: float, y: float) -> None:
        self._entries.append((x, y))

    def is_blacklisted(self, x: float, y: float) -> bool:
        for bx, by in self._entries:
            if math.sqrt((x - bx) ** 2 + (y - by) ** 2) < self.radius_m:
                return True
        return False

    def __len__(self) -> int:
        return len(self._entries)
