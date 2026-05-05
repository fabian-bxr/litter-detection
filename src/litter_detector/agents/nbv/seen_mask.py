from __future__ import annotations

import numpy as np

from litter_detector.agents.models import NBVParams, Pose
from litter_detector.agents.nbv.fov import raycast_wedge
from litter_detector.agents.tools.occupancy import OccupancyGrid


class SeenMask:
    """Tracks which cells the camera FOV has covered so far.

    The mask is keyed to a fixed grid geometry (resolution, origin, shape).
    If the underlying occupancy grid resizes / reorigins, the caller is
    responsible for calling `rebind` (the simple v1 strategy is to lock the
    geometry at mission start).
    """

    def __init__(self, grid: OccupancyGrid) -> None:
        self.shape = (grid.height, grid.width)
        self.resolution = grid.resolution
        self.origin_x = grid.origin_x
        self.origin_y = grid.origin_y
        self._mask = np.zeros(self.shape, dtype=bool)

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def matches(self, grid: OccupancyGrid) -> bool:
        return (
            self.shape == (grid.height, grid.width)
            and abs(self.resolution - grid.resolution) < 1e-9
            and abs(self.origin_x - grid.origin_x) < 1e-9
            and abs(self.origin_y - grid.origin_y) < 1e-9
        )

    def update(self, pose: Pose, grid: OccupancyGrid, params: NBVParams) -> None:
        """Cast the FOV wedge from `pose` and OR it into the seen mask.

        If the grid geometry has changed, the mask is rebound first.
        """
        if not self.matches(grid):
            self.rebind(grid)
        wedge = raycast_wedge(pose, params, grid)
        self._mask |= wedge

    def rebind(self, new_grid: OccupancyGrid) -> None:
        """Re-anchor the seen mask to a new grid geometry, preserving content.

        Each previously-seen cell's world center is mapped onto the new grid;
        cells that fall out of bounds are dropped. Same-geometry calls are no-ops.
        """
        if self.matches(new_grid):
            return
        new_h, new_w = new_grid.height, new_grid.width
        new_mask = np.zeros((new_h, new_w), dtype=bool)
        if self._mask.any():
            rows, cols = np.where(self._mask)
            xs = self.origin_x + (cols + 0.5) * self.resolution
            ys = self.origin_y + (rows + 0.5) * self.resolution
            new_cols = ((xs - new_grid.origin_x) / new_grid.resolution).astype(int)
            new_rows = ((ys - new_grid.origin_y) / new_grid.resolution).astype(int)
            valid = (
                (new_rows >= 0)
                & (new_rows < new_h)
                & (new_cols >= 0)
                & (new_cols < new_w)
            )
            new_mask[new_rows[valid], new_cols[valid]] = True
        self._mask = new_mask
        self.shape = (new_h, new_w)
        self.resolution = new_grid.resolution
        self.origin_x = new_grid.origin_x
        self.origin_y = new_grid.origin_y

    def coverage_inside(self, target_mask: np.ndarray) -> float:
        """Fraction of `target_mask` cells that are also seen.

        `target_mask` is typically `polygon_mask & ~occupied_mask` so unknown
        cells inside the polygon also count toward coverage.
        """
        total = int(target_mask.sum())
        if total == 0:
            return 1.0
        seen_inside = int((self._mask & target_mask).sum())
        return seen_inside / total

    def unseen_inside(self, target_mask: np.ndarray) -> np.ndarray:
        return target_mask & ~self._mask
