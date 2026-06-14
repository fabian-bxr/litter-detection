"""Tracks which grid cells have been seen during a mission."""

from __future__ import annotations

import math

import numpy as np

from ..interfaces.robodog import Pose2D
from ..mapping.grid import GridMap
from .raycast import visible_cells


class CoverageTracker:
    def __init__(
        self,
        raw_grid: GridMap,
        target_mask: np.ndarray,
        fov_deg: float = 70.0,
        range_m: float = 2.5,
        min_range_m: float = 0.3,
        n_rays: int = 90,
        min_move_m: float = 0.02,
        min_rotate_rad: float = 0.017,
    ) -> None:
        self._grid = raw_grid
        self._target = target_mask.astype(bool)
        self._seen = np.zeros((raw_grid.height, raw_grid.width), dtype=bool)
        self._reachable: np.ndarray | None = None
        self._fov_deg = fov_deg
        self._range_m = range_m
        self._min_range_m = min_range_m
        self._n_rays = n_rays
        self._min_move_m = min_move_m
        self._min_rotate_rad = min_rotate_rad
        self._last_pose: Pose2D | None = None

    def update(self, pose: Pose2D) -> None:
        """OR new visible cells into the seen grid. Skip if robot barely moved."""
        if self._last_pose is not None:
            dx = pose.x - self._last_pose.x
            dy = pose.y - self._last_pose.y
            move = math.sqrt(dx * dx + dy * dy)
            dturn = abs(math.atan2(
                math.sin(pose.theta - self._last_pose.theta),
                math.cos(pose.theta - self._last_pose.theta),
            ))
            if move < self._min_move_m and dturn < self._min_rotate_rad:
                return
        self._last_pose = pose
        self._seen |= visible_cells(
            pose, self._grid,
            self._fov_deg, self._range_m, self._min_range_m, self._n_rays,
        )

    def set_reachable(self, reachable: np.ndarray) -> None:
        self._reachable = reachable.astype(bool)

    @property
    def seen(self) -> np.ndarray:
        return self._seen

    def denominator_mask(self) -> np.ndarray:
        """Cells that count toward coverage: all free cells in the target area."""
        return self._target & (self._grid.data == 0)

    def fraction(self) -> float:
        denom = self.denominator_mask()
        total = int(denom.sum())
        if total == 0:
            return 1.0
        return float((self._seen & denom).sum()) / total
