from __future__ import annotations

import math

import numpy as np

from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.raycast import visible_cells
from litter_agents.interfaces.robodog import Pose2D, normalize_angle
from litter_agents.mapping.grid import GridMap


class CoverageTracker:
    """Tracks which target cells the camera has actually seen.

    ``seen`` accumulates FoV raycasts from live robot poses. The coverage
    denominator is the set of cells that *can and must* be seen: inside the
    requested area, free on the static map (occupied/unknown cells are opaque
    — they can never be observed), and currently believed reachable.
    """

    # Pose deltas below these don't change visibility at grid resolution.
    _MIN_MOVE_M = 0.02
    _MIN_TURN_RAD = math.radians(1.0)

    def __init__(
        self,
        grid: GridMap,
        target: np.ndarray,
        reachable: np.ndarray,
        params: HunterParams,
    ) -> None:
        self._grid = grid
        self._blocked = grid.blocked_mask()
        self._target = target
        self._reachable = reachable
        self._params = params
        self._max_range_cells = params.camera_range_m / grid.resolution
        self._min_range_cells = params.camera_min_range_m / grid.resolution
        self.seen: np.ndarray = np.zeros_like(target, dtype=bool)
        self._denom: np.ndarray | None = None
        self._last_pose: Pose2D | None = None

    def update(self, pose: Pose2D) -> int:
        """Raycast from ``pose`` and absorb into ``seen``; returns newly seen cells."""
        if self._last_pose is not None:
            moved = pose.distance_to(self._last_pose)
            turned = abs(normalize_angle(pose.theta - self._last_pose.theta))
            if moved < self._MIN_MOVE_M and turned < self._MIN_TURN_RAD:
                return 0
        vis = visible_cells(
            self._blocked,
            self._grid.world_to_grid_f(pose.x, pose.y),
            pose.theta,
            self._params.fov_rad,
            self._max_range_cells,
            self._min_range_cells,
            self._params.n_fov_rays,
        )
        n_new = int((vis & ~self.seen).sum())
        self.seen |= vis
        self._last_pose = pose
        return n_new

    def set_reachable(self, reachable: np.ndarray) -> None:
        self._reachable = reachable
        self._denom = None

    @property
    def reachable(self) -> np.ndarray:
        """Cells in the start's connected free component (frontier search uses it)."""
        return self._reachable

    def denominator(self) -> np.ndarray:
        if self._denom is None:
            self._denom = self._target & self._grid.free_mask() & self._reachable
        return self._denom

    def fraction(self) -> float:
        denom = self.denominator()
        total = int(denom.sum())
        if total == 0:
            return 1.0
        return float((self.seen & denom).sum()) / total

    def unseen_target(self) -> np.ndarray:
        return self.denominator() & ~self.seen

    def denominator_m2(self) -> float:
        return float(self.denominator().sum()) * self._grid.resolution**2
