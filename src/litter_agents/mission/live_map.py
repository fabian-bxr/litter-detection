"""Live matplotlib visualisation for a running litter-search mission.

Shows the occupancy map, search area (blue), seen coverage (green),
robot position (blue dot), and current waypoint target (yellow star).
Refreshes every 0.5 s from within the asyncio event loop.
"""

from __future__ import annotations

import asyncio

import numpy as np

from ..hunter.coverage import CoverageTracker
from ..mapping.grid import GridMap
from .pose_tracker import ZenohPoseTracker

_REFRESH_S = 0.5


class LiveMap:
    """Asyncio task that keeps a matplotlib figure in sync with mission state."""

    def __init__(
        self,
        raw_grid: GridMap,
        coverage: CoverageTracker,
        pose_tracker: ZenohPoseTracker,
        area_mask: np.ndarray,
    ) -> None:
        self._grid = raw_grid
        self._coverage = coverage
        self._pose_tracker = pose_tracker
        self._area_mask = area_mask.astype(bool)
        self._waypoint: tuple[float, float] | None = None
        self._wp_count = 0
        self._dist_m = 0.0
        # matplotlib objects — created lazily in _setup()
        self._plt = None
        self._fig = None
        self._ax = None
        self._seen_img = None
        self._robot_dot = None
        self._wp_marker = None

    # ------------------------------------------------------------------
    # State setters (called from the mission loop)
    # ------------------------------------------------------------------

    def set_waypoint(self, x: float | None, y: float | None) -> None:
        self._waypoint = (x, y) if x is not None else None

    def update_stats(self, wp_count: int, dist_m: float) -> None:
        self._wp_count = wp_count
        self._dist_m = dist_m

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extent(self) -> list[float]:
        g = self._grid
        return [
            g.origin_x,
            g.origin_x + g.width * g.resolution,
            g.origin_y,
            g.origin_y + g.height * g.resolution,
        ]

    def _setup(self) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt

        h, w = self._grid.data.shape
        extent = self._extent()

        plt.ion()
        fig, ax = plt.subplots(figsize=(11, 7))
        self._fig = fig
        self._ax = ax
        try:
            fig.canvas.manager.set_window_title("Litter Mission — Live Map")
        except Exception:
            pass

        # --- static layers ---

        # 1. Base map: free (light grey) / unknown (mid grey) / wall (dark)
        base = np.full((h, w, 4), [120, 120, 120, 255], dtype=np.uint8)
        base[self._grid.data == 0] = [210, 210, 210, 255]
        base[self._grid.data == 100] = [40, 40, 40, 255]
        ax.imshow(base, origin="lower", extent=extent)

        # 2. Search area overlay (blue, semi-transparent)
        area_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        search_cells = self._area_mask & (self._grid.data == 0)
        area_rgba[search_cells] = [30, 120, 220, 60]
        ax.imshow(area_rgba, origin="lower", extent=extent)

        # --- dynamic layers ---

        # 3. Seen coverage (green) — updated every frame
        seen_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        self._seen_img = ax.imshow(seen_rgba, origin="lower", extent=extent)

        # 4. Robot dot + waypoint star
        self._robot_dot, = ax.plot([], [], "bo", markersize=10, label="Robot", zorder=6)
        self._wp_marker, = ax.plot([], [], "y*", markersize=16, label="Waypoint", zorder=6)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        fig.tight_layout()
        plt.show(block=False)

    def _redraw(self) -> None:
        if self._fig is None:
            return

        h, w = self._grid.data.shape

        # Seen overlay
        seen = self._coverage.seen
        seen_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        seen_rgba[seen] = [0, 200, 60, 110]
        self._seen_img.set_data(seen_rgba)

        # Robot
        pose = self._pose_tracker.current()
        self._robot_dot.set_data([pose.x], [pose.y])

        # Waypoint
        if self._waypoint:
            self._wp_marker.set_data([self._waypoint[0]], [self._waypoint[1]])
        else:
            self._wp_marker.set_data([], [])

        cov_pct = self._coverage.fraction() * 100
        self._ax.set_title(
            f"Coverage: {cov_pct:.1f}%  |  WPs: {self._wp_count}"
            f"  |  dist: {self._dist_m:.1f} m",
            fontsize=11,
        )
        self._fig.canvas.draw_idle()
        self._plt.pause(0.02)

    # ------------------------------------------------------------------
    # Asyncio entry point
    # ------------------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        """Long-running coroutine — launch with asyncio.create_task(live_map.run(stop))."""
        self._setup()
        while not stop.is_set():
            self._redraw()
            await asyncio.sleep(_REFRESH_S)
        if self._plt and self._fig:
            self._plt.close(self._fig)
