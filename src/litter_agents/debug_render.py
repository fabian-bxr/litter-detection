"""Debug rendering for the exploration loop.

Draws the static map, the requested search area, accumulated camera coverage,
the driven trajectory and the current robot pose into PNG frames. Shared
verbatim between the offline sim (``uv run litter-sim``) and a real mission so
both produce the same kind of path-planning debug images under ``runs/``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap


class TrajectoryRenderer:
    """Draws map + coverage + trajectory; +y is up in the saved images."""

    def __init__(
        self, grid: GridMap, target: np.ndarray, out_dir: Path, scale: int = 3
    ) -> None:
        self._grid = grid
        self._scale = scale
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = np.full((grid.height, grid.width, 3), 128, dtype=np.uint8)
        base[grid.free_mask()] = (255, 255, 255)
        base[grid.occupied_mask()] = (0, 0, 0)
        contours, _ = cv2.findContours(
            target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self._base = base
        self._target_contours = contours
        self.trajectory: list[tuple[float, float]] = []
        self._n_frames = 0

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        row, col = self._grid.world_to_grid(x, y)
        return col, row

    def render(
        self,
        seen: np.ndarray,
        pose: Pose2D,
        *,
        reachable: np.ndarray | None = None,
        obstacles: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compose one frame.

        ``reachable`` is the coverage denominator (reachable ∩ target ∩ free) —
        the area the robot can and should cover; it is tinted blue so the
        unseen-but-reachable remainder stands apart from free-but-unreachable
        ground. ``obstacles`` (e.g. discovered dynamic obstacles) is overlaid
        in red when given — useful for debugging a real run.
        """
        img = self._base.copy()
        if reachable is not None:
            img[reachable] = (
                img[reachable] * 0.5 + np.array([220, 150, 60]) * 0.5
            ).astype(np.uint8)
        img[seen] = (img[seen] * 0.4 + np.array([80, 200, 80]) * 0.6).astype(np.uint8)
        if obstacles is not None:
            img[obstacles] = (
                img[obstacles] * 0.5 + np.array([0, 0, 220]) * 0.5
            ).astype(np.uint8)
        cv2.drawContours(img, self._target_contours, -1, (255, 120, 0), 1)
        if len(self.trajectory) > 1:
            pts = np.array(
                [self._to_px(x, y) for x, y in self.trajectory], dtype=np.int32
            )
            cv2.polylines(img, [pts], False, (0, 0, 255), 1)
        px = self._to_px(pose.x, pose.y)
        cv2.circle(img, px, 2, (255, 0, 255), -1)
        tip = self._to_px(
            pose.x + 0.5 * np.cos(pose.theta), pose.y + 0.5 * np.sin(pose.theta)
        )
        cv2.line(img, px, tip, (255, 0, 255), 1)
        img = np.flipud(img)  # display with +y up
        return cv2.resize(
            img,
            (img.shape[1] * self._scale, img.shape[0] * self._scale),
            interpolation=cv2.INTER_NEAREST,
        )

    def save_frame(
        self,
        seen: np.ndarray,
        pose: Pose2D,
        *,
        reachable: np.ndarray | None = None,
        obstacles: np.ndarray | None = None,
        name: str | None = None,
    ) -> Path:
        """Render and write a frame; returns the path. Auto-numbers if ``name``
        is omitted."""
        filename = name or f"frame_{self._n_frames:04d}.png"
        path = self.out_dir / filename
        cv2.imwrite(
            str(path),
            self.render(seen, pose, reachable=reachable, obstacles=obstacles),
        )
        if name is None:
            self._n_frames += 1
        return path


# ── Live UI overlay ───────────────────────────────────────────────────────────

# Translucent BGRA fills for the map overlay the web UI paints on top of the
# static map image. Same idea as TrajectoryRenderer's tints, but rendered as a
# transparent RGBA layer (robot/path/pose are drawn as vector layers in the
# browser, so only the area masks live here).
_SEEN_BGRA = (120, 211, 52, 130)       # green   — camera has observed this
_REMAINING_BGRA = (255, 132, 56, 105)  # blue    — reachable target, still unseen
_OBSTACLE_BGRA = (68, 68, 239, 175)    # red     — discovered dynamic obstacle
_OUTLINE_BGR = (238, 211, 34)          # cyan    — requested search-area boundary


def _paint(bgra: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int, int]) -> None:
    if mask.any():
        bgra[mask] = colour


def render_coverage_overlay(
    grid: GridMap,
    target: np.ndarray,
    seen: np.ndarray,
    denominator: np.ndarray,
    obstacles: np.ndarray | None = None,
) -> bytes:
    """A world-aligned translucent RGBA PNG of the current exploration state.

    Mirrors the ``runs/`` debug frames — green where the camera has swept,
    blue for the reachable-but-unseen remainder, red for discovered obstacles,
    a cyan outline for the requested area — but as a transparent overlay the UI
    stacks on the static map image. The array is flipped to map_server
    orientation (row 0 = max y) so it lines up pixel-for-pixel with the map PNG
    Leaflet already renders. Returns encoded PNG bytes.
    """
    bgra = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
    _paint(bgra, denominator & ~seen, _REMAINING_BGRA)
    _paint(bgra, seen, _SEEN_BGRA)
    if obstacles is not None:
        _paint(bgra, obstacles, _OBSTACLE_BGRA)

    # Alpha-bleed: let the fill colours seep one step into their transparent
    # neighbours so the browser's bilinear upscaling doesn't blend the edges
    # toward black (RGB stays coloured; alpha stays 0, so nothing new shows).
    transparent = bgra[..., 3] == 0
    if transparent.any() and not transparent.all():
        bled = cv2.dilate(bgra[..., :3], np.ones((3, 3), np.uint8), iterations=2)
        bgra[..., :3][transparent] = bled[transparent]

    contours, _ = cv2.findContours(
        target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(bgra, contours, -1, (*_OUTLINE_BGR, 235), 2)

    bgra = np.flipud(bgra)  # match the map PNG (top row = max y)
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:  # pragma: no cover — cv2 PNG encode does not realistically fail
        raise RuntimeError("failed to encode coverage overlay PNG")
    return buf.tobytes()