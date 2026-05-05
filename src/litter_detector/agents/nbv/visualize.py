from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Polygon as MplPolygon

from litter_detector.agents.models import Candidate, Pose
from litter_detector.agents.nbv.geometry import Polygon
from litter_detector.agents.tools.occupancy import OccupancyGrid


def render_debug_png(
    *,
    out_path: Path,
    grid: OccupancyGrid,
    seen_mask: np.ndarray,
    target_mask: np.ndarray,
    polygon: Polygon,
    pose: Pose,
    candidates: list[Candidate],
    chosen: Candidate | None,
    iteration: int,
    coverage: float,
) -> None:
    """Render an iteration snapshot to `out_path` (parents created if needed).

    Layers (bottom to top): occupancy / seen overlay / target outline /
    polygon outline / candidates / chosen / pose arrow.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build an RGB image at the grid's resolution.
    h, w = grid.height, grid.width
    img = np.full((h, w, 3), 200, dtype=np.uint8)  # unknown = light gray
    img[grid.free_mask()] = (245, 245, 245)
    img[grid.occupied_mask()] = (40, 40, 40)
    # Seen overlay: tint free+seen cells light blue
    seen_free = seen_mask & grid.free_mask()
    img[seen_free] = (180, 215, 240)
    # Target outline (cells in target but not seen): pale yellow
    target_unseen = target_mask & ~seen_mask
    img[target_unseen] = (255, 240, 180)

    extent = (
        grid.origin_x,
        grid.origin_x + w * grid.resolution,
        grid.origin_y,
        grid.origin_y + h * grid.resolution,
    )

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")

    poly_xy = list(polygon) + [polygon[0]]
    ax.add_patch(
        MplPolygon(
            poly_xy,
            closed=True,
            fill=False,
            edgecolor="#1f883d",
            linewidth=2.0,
            label="search area",
        )
    )

    for i, c in enumerate(candidates):
        ax.plot(c.pose.x, c.pose.y, "o", color="#1f6feb", markersize=5, alpha=0.6)
        ax.annotate(
            str(i),
            (c.pose.x, c.pose.y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#1f6feb",
        )

    if chosen is not None:
        ax.plot(
            chosen.pose.x,
            chosen.pose.y,
            "*",
            color="#d29922",
            markersize=18,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=f"chosen (gain={chosen.gain:.1%})",
        )

    arrow_len = max(0.3, grid.resolution * 8)
    ax.add_patch(
        FancyArrow(
            pose.x,
            pose.y,
            arrow_len * math.cos(pose.theta),
            arrow_len * math.sin(pose.theta),
            width=0.06,
            head_width=0.18,
            head_length=0.18,
            length_includes_head=True,
            color="#cf222e",
        )
    )

    ax.set_title(f"iter {iteration} — coverage {coverage:.1%}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
