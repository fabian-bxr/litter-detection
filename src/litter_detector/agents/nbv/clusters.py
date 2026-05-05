from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from litter_detector.agents.models import NBVParams, Pose
from litter_detector.agents.tools.occupancy import OccupancyGrid


@dataclass(frozen=True)
class FrontierCluster:
    """A connected component of unseen target cells.

    `cells` is a bool mask the same shape as the grid. `centroid_world` is the
    cluster's geometric centroid in world coordinates — used both as the goal
    of cluster-aware sampling and as the identity for cross-iteration
    persistence (we re-find the "active" cluster by spatial proximity).
    """

    label: int
    cells: np.ndarray
    size: int
    centroid_world: tuple[float, float]


def find_frontier_clusters(
    unseen_target: np.ndarray,
    grid: OccupancyGrid,
    min_size: int,
) -> list[FrontierCluster]:
    """Connected-component cluster the unseen target with 8-connectivity.

    Components below `min_size` cells are dropped — isolated specks aren't
    worth the commit/replan overhead, and they're usually raycast-shadow
    artifacts that fill in opportunistically anyway.
    """
    if not unseen_target.any():
        return []
    binary = unseen_target.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    out: list[FrontierCluster] = []
    for lbl in range(1, n_labels):
        size = int(stats[lbl, cv2.CC_STAT_AREA])
        if size < min_size:
            continue
        cx_cell, cy_cell = centroids[lbl]  # (col, row) in cell space
        wx = grid.origin_x + (float(cx_cell) + 0.5) * grid.resolution
        wy = grid.origin_y + (float(cy_cell) + 0.5) * grid.resolution
        out.append(
            FrontierCluster(
                label=int(lbl),
                cells=(labels == lbl),
                size=size,
                centroid_world=(wx, wy),
            )
        )
    return out


def _cluster_utility(c: FrontierCluster, pose: Pose, params: NBVParams) -> float:
    """U(c) = size * exp(-lambda * d) * (1 + gamma * cos(dtheta))."""
    dx = c.centroid_world[0] - pose.x
    dy = c.centroid_world[1] - pose.y
    d = math.hypot(dx, dy)
    if d > 1e-9:
        bearing = math.atan2(dy, dx)
        dth = math.atan2(
            math.sin(bearing - pose.theta), math.cos(bearing - pose.theta)
        )
        heading_factor = 1.0 + params.gamma_heading * math.cos(dth)
    else:
        heading_factor = 1.0
    return c.size * math.exp(-params.lambda_cost * d) * heading_factor


def pick_active_cluster(
    clusters: list[FrontierCluster],
    pose: Pose,
    active_centroid: tuple[float, float] | None,
    params: NBVParams,
) -> tuple[FrontierCluster | None, tuple[float, float] | None]:
    """Pick the cluster to commit to, with hysteresis against the previous one.

    - Compute utility for every cluster (size × distance discount × heading bonus).
    - If `active_centroid` is set, find the closest current cluster to it
      (re-association across iterations).
    - Switch only if `U(best) > (1 + cluster_hysteresis) * U(active)`.

    Returns (chosen_cluster, its_centroid). Both None if no clusters exist.
    """
    if not clusters:
        return None, None

    best = max(clusters, key=lambda c: _cluster_utility(c, pose, params))

    if active_centroid is None:
        return best, best.centroid_world

    ax, ay = active_centroid
    nearest = min(
        clusters,
        key=lambda c: (c.centroid_world[0] - ax) ** 2
        + (c.centroid_world[1] - ay) ** 2,
    )
    nx, ny = nearest.centroid_world
    # Treat as "same cluster" if its centroid stayed within ~one FOV-range.
    same_cluster_radius = 2.0 * params.fov_range_m
    if (nx - ax) ** 2 + (ny - ay) ** 2 > same_cluster_radius ** 2:
        # Active cluster has effectively disappeared (covered or split far away).
        return best, best.centroid_world

    if best.label == nearest.label:
        return nearest, nearest.centroid_world

    u_best = _cluster_utility(best, pose, params)
    u_active = _cluster_utility(nearest, pose, params)
    if u_best > (1.0 + params.cluster_hysteresis) * u_active:
        return best, best.centroid_world
    return nearest, nearest.centroid_world
