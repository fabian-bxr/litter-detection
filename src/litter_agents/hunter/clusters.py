"""Frontier clustering with commit + hysteresis for the NBV planner.

The greedy ray-scorer stalls and wanders because it re-decides "where to look
next" from scratch every iteration. Clustering the unseen target into connected
regions and *committing* to one (switching only when a rival clearly wins)
gives the robot a region to finish before moving on — this is what stops the
zig-zag / back-jump pattern. Ported from the feature/agent-setup NBV planner,
adapted to GridMap / HunterParams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from litter_agents.hunter.params import HunterParams
from litter_agents.interfaces.robodog import Pose2D, normalize_angle
from litter_agents.mapping.grid import GridMap


@dataclass(frozen=True)
class FrontierCluster:
    """A connected component of unseen target cells.

    ``cells`` is a bool mask the size of the grid; ``centroid_world`` is its
    centroid in world coordinates, used both as the sampling goal and as the
    identity for cross-iteration re-association (by spatial proximity).
    """

    label: int
    cells: np.ndarray
    size: int
    centroid_world: tuple[float, float]


def find_frontier_clusters(
    unseen_target: np.ndarray, grid: GridMap, min_size: int
) -> list[FrontierCluster]:
    """8-connectivity connected components of the unseen target.

    Components below ``min_size`` cells are dropped — specks aren't worth a
    commit/replan and usually fill in opportunistically as raycast shadows.
    """
    if not unseen_target.any():
        return []
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        unseen_target.astype(np.uint8), connectivity=8
    )
    out: list[FrontierCluster] = []
    for lbl in range(1, n_labels):
        size = int(stats[lbl, cv2.CC_STAT_AREA])
        if size < min_size:
            continue
        cx_cell, cy_cell = centroids[lbl]  # (col, row)
        wx, wy = grid.grid_to_world(int(round(cy_cell)), int(round(cx_cell)))
        out.append(
            FrontierCluster(
                label=int(lbl),
                cells=(labels == lbl),
                size=size,
                centroid_world=(wx, wy),
            )
        )
    return out


def _cluster_utility(c: FrontierCluster, pose: Pose2D, params: HunterParams) -> float:
    """U(c) = size * exp(-lambda * d) * (1 + gamma * cos(dtheta))."""
    dx = c.centroid_world[0] - pose.x
    dy = c.centroid_world[1] - pose.y
    d = math.hypot(dx, dy)
    if d > 1e-9:
        bearing = math.atan2(dy, dx)
        heading_factor = 1.0 + params.gamma_heading * math.cos(
            normalize_angle(bearing - pose.theta)
        )
    else:
        heading_factor = 1.0
    return c.size * math.exp(-params.lambda_cost * d) * heading_factor


def pick_active_cluster(
    clusters: list[FrontierCluster],
    pose: Pose2D,
    active_centroid: tuple[float, float] | None,
    params: HunterParams,
) -> tuple[FrontierCluster | None, tuple[float, float] | None]:
    """Pick the cluster to commit to, with hysteresis against the previous one.

    Re-associates the previously active cluster by nearest centroid; keeps it
    unless a rival's utility exceeds ``(1 + cluster_hysteresis)`` times it (or
    the active cluster has effectively vanished). Returns the chosen cluster
    and its centroid; both None when there are no clusters.
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
    same_cluster_radius = 2.0 * params.camera_range_m
    if (nx - ax) ** 2 + (ny - ay) ** 2 > same_cluster_radius**2:
        return best, best.centroid_world  # active cluster gone (covered/split away)

    if best.label == nearest.label:
        return nearest, nearest.centroid_world

    u_best = _cluster_utility(best, pose, params)
    u_active = _cluster_utility(nearest, pose, params)
    if u_best > (1.0 + params.cluster_hysteresis) * u_active:
        return best, best.centroid_world
    return nearest, nearest.centroid_world
