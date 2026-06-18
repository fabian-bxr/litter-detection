"""Collision-free path planning to a target pose (the NBV nav bridge).

The robodog nav executes *straight* segments only, so an NBV viewpoint that
isn't on a clear straight line must be reached via a polyline that goes around
obstacles. This plans that polyline in configuration space (BFS over
``~blocked_inflated`` + line-of-sight string-pulling) and returns the minimal
sequence of straight legs, or None when the target is unreachable.

Pure numpy, no I/O — same contract as the rest of ``hunter/``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.raycast import ray_clearance_cells
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap

_NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def plan_path(
    pose: Pose2D,
    target: Pose2D,
    *,
    grid: GridMap,
    blocked_inflated: np.ndarray,
    params: HunterParams,
) -> list[Pose2D] | None:
    """Straight legs from ``pose`` to ``target`` avoiding ``blocked_inflated``.

    Returns ``[target]`` when the direct line is clear, a multi-leg polyline
    (each leg a collision-free straight segment, final pose carrying
    ``target.theta``) when it must go around, or None if ``target`` can't be
    reached through configuration space.
    """
    rr_cells = params.robot_radius_m / grid.resolution
    if _straight_clear(pose, target, grid, blocked_inflated, rr_cells):
        return [Pose2D(x=target.x, y=target.y, theta=target.theta)]

    free = ~blocked_inflated
    start_rc = _snap_to_free(free, grid.world_to_grid(pose.x, pose.y), params, grid)
    target_rc = grid.world_to_grid(target.x, target.y)
    h, w = free.shape
    if start_rc is None or not (0 <= target_rc[0] < h and 0 <= target_rc[1] < w):
        return None
    if not free[target_rc]:
        snapped = _snap_to_free(free, target_rc, params, grid)
        if snapped is None:
            return None
        target_rc = snapped

    cells = _bfs_path(free, start_rc, target_rc)
    if cells is None:
        return None

    nodes = _string_pull(pose, cells, grid, blocked_inflated, rr_cells)
    path: list[Pose2D] = []
    prev = pose
    for rc in nodes:
        wx, wy = grid.grid_to_world(*rc)
        heading = prev.bearing_to(Pose2D(x=wx, y=wy, theta=0.0))
        leg = Pose2D(x=wx, y=wy, theta=heading)
        path.append(leg)
        prev = leg
    # End exactly at the target with its viewpoint orientation (the last BFS
    # node is within half a cell of it; the segment stays clear).
    path[-1] = Pose2D(x=target.x, y=target.y, theta=target.theta)
    return path


def _straight_clear(
    a: Pose2D,
    b: Pose2D,
    grid: GridMap,
    blocked_inflated: np.ndarray,
    robot_radius_cells: float,
) -> bool:
    """True if the straight segment a→b is collision-free in config space."""
    d_cells = a.distance_to(b) / grid.resolution
    if d_cells < 1e-6:
        return True
    clear = ray_clearance_cells(
        blocked_inflated,
        grid.world_to_grid_f(a.x, a.y),
        a.bearing_to(b),
        d_cells + 1.0,
        skip_cells=robot_radius_cells,
        # Sample sub-cell so a leg can't step over a thin obstacle that the
        # executor's denser interpolation would hit (keeps paths drivable).
        step_cells=0.25,
    )
    return clear >= d_cells


def _string_pull(
    pose: Pose2D,
    cells: list[tuple[int, int]],
    grid: GridMap,
    blocked_inflated: np.ndarray,
    robot_radius_cells: float,
) -> list[tuple[int, int]]:
    """Collapse a cell path into the fewest straight legs (line-of-sight pull)."""
    nodes: list[tuple[int, int]] = []
    anchor = pose
    far = 1
    idx = 1
    n = len(cells)
    while idx < n:
        wx, wy = grid.grid_to_world(*cells[idx])
        if _straight_clear(
            anchor, Pose2D(x=wx, y=wy, theta=0.0), grid, blocked_inflated, robot_radius_cells
        ):
            far = idx
            idx += 1
        else:
            nodes.append(cells[far])
            ax, ay = grid.grid_to_world(*cells[far])
            anchor = Pose2D(x=ax, y=ay, theta=0.0)
            idx = far + 1
            far = idx
    if not nodes or nodes[-1] != cells[-1]:
        nodes.append(cells[-1])
    return nodes


def _bfs_path(
    free: np.ndarray, start: tuple[int, int], target: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """BFS over ``free`` from ``start`` to ``target`` (8-connectivity)."""
    h, w = free.shape
    sr, sc = start
    tr, tc = target
    if (sr, sc) == (tr, tc):
        return [(sr, sc)]
    visited = np.zeros((h, w), dtype=bool)
    parent_r = np.full((h, w), -1, dtype=np.int32)
    parent_c = np.full((h, w), -1, dtype=np.int32)
    visited[sr, sc] = True
    q: deque[tuple[int, int]] = deque(((sr, sc),))
    found = False
    while q and not found:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and free[nr, nc]:
                visited[nr, nc] = True
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                if (nr, nc) == (tr, tc):
                    found = True
                    break
                q.append((nr, nc))
    if not found:
        return None
    path = [(tr, tc)]
    r, c = tr, tc
    while (r, c) != (sr, sc):
        r, c = int(parent_r[r, c]), int(parent_c[r, c])
        path.append((r, c))
    path.reverse()
    return path


def _snap_to_free(
    free: np.ndarray, rc: tuple[int, int], params: HunterParams, grid: GridMap
) -> tuple[int, int] | None:
    """Nearest free-config cell to ``rc`` within ~one robot radius, or None."""
    h, w = free.shape
    r0, c0 = rc
    if 0 <= r0 < h and 0 <= c0 < w and free[r0, c0]:
        return r0, c0
    reach = int(round(params.robot_radius_m / grid.resolution)) + 4
    r_lo, r_hi = max(0, r0 - reach), min(h, r0 + reach + 1)
    c_lo, c_hi = max(0, c0 - reach), min(w, c0 + reach + 1)
    window = free[r_lo:r_hi, c_lo:c_hi]
    if not window.any():
        return None
    local = np.argwhere(window) + np.array([r_lo, c_lo])
    d2 = ((local - np.array(rc)) ** 2).sum(axis=1)
    r, c = local[int(np.argmin(d2))]
    return int(r), int(c)
