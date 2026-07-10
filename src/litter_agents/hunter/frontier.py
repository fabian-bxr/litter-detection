"""Frontier-seeking fallback for the exploration planner.

The greedy scorer (``scoring.generate_candidates``) only proposes single
straight-line moves from the current pose, ranked by *immediate* FoV gain. When
the only unseen reachable area sits around a corner, no single move clears
``min_gain_m2`` and the planner stops with ``no_information_gain`` — even though
that area is reachable.

This module decouples "where to reposition" from "immediate gain": it finds the
nearest unseen reachable cell, paths to a vantage near it through configuration
space (multi-leg, around corners), and returns the *first straight-line leg* of
that path. The robot accepts a zero-gain traversal; once near the frontier the
greedy scorer resumes and sweeps the freshly revealed area.

Pure numpy/scipy, no I/O — same contract as the rest of ``hunter/``.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy import ndimage

from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.raycast import ray_clearance_cells
from litter_agents.hunter.reachability import Blacklist
from litter_agents.hunter.scoring import Candidate
from litter_agents.interfaces.robodog import Pose2D, normalize_angle
from litter_agents.mapping.grid import GridMap

# 8-connectivity keeps diagonal gaps walkable; the per-leg straight-line
# clearance check below is what actually validates the issued goto.
_NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
# Cap dud-vantage blacklisting per call so one stall can't spin forever.
_MAX_FRONTIER_TRIES = 8


def frontier_waypoint(
    pose: Pose2D,
    *,
    grid: GridMap,
    blocked_inflated: np.ndarray,
    reachable: np.ndarray,
    unseen_target: np.ndarray,
    frontier_blacklist: Blacklist,
    params: HunterParams,
) -> Candidate | None:
    """A repositioning leg toward the nearest unseen reachable cell, or None.

    Returns None when nothing observable remains reachable — the caller then
    treats the stall as a genuine ``no_information_gain`` stop.
    """
    h, w = blocked_inflated.shape
    free_cspace = (~blocked_inflated) & reachable

    sr, sc = grid.world_to_grid(pose.x, pose.y)
    if not (0 <= sr < h and 0 <= sc < w):
        return None
    if not free_cspace[sr, sc]:
        snapped = _snap_to_free(free_cspace, (sr, sc), params, grid)
        if snapped is None:
            return None
        sr, sc = snapped

    res = grid.resolution
    bl_mask = _blacklist_mask((h, w), frontier_blacklist, grid)

    for _ in range(_MAX_FRONTIER_TRIES):
        unseen = unseen_target & ~bl_mask
        if not unseen.any():
            return None

        # Vantage cells: walkable and within camera range of an unseen cell.
        # Euclidean standoff ignores occlusion; a vantage that turns out to
        # reveal nothing is caught below and blacklisted.
        dist_to_unseen_m = ndimage.distance_transform_edt(~unseen) * res
        vantage = free_cspace & (dist_to_unseen_m <= params.camera_range_m)
        if not vantage.any():
            return None

        path = _bfs_first_to(free_cspace, (sr, sc), vantage)
        if path is None:
            return None  # no reachable vantage for any remaining frontier

        if len(path) >= 2:
            next_rc = _farthest_clear(pose, path, blocked_inflated, grid, params)
            return _make_leg(pose, next_rc, unseen, grid, params)

        # len 1: already standing on a vantage yet greedy found no gain → this
        # frontier is occluded from here. Blacklist it and try the next one.
        fr, fc = _nearest_true(unseen, (sr, sc))
        fx, fy = grid.grid_to_world(fr, fc)
        frontier_blacklist.add(fx, fy)
        rad = max(1, round(params.frontier_blacklist_radius_m / res))
        cv2.circle(bl_mask.view(np.uint8), (fc, fr), rad, 1, thickness=-1)
        bl_mask = bl_mask.astype(bool)

    return None


def _make_leg(
    pose: Pose2D,
    next_rc: tuple[int, int],
    unseen: np.ndarray,
    grid: GridMap,
    params: HunterParams,
) -> Candidate:
    wx, wy = grid.grid_to_world(*next_rc)
    fr, fc = _nearest_true(unseen, next_rc)
    fx, fy = grid.grid_to_world(fr, fc)
    target = Pose2D(x=wx, y=wy, theta=0.0)
    heading = target.bearing_to(Pose2D(x=fx, y=fy, theta=0.0))  # face the frontier
    d = pose.distance_to(target)
    return Candidate(
        target=Pose2D(x=wx, y=wy, theta=heading),
        distance_m=float(d),
        gain_m2=0.0,  # repositioning move; gain is realized after arrival
        turn_rad=float(abs(normalize_angle(heading - pose.theta))),
        score=float(-params.w_dist * d),  # informational only
    )


def _farthest_clear(
    pose: Pose2D,
    path: list[tuple[int, int]],
    blocked_inflated: np.ndarray,
    grid: GridMap,
    params: HunterParams,
) -> tuple[int, int]:
    """Farthest path node reachable by ONE clear straight leg in config space.

    String-pulls the BFS path into a single straight goto (the robodog nav only
    executes straight segments), capped at ``candidate_max_dist_m`` so far
    frontiers are crossed over several replans rather than one giant goto.
    """
    origin_rc = grid.world_to_grid_f(pose.x, pose.y)
    res = grid.resolution
    robot_radius_cells = params.robot_radius_m / res
    best = path[1]  # adjacent free cell is always reachable
    for r, c in path[1:]:
        wx, wy = grid.grid_to_world(r, c)
        node = Pose2D(x=wx, y=wy, theta=0.0)
        d_m = pose.distance_to(node)
        if d_m > params.candidate_max_dist_m:
            break
        heading = pose.bearing_to(node)
        d_cells = d_m / res
        clear = ray_clearance_cells(
            blocked_inflated, origin_rc, heading, d_cells + 1.0,
            skip_cells=robot_radius_cells,
        )
        if clear >= d_cells:
            best = (r, c)
    return best


def _bfs_first_to(
    free: np.ndarray, start: tuple[int, int], target: np.ndarray
) -> list[tuple[int, int]] | None:
    """BFS over ``free`` from ``start`` to the nearest ``target`` cell.

    Returns the cell path (start..target inclusive) or None if no target cell
    is reachable. Early-exits at the first target cell popped.
    """
    h, w = free.shape
    sr, sc = start
    if target[sr, sc]:
        return [(sr, sc)]

    visited = np.zeros((h, w), dtype=bool)
    parent_r = np.full((h, w), -1, dtype=np.int32)
    parent_c = np.full((h, w), -1, dtype=np.int32)
    visited[sr, sc] = True
    q: deque[tuple[int, int]] = deque(((sr, sc),))
    found: tuple[int, int] | None = None
    while q and found is None:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and free[nr, nc]:
                visited[nr, nc] = True
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                if target[nr, nc]:
                    found = (nr, nc)
                    break
                q.append((nr, nc))
    if found is None:
        return None

    path = [found]
    r, c = found
    while (r, c) != (sr, sc):
        r, c = int(parent_r[r, c]), int(parent_c[r, c])
        path.append((r, c))
    path.reverse()
    return path


def _nearest_true(mask: np.ndarray, rc: tuple[int, int]) -> tuple[int, int]:
    """Nearest True cell in ``mask`` to ``rc`` (Euclidean). ``mask`` is non-empty."""
    cells = np.argwhere(mask)
    d2 = ((cells - np.array(rc)) ** 2).sum(axis=1)
    r, c = cells[int(np.argmin(d2))]
    return int(r), int(c)


def _snap_to_free(
    free: np.ndarray, rc: tuple[int, int], params: HunterParams, grid: GridMap
) -> tuple[int, int] | None:
    """Nearest free-config cell to ``rc`` within ~one robot radius, or None."""
    h, w = free.shape
    reach = int(round(params.robot_radius_m / grid.resolution)) + 4
    r0, c0 = rc
    r_lo, r_hi = max(0, r0 - reach), min(h, r0 + reach + 1)
    c_lo, c_hi = max(0, c0 - reach), min(w, c0 + reach + 1)
    window = free[r_lo:r_hi, c_lo:c_hi]
    if not window.any():
        return None
    local = np.argwhere(window) + np.array([r_lo, c_lo])
    d2 = ((local - np.array(rc)) ** 2).sum(axis=1)
    r, c = local[int(np.argmin(d2))]
    return int(r), int(c)


def _blacklist_mask(
    shape: tuple[int, int], blacklist: Blacklist, grid: GridMap
) -> np.ndarray:
    """Rasterize blacklist discs (world points) into a bool mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(blacklist) == 0:
        return mask.astype(bool)
    rad = max(1, round(blacklist.radius_m / grid.resolution))
    for x, y in blacklist.points:
        row, col = grid.world_to_grid(x, y)
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            cv2.circle(mask, (col, row), rad, 1, thickness=-1)
    return mask.astype(bool)
