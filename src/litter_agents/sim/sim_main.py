"""Offline exploration simulation — CLI: uv run litter-sim [--circle R | --rect W D]"""

from __future__ import annotations

import argparse
import asyncio
import math

import cv2
import numpy as np

from ..config import AgentSettings, REPO_ROOT
from ..hunter.coverage import CoverageTracker
from ..hunter.planner import ExplorationPlanner
from ..interfaces.robodog import NavigationState, Pose2D
from ..mapping.provider import FileMapProvider
from ..mapping.raster import AreaSpec, rasterize_area
from .fake_nav import FakeNav, FakePoseSource


def _find_free_start(grid, hint: Pose2D) -> Pose2D:
    """Return a free cell closest to hint.x/y (used when (0,0) lands in unknown)."""
    free = np.argwhere(grid.data == 0)
    if len(free) == 0:
        return hint
    hr, hc = grid.world_to_grid(hint.x, hint.y)
    dists = (free[:, 0] - hr) ** 2 + (free[:, 1] - hc) ** 2
    best = free[int(np.argmin(dists))]
    x, y = grid.grid_to_world(int(best[0]), int(best[1]))
    return Pose2D(x=x, y=y, theta=0.0)


def _render(
    grid,
    coverage: CoverageTracker,
    trajectory: list[Pose2D],
    target: Pose2D | None,
    area_mask: np.ndarray,
    scale: int = 3,
) -> np.ndarray:
    h, w = grid.height, grid.width
    base = np.where(grid.data == 0, np.uint8(255),
           np.where(grid.data == 100, np.uint8(0), np.uint8(128)))
    frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # Search area: light blue tint
    tint = frame[area_mask].astype(np.int16)
    tint[:, 0] = np.clip(tint[:, 0] + 60, 0, 255)
    frame[area_mask] = tint.astype(np.uint8)

    # Seen free cells: green overlay
    seen_free = coverage.seen & (grid.data == 0)
    frame[seen_free] = (frame[seen_free].astype(np.float32) * 0.4 + np.array([0, 100, 0])).astype(np.uint8)

    # Flip: row 0 = world bottom → display at bottom
    frame = np.flipud(frame)
    frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def to_px(x: float, y: float) -> tuple[int, int]:
        r, c = grid.world_to_grid(x, y)
        return int(c * scale + scale // 2), int((h - 1 - r) * scale + scale // 2)

    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            cv2.line(frame, to_px(trajectory[i - 1].x, trajectory[i - 1].y),
                     to_px(trajectory[i].x, trajectory[i].y), (0, 165, 255), 1)

    if target is not None:
        cv2.drawMarker(frame, to_px(target.x, target.y), (0, 0, 220), cv2.MARKER_CROSS, 14, 2)

    if trajectory:
        p = trajectory[-1]
        px, py = to_px(p.x, p.y)
        cv2.circle(frame, (px, py), 5, (220, 30, 30), -1)
        ax = int(px + math.cos(p.theta) * 14)
        ay = int(py - math.sin(p.theta) * 14)
        cv2.arrowedLine(frame, (px, py), (ax, ay), (220, 30, 30), 2, tipLength=0.4)

    return frame


async def _run(args: argparse.Namespace) -> None:
    settings = AgentSettings()
    grid = FileMapProvider(settings.map_path).load()
    inflated = grid.inflate(settings.robot_radius_m)

    if args.circle is not None:
        spec = AreaSpec(shape="circle", radius_m=float(args.circle))
    elif args.rect is not None:
        spec = AreaSpec(shape="rectangle", width_m=float(args.rect[0]), depth_m=float(args.rect[1]))
    else:
        spec = AreaSpec(shape="circle", radius_m=6.0)

    start = _find_free_start(inflated, Pose2D(x=0.0, y=0.0, theta=0.0))
    area_mask = rasterize_area(spec, start, grid)

    coverage = CoverageTracker(
        grid, area_mask,
        fov_deg=settings.fov_deg,
        range_m=settings.seen_range_m,
        min_range_m=settings.camera_min_range_m,
    )
    pose_source = FakePoseSource(start)
    nav = FakeNav(pose_source, speed=0.5, on_step=coverage.update)
    planner = ExplorationPlanner(grid, inflated, coverage, settings)

    out_dir = REPO_ROOT / "runs" / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory: list[Pose2D] = [start]
    frame_idx = 0
    coverage.update(start)

    denom_cells = int(coverage.denominator_mask().sum())
    print(f"Start: ({start.x:.2f}, {start.y:.2f})  area={spec.shape}  "
          f"denom_cells={denom_cells}  area_m²={denom_cells * grid.resolution**2:.1f}")

    while not planner.done():
        pose = pose_source.current()
        candidate = planner.next_waypoint(pose)
        if candidate is None:
            print(f"  No candidate (consec_low={planner.consecutive_low_gain})")
            break

        target = candidate.pose
        print(f"  WP {planner.n_waypoints:3d}  ({target.x:6.2f},{target.y:6.2f})"
              f"  gain={candidate.gain_m2:.2f}m²  score={candidate.score:.3f}"
              f"  cov={coverage.fraction() * 100:.1f}%")

        result = await nav.goto(target)
        trajectory.append(pose_source.current())

        if result == NavigationState.BLOCKED:
            stall = pose_source.current()
            print(f"  BLOCKED at ({stall.x:.2f},{stall.y:.2f})")
            planner.register_block(stall, candidate)

        img = _render(grid, coverage, trajectory, target, area_mask)
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:04d}.png"), img)
        frame_idx += 1

    final_cov = coverage.fraction()
    print(f"\nDone — coverage={final_cov * 100:.1f}%  waypoints={planner.n_waypoints}"
          f"  frames={frame_idx}")
    cv2.imwrite(str(out_dir / "final.png"), _render(grid, coverage, trajectory, None, area_mask))
    print(f"Saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline litter-search simulation")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--circle", type=float, metavar="R", help="Search circle radius (m)")
    grp.add_argument("--rect", type=float, nargs=2, metavar=("W", "D"), help="Search rectangle W×D (m)")
    asyncio.run(_run(parser.parse_args()))
