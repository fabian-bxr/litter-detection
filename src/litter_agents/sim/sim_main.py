"""Offline exploration sim: run the real search loop against the static map.

Usage:
    uv run litter-sim --circle 5
    uv run litter-sim --rect 6 8 --start 0 0 0 --block -2 1 0.4

Renders coverage evolution frames and a final overview into runs/sim/.
This doubles as the tuning harness for the scoring weights.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

from litter_agents.debug_render import TrajectoryRenderer
from litter_agents.hunter.explore import explore
from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.planner import ExplorationPlanner
from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap
from litter_agents.mapping.provider import FileMapProvider
from litter_agents.mapping.raster import rasterize_area
from litter_agents.sim.fake_nav import FakeNav, FakePoseSource


def default_start(grid: GridMap, robot_radius_m: float) -> Pose2D:
    """Most open free cell — robust default when no start pose is given."""
    free = (~grid.inflated_blocked(robot_radius_m)).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    row, col = np.unravel_index(int(np.argmax(dist)), dist.shape)
    x, y = grid.grid_to_world(int(row), int(col))
    return Pose2D(x=x, y=y, theta=0.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="my_lab_grid.yaml", help="map_server YAML path")
    shape = p.add_mutually_exclusive_group()
    shape.add_argument("--circle", type=float, metavar="R", help="circle radius (m)")
    shape.add_argument(
        "--rect", type=float, nargs=2, metavar=("W", "D"), help="rectangle (m)"
    )
    p.add_argument(
        "--start", type=float, nargs=3, metavar=("X", "Y", "THETA"), default=None
    )
    p.add_argument(
        "--block",
        type=float,
        nargs=3,
        metavar=("X", "Y", "R"),
        action="append",
        default=[],
        help="invisible obstacle disc the map doesn't know about (repeatable)",
    )
    p.add_argument("--out", default=None, help="output dir (default runs/sim/<ts>)")
    p.add_argument("--render-every", type=int, default=20, help="ticks per frame")
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--max-waypoints", type=int, default=200)
    p.add_argument("--robot-radius", type=float, default=HunterParams.robot_radius_m)
    p.add_argument(
        "--frontier", action=argparse.BooleanOptionalAction, default=True,
        help="frontier-seeking fallback around corners (default: on)",
    )
    return p.parse_args(argv)


async def run_sim(args: argparse.Namespace) -> dict:
    params = HunterParams(
        coverage_target_fraction=args.coverage,
        robot_radius_m=args.robot_radius,
        enable_frontier_fallback=args.frontier,
    )
    grid = await FileMapProvider(args.map).load()

    start = (
        Pose2D(x=args.start[0], y=args.start[1], theta=args.start[2])
        if args.start
        else default_start(grid, params.robot_radius_m)
    )
    if args.circle:
        spec = SearchAreaSpec(shape="circle", radius_m=args.circle)
    elif args.rect:
        spec = SearchAreaSpec(shape="rectangle", width_m=args.rect[0], depth_m=args.rect[1])
    else:
        spec = SearchAreaSpec(shape="circle", radius_m=5.0)
    target = rasterize_area(spec, start, grid)

    planner = ExplorationPlanner(grid, target, params, start)
    pose_source = FakePoseSource(start)
    out_dir = Path(args.out) if args.out else Path("runs/sim") / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    renderer = TrajectoryRenderer(grid, target, out_dir)

    tick_count = 0

    def on_tick(pose: Pose2D) -> None:
        nonlocal tick_count
        planner.coverage.update(pose)
        renderer.trajectory.append((pose.x, pose.y))
        tick_count += 1
        if args.render_every and tick_count % args.render_every == 0:
            renderer.save_frame(
                planner.coverage.seen, pose, reachable=planner.coverage.denominator()
            )

    nav = FakeNav(
        pose_source,
        blocked_discs=[tuple(b) for b in args.block],
        on_tick=on_tick,
        grid=grid,
        blocked_inflated=planner.blocked_inflated(),
        skip_start_m=params.robot_radius_m,
    )
    planner.coverage.update(start)
    renderer.trajectory.append((start.x, start.y))

    stats = await explore(
        planner,
        nav,
        pose_source,
        max_speed=0.6,
        max_waypoints=args.max_waypoints,
        max_duration_s=600.0,
        blocked_wait_s=0.0,
        replan_idle_s=0.0,
    )

    final_pose = pose_source.latest or start
    renderer.save_frame(
        planner.coverage.seen, final_pose, reachable=planner.coverage.denominator()
    )
    result = {
        "stop_reason": stats.stop_reason,
        "coverage": planner.coverage.fraction(),
        "reachable_target_m2": planner.coverage.denominator_m2(),
        "n_waypoints": stats.n_waypoints,
        "n_blocked": stats.n_blocked,
        "distance_m": pose_source.distance_traveled,
        "out_dir": str(out_dir),
    }
    print(
        f"\n=== sim result ===\n"
        f"stop reason     : {result['stop_reason']}\n"
        f"coverage        : {result['coverage']:.1%} of "
        f"{result['reachable_target_m2']:.1f} m² reachable target\n"
        f"waypoints       : {result['n_waypoints']} ({result['n_blocked']} blocked)\n"
        f"distance walked : {result['distance_m']:.1f} m\n"
        f"frames          : {result['out_dir']}"
    )
    return result


def main() -> None:
    asyncio.run(run_sim(parse_args()))


if __name__ == "__main__":
    main()
