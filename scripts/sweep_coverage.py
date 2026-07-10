"""Headless parameter sweep for exploration coverage.

Runs the real hunter loop (planner + FakeNav, identical to litter-sim) without
rendering, for a one-at-a-time sweep around a baseline, and prints how each
parameter changes coverage. Reports both:
  - coverage %  : fraction of the *reachable* denominator that gets seen
  - covered m2  : absolute area actually seen  (= coverage% x reachable m2)
  - reachable m2: the denominator itself (grows as robot_radius shrinks)

"Coverage of the map" is best judged by *covered m2* — coverage % can look
high simply because the reachable denominator collapsed to a tiny pocket.

Usage:
    uv run python scripts/sweep_coverage.py --circle 20 --start -0.31 -0.37 -1.09
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace

from loguru import logger

from litter_agents.hunter.explore import explore
from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.planner import ExplorationPlanner
from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap
from litter_agents.mapping.provider import FileMapProvider
from litter_agents.mapping.raster import rasterize_area
from litter_agents.sim.fake_nav import FakeNav, FakePoseSource
from litter_agents.sim.sim_main import default_start


async def run_one(
    grid: GridMap,
    spec: SearchAreaSpec,
    start: Pose2D,
    params: HunterParams,
    *,
    no_gain_limit: int,
    max_waypoints: int,
) -> dict:
    target = rasterize_area(spec, start, grid)
    planner = ExplorationPlanner(grid, target, params, start)
    pose_source = FakePoseSource(start)

    def on_tick(pose: Pose2D) -> None:
        planner.coverage.update(pose)

    nav = FakeNav(
        pose_source,
        on_tick=on_tick,
        grid=grid,
        blocked_inflated=planner.blocked_inflated(),
        skip_start_m=params.robot_radius_m,
    )
    planner.coverage.update(start)

    stats = await explore(
        planner,
        nav,
        pose_source,
        max_speed=0.6,
        max_waypoints=max_waypoints,
        max_duration_s=600.0,
        no_gain_limit=no_gain_limit,
        blocked_wait_s=0.0,
        replan_idle_s=0.0,
    )
    frac = planner.coverage.fraction()
    reachable_m2 = planner.coverage.denominator_m2()
    return {
        "coverage": frac,
        "reachable_m2": reachable_m2,
        "covered_m2": frac * reachable_m2,
        "waypoints": stats.n_waypoints,
        "blocked": stats.n_blocked,
        "distance_m": pose_source.distance_traveled,
        "stop": stats.stop_reason,
    }


logger.remove()  # quiet the per-waypoint INFO spam across many runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="my_lab_grid.yaml")
    p.add_argument("--circle", type=float, default=20.0, help="circle radius (m)")
    p.add_argument(
        "--start", type=float, nargs=3, metavar=("X", "Y", "THETA"), default=None,
        help="robot start pose (default: most open free cell)",
    )
    p.add_argument("--max-waypoints", type=int, default=300)
    return p.parse_args(argv)


async def run(args: argparse.Namespace) -> None:
    grid = await FileMapProvider(args.map).load()
    start = (
        Pose2D(x=args.start[0], y=args.start[1], theta=args.start[2])
        if args.start
        else default_start(grid, HunterParams.robot_radius_m)
    )
    spec = SearchAreaSpec(shape="circle", radius_m=args.circle)
    # Three planners compared head-to-head: greedy-only (legacy), greedy +
    # frontier fallback, and the cluster-commit NBV planner.
    greedy = HunterParams(planner_mode="greedy", enable_frontier_fallback=False)
    frontier = replace(greedy, enable_frontier_fallback=True)
    nbv = replace(greedy, planner_mode="nbv")

    runs: list[tuple[str, HunterParams, int]] = []
    for r in (0.30, 0.20):
        runs.append((f"r={r} greedy", replace(greedy, robot_radius_m=r), 3))
        runs.append((f"r={r} greedy+frontier", replace(frontier, robot_radius_m=r), 3))
        runs.append((f"r={r} NBV", replace(nbv, robot_radius_m=r), 3))

    print(f"\nstart=({start.x:.2f},{start.y:.2f},th {start.theta:.2f})  "
          f"circle r={args.circle} m\n")
    header = (
        f"{'config':<32} {'cov%':>6} {'covered':>9} {'reach':>8} "
        f"{'wp/blk':>6} {'dist':>7}  stop"
    )
    print(header)
    print("-" * len(header))
    for label, params, ngl in runs:
        res = await run_one(
            grid, spec, start, params,
            no_gain_limit=ngl, max_waypoints=args.max_waypoints,
        )
        print(
            f"{label:<32} {res['coverage']*100:5.1f}% {res['covered_m2']:8.1f} "
            f"{res['reachable_m2']:7.1f} {res['waypoints']:3d}/{res['blocked']:<2d} "
            f"{res['distance_m']:6.1f}m  {res['stop']}"
        )


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()