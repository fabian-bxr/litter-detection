"""Visualize the coverable area for a search-area spec on a static map.

Answers "why does coverage stop early?" by drawing the exact layers the
exploration planner uses to build its coverage *denominator*
(target ∩ free ∩ reachable, see hunter/coverage.py) and printing the area
breakdown in m². Unknown map cells and robot-radius inflation are usually
what shrink a large requested circle down to a tiny reachable region.

Usage:
    uv run python scripts/visualize_coverage.py --circle 20
    uv run python scripts/visualize_coverage.py --rect 6 8 --start 0 0 0
    uv run python scripts/visualize_coverage.py --circle 20 --map my_lab_grid.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np

from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.reachability import reachable_mask
from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import GridMap
from litter_agents.mapping.provider import FileMapProvider
from litter_agents.mapping.raster import rasterize_area
from litter_agents.sim.sim_main import default_start


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="my_lab_grid.yaml", help="map_server YAML path")
    shape = p.add_mutually_exclusive_group()
    shape.add_argument("--circle", type=float, metavar="R", help="circle radius (m)")
    shape.add_argument(
        "--rect", type=float, nargs=2, metavar=("W", "D"), help="rectangle (m)"
    )
    p.add_argument(
        "--start", type=float, nargs=3, metavar=("X", "Y", "THETA"), default=None,
        help="robot start pose (default: most open free cell)",
    )
    p.add_argument("--robot-radius", type=float, default=HunterParams.robot_radius_m)
    p.add_argument("--out", default="runs/coverage_area.png", help="output PNG path")
    p.add_argument("--scale", type=int, default=3, help="pixel upscale factor")
    return p.parse_args(argv)


def render(
    grid: GridMap,
    target: np.ndarray,
    free: np.ndarray,
    inflated_blocked: np.ndarray,
    reachable: np.ndarray,
    denom: np.ndarray,
    start: Pose2D,
    scale: int,
) -> np.ndarray:
    """BGR overlay; +y is drawn up. Layers are painted darkest-first."""
    occupied = grid.occupied_mask()
    unknown = grid.unknown_mask()

    img = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
    img[free] = (255, 255, 255)        # free            → white
    img[unknown] = (110, 110, 110)     # unknown         → gray
    img[occupied] = (0, 0, 0)          # occupied        → black

    # Configuration-space loss from inflation: cells that are raw-free but
    # blocked once the robot radius is dilated in. Light red.
    inflation_only = inflated_blocked & free
    img[inflation_only] = (140, 140, 230)

    # Free + inside the target but NOT reachable from start (fragmented islands
    # cut off by unknown gaps / inflated walls). Orange.
    free_target = target & free
    unreachable = free_target & ~reachable
    img[unreachable] = (40, 140, 230)

    # The coverage denominator: what the planner actually tries to cover. Green.
    img[denom] = (60, 200, 60)

    # Requested target outline, in blue, regardless of free/unknown.
    contours, _ = cv2.findContours(
        target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, (255, 120, 0), 1)

    # Start pose marker + heading.
    row, col = grid.world_to_grid(start.x, start.y)
    cv2.circle(img, (col, row), 3, (255, 0, 255), -1)
    tip_row, tip_col = grid.world_to_grid(
        start.x + 0.6 * np.cos(start.theta), start.y + 0.6 * np.sin(start.theta)
    )
    cv2.line(img, (col, row), (tip_col, tip_row), (255, 0, 255), 1)

    img = np.flipud(img)  # display with +y up
    return cv2.resize(
        img,
        (img.shape[1] * scale, img.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )


async def run(args: argparse.Namespace) -> None:
    params = HunterParams(robot_radius_m=args.robot_radius)
    grid = await FileMapProvider(args.map).load()
    res2 = grid.resolution**2

    start = (
        Pose2D(x=args.start[0], y=args.start[1], theta=args.start[2])
        if args.start
        else default_start(grid, params.robot_radius_m)
    )

    if args.circle:
        spec = SearchAreaSpec(shape="circle", radius_m=args.circle)
        requested_m2 = np.pi * args.circle**2
    elif args.rect:
        spec = SearchAreaSpec(
            shape="rectangle", width_m=args.rect[0], depth_m=args.rect[1]
        )
        requested_m2 = args.rect[0] * args.rect[1]
    else:
        spec = SearchAreaSpec(shape="circle", radius_m=5.0)
        requested_m2 = np.pi * 25.0

    target = rasterize_area(spec, start, grid)
    free = grid.free_mask()
    inflated_blocked = grid.inflated_blocked(params.robot_radius_m)
    reachable = reachable_mask(~inflated_blocked, grid.world_to_grid(start.x, start.y))
    denom = target & free & reachable

    def m2(mask: np.ndarray) -> float:
        return float(mask.sum()) * res2

    print(
        "\n=== coverable area breakdown (m2) ===\n"
        f"start pose             : ({start.x:.2f}, {start.y:.2f}, th {start.theta:.2f})\n"
        f"requested shape        : {requested_m2:8.1f}\n"
        f"drawn on map (in-bnds) : {m2(target):8.1f}\n"
        f"  & free               : {m2(target & free):8.1f}  "
        f"(lost {m2(target & ~free):.1f} to unknown/occupied)\n"
        f"  & free & reachable   : {m2(denom):8.1f}  "
        f"(lost {m2(target & free & ~reachable):.1f} to fragmentation)\n"
        f"  -> this is the coverage denominator the planner targets."
    )

    img = render(
        grid, target, free, inflated_blocked, reachable, denom, start, args.scale
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(
        f"\nsaved {out}\n"
        "legend: white=free  gray=unknown  black=occupied  blue=requested area\n"
        "        green=coverable (denominator)  orange=free-but-unreachable\n"
        "        light-red=lost to robot-radius inflation  magenta=start+heading"
    )


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
