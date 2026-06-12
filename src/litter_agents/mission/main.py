"""Litter-search mission CLI.

Usage:
    uv run litter-mission "Search 10m around me for litter"
    uv run litter-mission "Check the area in front of me" --confirm
    uv run litter-mission --circle 5 "manual area"   # bypass the area agent

Requires running alongside: the zenoh router, the robodog stack (localization
+ nav), and `uv run camera` + `uv run detector` for litter validation.
"""

from __future__ import annotations

import argparse
import asyncio
import math

from loguru import logger

from litter_detector.telemetry import setup_telemetry

from litter_agents.interfaces.mission import MissionReport, SearchAreaSpec
from litter_agents.mission.orchestrator import MissionController


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt", help="natural-language search request")
    p.add_argument("--map", default=None, help="map_server YAML (default: settings)")
    shape = p.add_mutually_exclusive_group()
    shape.add_argument(
        "--circle", type=float, metavar="R",
        help="skip the area agent: circle of radius R (m) around the robot",
    )
    shape.add_argument(
        "--rect", type=float, nargs=2, metavar=("W", "D"),
        help="skip the area agent: W×D m rectangle centered on the robot",
    )
    p.add_argument("--confirm", action="store_true",
                   help="show the parsed area and wait before moving")
    p.add_argument("--no-llm-summary", action="store_true")
    p.add_argument("--no-validation", action="store_true",
                   help="search without the detection/validation pipeline")
    return p.parse_args(argv)


def print_report(report: MissionReport) -> None:
    lines = [
        "",
        f"=== mission {report.mission_id} ===",
        f"prompt          : {report.prompt}",
        f"coverage        : {report.coverage_fraction:.1%} of "
        f"{report.reachable_target_m2:.1f} m² reachable target area",
        f"waypoints       : {report.n_waypoints} ({report.n_blocked} blocked)",
        f"distance / time : {report.distance_traveled_m:.1f} m in "
        f"{report.duration_s:.0f} s",
        f"findings        : {len(report.findings)} validated, "
        f"{report.n_rejected} rejected, {report.n_errors} errors",
    ]
    for f in report.findings:
        dup = f" (dup of #{f.possible_duplicate_of})" if f.possible_duplicate_of else ""
        lines.append(
            f"  #{f.track_id:<4} {f.category:<10} {f.confidence:.0%}  "
            f"robot at ({f.robot_pose.x:.2f}, {f.robot_pose.y:.2f}) "
            f"bearing {math.degrees(f.bearing_rad):+.0f}°{dup}  {f.image_path}"
        )
    if report.summary_text:
        lines += ["", report.summary_text]
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    setup_telemetry("litter-mission")

    area_spec: SearchAreaSpec | None = None
    if args.circle:
        area_spec = SearchAreaSpec(shape="circle", radius_m=args.circle)
    elif args.rect:
        area_spec = SearchAreaSpec(
            shape="rectangle", width_m=args.rect[0], depth_m=args.rect[1]
        )

    controller = MissionController()
    if args.map:
        from litter_agents.mapping.provider import FileMapProvider

        controller = MissionController(map_provider=FileMapProvider(args.map))

    try:
        report = asyncio.run(
            controller.run(
                args.prompt,
                area_spec=area_spec,
                confirm=args.confirm,
                llm_summary=not args.no_llm_summary,
                enable_validation=not args.no_validation,
            )
        )
    except KeyboardInterrupt:
        logger.warning("Mission aborted by user")
        return
    print_report(report)


if __name__ == "__main__":
    main()
