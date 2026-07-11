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
from litter_agents.tracing import setup_mlflow_tracing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt", help="natural-language search request")
    p.add_argument("--map", default=None,
                   help="map_server YAML file (implies --map-source file; "
                   "default: settings)")
    p.add_argument(
        "--map-source", choices=("file", "mola"), default=None,
        help="where to load the static map from: 'file' (map_server YAML) or "
        "'mola' (robodog MOLA SLAM control API) (default: settings)",
    )
    p.add_argument("--mola-url", default=None, metavar="URL",
                   help="MOLA control API base URL (default: settings, "
                   "http://localhost:8088)")
    p.add_argument("--mola-session", default=None, metavar="NAME",
                   help="MOLA map session to load (default: most recent)")
    p.add_argument("--mola-build-grid", action="store_true",
                   help="build the 2D costmap via the MOLA API if it isn't ready")
    p.add_argument(
        "--robot-radius", type=float, default=None,
        help="override robot_radius_m (config-space inflation); never go below "
        "the real Go2 half-width",
    )
    p.add_argument(
        "--min-gain", type=float, default=None, metavar="M2",
        help="override min_gain_m2: smallest est. gain (m²) worth a move; lower "
        "it to chase leftover slivers instead of stopping",
    )
    p.add_argument(
        "--max-dist", type=float, default=None, metavar="M",
        help="override candidate_max_dist_m: how far (m) a single waypoint may "
        "reach; raise it to reach distant unseen pockets",
    )
    p.add_argument(
        "--frontier", action=argparse.BooleanOptionalAction, default=None,
        help="enable/disable the frontier-seeking fallback that repositions "
        "around corners when greedy scoring stalls (default: settings)",
    )
    p.add_argument(
        "--planner", choices=("greedy", "nbv"), default=None,
        help="exploration planner: 'nbv' (cluster-commit next-best-view, "
        "default) or 'greedy' (legacy ray-scorer + frontier fallback)",
    )
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

    from litter_agents.config import AgentSettings

    settings = AgentSettings()
    overrides = {
        "robot_radius_m": args.robot_radius,
        "min_gain_m2": args.min_gain,
        "candidate_max_dist_m": args.max_dist,
        "enable_frontier_fallback": args.frontier,
        "planner_mode": args.planner,
        "map_source": args.map_source,
        "mola_api_url": args.mola_url,
        "mola_map_session": args.mola_session,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    # A YAML path implies the file provider unless the source was set explicitly.
    if args.map:
        overrides["map_yaml_path"] = args.map
        overrides.setdefault("map_source", "file")
    if args.mola_build_grid:
        overrides["mola_build_grid"] = True
    if overrides:
        settings = settings.model_copy(update=overrides)

    setup_mlflow_tracing(settings)

    # MissionController builds the provider from settings.map_source (see
    # litter_agents.mapping.provider.build_map_provider).
    controller = MissionController(settings)

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
