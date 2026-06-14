"""CLI entry point for the litter-search mission.

Usage:
    # Natural language (calls SearchArea LLM agent):
    uv run litter-mission "Search 10 m around me for litter"

    # Explicit geometry (no LLM needed):
    uv run litter-mission --area-circle 10
    uv run litter-mission --area-rect 6 4 --confirm

    # Save report to disk:
    uv run litter-mission --area-circle 8 --save-report runs/reports
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from ..config import AgentSettings
from ..mapping.raster import AreaSpec
from .orchestrator import MissionController
from .reporter import print_report, save_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="litter-mission",
        description=(
            "Autonomous litter-search mission for the Unitree Go2 robot.\n\n"
            "Provide either a natural-language PROMPT or an explicit --area-* flag.\n"
            "When only a PROMPT is given, the SearchArea LLM agent interprets it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        metavar="PROMPT",
        help=(
            'Natural language search description, e.g. '
            '"Search 10 m around me for litter". '
            "Parsed by the SearchArea LLM agent when no --area-* flag is given."
        ),
    )

    area_group = parser.add_mutually_exclusive_group()
    area_group.add_argument(
        "--area-circle",
        type=float,
        metavar="RADIUS_M",
        help="Circular search area of RADIUS_M metres centred on the robot.",
    )
    area_group.add_argument(
        "--area-rect",
        type=float,
        nargs=2,
        metavar=("WIDTH_M", "DEPTH_M"),
        help="Rectangular area WIDTH_M × DEPTH_M metres extending forward from the robot.",
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Pause and wait for Enter before the robot starts moving.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Open a live matplotlib map window showing coverage during the mission.",
    )
    parser.add_argument(
        "--save-report",
        metavar="DIR",
        help="Save mission report JSON to DIR/mission_id.json.",
    )

    # Hidden overrides for scripting / testing
    parser.add_argument("--map", metavar="YAML", help=argparse.SUPPRESS)
    parser.add_argument("--endpoint", metavar="URI", help=argparse.SUPPRESS)

    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    settings = AgentSettings()
    if args.map:
        settings = settings.model_copy(update={"map_file": args.map})
    if args.endpoint:
        settings = settings.model_copy(update={"zenoh_router_endpoint": args.endpoint})

    # Resolve area spec — explicit flags win; fall back to SearchArea LLM agent
    area_spec: AreaSpec | None = None
    if args.area_circle is not None:
        area_spec = AreaSpec(shape="circle", radius_m=args.area_circle)
    elif args.area_rect is not None:
        w, d = args.area_rect
        area_spec = AreaSpec(
            shape="rectangle", width_m=w, depth_m=d, center_dx_m=d / 2
        )
    elif args.prompt:
        from ..agents.search_area import SearchAreaAgent
        agent = SearchAreaAgent(settings)
        print(f"Interpreting prompt: \"{args.prompt}\" …")
        try:
            plan = await agent.parse(args.prompt)
        except Exception as exc:
            print(f"ERROR: SearchArea agent failed: {exc}", file=sys.stderr)
            return 2
        area_spec = plan.to_area_spec()
        print(f"  → {plan.interpretation}")
        print(f"  → area: {plan.shape}  ", end="")
        if plan.shape == "circle":
            print(f"radius={plan.radius_m} m")
        else:
            print(f"{plan.width_m} m wide × {plan.depth_m} m deep")
    else:
        print(
            "ERROR: provide either a PROMPT or --area-circle / --area-rect.",
            file=sys.stderr,
        )
        return 2

    ctrl = MissionController(settings)
    try:
        result = await ctrl.run(
            area_spec=area_spec,
            prompt=args.prompt,
            confirm=args.confirm,
            viz=args.viz,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted — mission aborted.")
        return 130

    print_report(result)

    if args.save_report:
        from pathlib import Path
        from datetime import datetime, timezone
        mission_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        report_path = save_report(result, Path(args.save_report), mission_id)
        print(f"Report saved → {report_path}")

    return 0 if result.coverage_fraction >= settings.coverage_threshold else 1


def main() -> None:
    args = _parse_args()
    sys.exit(asyncio.run(_async_main(args)))
