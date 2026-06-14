"""Mission reporter — formats and saves MissionResult to stdout and disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .orchestrator import MissionResult


def _fmt_duration(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m} min {s:02d} s" if m else f"{s} s"


def print_report(result: MissionResult) -> None:
    """Print a human-readable mission report to stdout."""
    confirmed = [f for f in result.findings if f.get("confirmed")]
    n_confirmed = len(confirmed)
    n_total = len(result.findings)

    w = 56
    bar = "=" * w
    print(f"\n{'LITTER MISSION REPORT':^{w}}")
    print(bar)
    print(f"  Coverage      {result.coverage_fraction * 100:5.1f} %")
    print(f"  Waypoints     {result.waypoints_visited}")
    print(f"  Distance      {result.distance_m:.1f} m")
    print(f"  Duration      {_fmt_duration(result.duration_s)}")
    print(f"  Termination   {result.termination_reason}")
    print(bar)

    if n_total == 0:
        print("  No litter detections during this mission.")
    else:
        print(f"  Litter found  {n_confirmed} confirmed / {n_total} candidates\n")
        for i, row in enumerate(result.findings, 1):
            tag = "✓" if row.get("confirmed") else "✗"
            cat = row.get("category") or "unknown"
            conf = row.get("confidence", 0.0)
            desc = row.get("description", "")
            px = row.get("pose_x", 0.0)
            py = row.get("pose_y", 0.0)
            print(
                f"  {tag} #{i:<3} {cat:<20} conf={conf:.2f}"
                f"  x={px:.1f} y={py:.1f}"
            )
            print(f"       {desc}")
    print(bar)


def save_report(result: MissionResult, out_dir: Path, mission_id: str) -> Path:
    """Save the full result as JSON to out_dir/mission_id.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{mission_id}.json"
    payload = {
        "mission_id": mission_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage_fraction": result.coverage_fraction,
        "waypoints_visited": result.waypoints_visited,
        "distance_m": result.distance_m,
        "duration_s": result.duration_s,
        "termination_reason": result.termination_reason,
        "findings": result.findings,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
