from __future__ import annotations

from loguru import logger
from pydantic_ai import Agent

from litter_agents.agents.models import build_model
from litter_agents.config import AgentSettings
from litter_agents.interfaces.mission import (
    FindingSummary,
    MissionReport,
    SearchAreaSpec,
)
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.validation.findings import FindingRow

REPORTER_INSTRUCTIONS = """\
You summarize the outcome of a robot litter-search mission for the operator.
You get the mission report as JSON. Write 2-4 plain sentences: what area was
searched and how completely, what litter was found (categories, counts,
notable items), and anything that needs attention (blocked areas, errors).
No markdown, no lists, no preamble.
"""

# Findings whose robot poses are within this distance and share a category are
# flagged as possible duplicates (no depth sensing → no true litter position).
_DUPLICATE_RADIUS_M = 1.0


def build_findings(rows: list[FindingRow]) -> list[FindingSummary]:
    summaries: list[FindingSummary] = []
    for row in rows:
        duplicate_of: int | None = None
        for prior in rows:
            if prior.track_id >= row.track_id:
                break
            if (
                prior.category == row.category
                and prior.robot_pose is not None
                and row.robot_pose is not None
                and prior.robot_pose.distance_to(row.robot_pose) < _DUPLICATE_RADIUS_M
            ):
                duplicate_of = prior.track_id
                break
        summaries.append(
            FindingSummary(
                track_id=row.track_id,
                category=row.category or "other",
                confidence=row.confidence or 0.0,
                robot_pose=row.robot_pose or Pose2D(x=0.0, y=0.0, theta=0.0),
                bearing_rad=row.bearing_rad,
                image_path=row.image_path or "",
                description=row.description or "",
                possible_duplicate_of=duplicate_of,
            )
        )
    return summaries


def build_report(
    *,
    mission_id: str,
    prompt: str,
    area: SearchAreaSpec,
    coverage_fraction: float,
    reachable_target_m2: float,
    duration_s: float,
    distance_traveled_m: float,
    n_waypoints: int,
    n_blocked: int,
    validated: list[FindingRow],
    status_counts: dict[str, int],
) -> MissionReport:
    return MissionReport(
        mission_id=mission_id,
        prompt=prompt,
        area=area,
        coverage_fraction=coverage_fraction,
        reachable_target_m2=reachable_target_m2,
        duration_s=duration_s,
        distance_traveled_m=distance_traveled_m,
        n_waypoints=n_waypoints,
        n_blocked=n_blocked,
        findings=build_findings(validated),
        n_rejected=status_counts.get("rejected", 0),
        n_errors=status_counts.get("error", 0),
    )


async def add_llm_summary(report: MissionReport, settings: AgentSettings) -> None:
    """Best-effort one-paragraph summary; the report is complete without it."""
    agent: Agent[None, str] = Agent(
        build_model(settings, settings.text_model_name),
        output_type=str,
        instructions=REPORTER_INSTRUCTIONS,
    )
    try:
        result = await agent.run(report.model_dump_json(exclude={"summary_text"}))
        report.summary_text = result.output
    except Exception:
        logger.opt(exception=True).warning("LLM mission summary failed; skipping")
