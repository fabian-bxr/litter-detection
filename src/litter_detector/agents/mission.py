from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from litter_detector.agents.config import AgentLLM
from litter_detector.agents.models import (
    MissionStatus,
    NBVParams,
    Pose,
    SearchArea,
)
from litter_detector.agents.nbv.geometry import rect_ahead, rect_around
from litter_detector.agents.planner import PlannerCore, PlannerRunner
from litter_detector.agents.tools.nav import NavClient
from litter_detector.agents.tools.occupancy import OccupancyClient
from litter_detector.agents.tools.pose import PoseClient


class _PoseSrc(Protocol):
    def get(self, timeout: float = 1.0) -> Pose | None: ...


@dataclass
class MissionDeps:
    """Dependencies injected into Mission Agent tool calls."""

    pose: _PoseSrc
    runner: PlannerRunner


SYSTEM_PROMPT = """\
You supervise a quadruped robot that searches bounded areas for litter.

When the user asks you to search:
1. Call `get_current_pose` if you need the robot's position.
2. Choose ONE of:
   - `start_search_around(diameter_m)` — circular-ish search around the robot.
   - `start_search_ahead(depth_m, width_m)` — rectangle starting at the robot,
     extending forward along its current heading.
3. Use `get_status` to report progress when asked.
4. Use `abort_search` only if the user explicitly cancels.

Defaults when ambiguous: width_m=2.0 for `ahead`, diameter_m=10.0 for `around`.
Be terse. State the area you started searching and the immediate status.
If the user asks a non-search question, answer directly without tools.
"""


def build_mission_agent(model: Model) -> Agent[MissionDeps, str]:
    """Construct the Mission Agent bound to the given model.

    The model is injected to keep tests independent of any LLM backend.
    """
    agent: Agent[MissionDeps, str] = Agent(
        model=model,
        deps_type=MissionDeps,
        system_prompt=SYSTEM_PROMPT,
        instrument=True,
    )

    @agent.tool
    def get_current_pose(ctx: RunContext[MissionDeps]) -> Pose:
        """Return the robot's current pose in the odometry frame."""
        pose = ctx.deps.pose.get(timeout=2.0)
        if pose is None:
            raise RuntimeError("no pose available from odometry")
        return pose

    @agent.tool
    def start_search_around(
        ctx: RunContext[MissionDeps], diameter_m: float
    ) -> MissionStatus:
        """Start searching a square area of `diameter_m` centered on the robot."""
        pose = ctx.deps.pose.get(timeout=2.0)
        if pose is None:
            raise RuntimeError("no pose available from odometry")
        polygon = rect_around(pose.x, pose.y, diameter_m, diameter_m)
        area = SearchArea(polygon=polygon, anchor_pose=pose, label=f"around {diameter_m}m")
        logger.info(f"Mission Agent: start_search_around({diameter_m})")
        return ctx.deps.runner.start(area)

    @agent.tool
    def start_search_ahead(
        ctx: RunContext[MissionDeps], depth_m: float, width_m: float = 2.0
    ) -> MissionStatus:
        """Start searching a rectangle ahead of the robot along its current heading."""
        pose = ctx.deps.pose.get(timeout=2.0)
        if pose is None:
            raise RuntimeError("no pose available from odometry")
        polygon = rect_ahead(pose, depth_m, width_m)
        area = SearchArea(
            polygon=polygon,
            anchor_pose=pose,
            label=f"ahead {depth_m}m × {width_m}m",
        )
        logger.info(f"Mission Agent: start_search_ahead({depth_m},{width_m})")
        return ctx.deps.runner.start(area)

    @agent.tool
    def get_status(ctx: RunContext[MissionDeps]) -> MissionStatus:
        """Return the current mission status."""
        return ctx.deps.runner.status()

    @agent.tool
    def abort_search(ctx: RunContext[MissionDeps]) -> MissionStatus:
        """Abort the active search mission."""
        ctx.deps.runner.abort()
        return ctx.deps.runner.status()

    return agent


def ollama_model(cfg: AgentLLM) -> OpenAIChatModel:
    """Build an OpenAI-compatible model that points at an Ollama endpoint."""
    provider = OpenAIProvider(base_url=cfg.base_url, api_key=cfg.api_key)
    return OpenAIChatModel(model_name=cfg.model, provider=provider)


def build_default_runner(
    nbv_params: NBVParams | None = None,
    debug_dir: Path | None = None,
) -> tuple[PlannerRunner, PoseClient, OccupancyClient, NavClient]:
    """Wire up the Zenoh-backed runner. Caller owns the lifecycle of returned clients."""
    pose = PoseClient()
    occ = OccupancyClient()
    nav = NavClient()
    core = PlannerCore(pose, occ, nav, params=nbv_params, debug_dir=debug_dir)
    return PlannerRunner(core), pose, occ, nav
