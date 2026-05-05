from __future__ import annotations

import numpy as np
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from litter_detector.agents.mission import MissionDeps, build_mission_agent
from litter_detector.agents.models import NBVParams, Pose
from litter_detector.agents.planner import PlannerCore, PlannerRunner
from litter_detector.agents.tools.nav import FakeNavClient
from litter_detector.agents.tools.occupancy import FakeOccupancyClient, OccupancyGrid
from litter_detector.agents.tools.pose import FakePoseClient


def _build_deps() -> MissionDeps:
    grid = OccupancyGrid(
        data=np.zeros((80, 80), dtype=np.int8),
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
    )
    pose = FakePoseClient(Pose(x=4.0, y=4.0, theta=0.0))
    occ = FakeOccupancyClient(grid)
    nav = FakeNavClient()
    runner = PlannerRunner(PlannerCore(pose, occ, nav, params=NBVParams()))
    return MissionDeps(pose=pose, runner=runner)


def _scripted(*responses: ModelResponse) -> FunctionModel:
    """FunctionModel that yields scripted responses in order."""
    state = {"i": 0}

    async def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state["i"]
        state["i"] += 1
        return responses[min(i, len(responses) - 1)]

    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_agent_invokes_start_search_around() -> None:
    deps = _build_deps()
    model = _scripted(
        ModelResponse(parts=[ToolCallPart(tool_name="start_search_around", args={"diameter_m": 4.0})]),
        ModelResponse(parts=[TextPart(content="Started searching a 4m square around you.")]),
    )
    agent = build_mission_agent(model)
    result = await agent.run("Search a 4 meter area around me", deps=deps)
    assert "4m" in result.output or "4 m" in result.output or "Started" in result.output
    # The runner must have been started.
    assert deps.runner.status().state in ("planning", "navigating", "completed", "blocked", "failed")
    deps.runner.abort()
    deps.runner.join(timeout=5.0)


@pytest.mark.asyncio
async def test_agent_invokes_start_search_ahead() -> None:
    deps = _build_deps()
    model = _scripted(
        ModelResponse(parts=[ToolCallPart(tool_name="start_search_ahead", args={"depth_m": 3.0, "width_m": 2.0})]),
        ModelResponse(parts=[TextPart(content="Searching 3m ahead.")]),
    )
    agent = build_mission_agent(model)
    result = await agent.run("Search 3m ahead of me", deps=deps)
    assert "3m" in result.output or "ahead" in result.output.lower()
    deps.runner.abort()
    deps.runner.join(timeout=5.0)


@pytest.mark.asyncio
async def test_agent_get_status_without_starting() -> None:
    deps = _build_deps()
    model = _scripted(
        ModelResponse(parts=[ToolCallPart(tool_name="get_status", args={})]),
        ModelResponse(parts=[TextPart(content="Status is idle.")]),
    )
    agent = build_mission_agent(model)
    result = await agent.run("What's the status?", deps=deps)
    assert "idle" in result.output.lower()


@pytest.mark.asyncio
async def test_agent_abort_tool() -> None:
    deps = _build_deps()
    model = _scripted(
        ModelResponse(parts=[ToolCallPart(tool_name="abort_search", args={})]),
        ModelResponse(parts=[TextPart(content="Aborted.")]),
    )
    agent = build_mission_agent(model)
    await agent.run("stop", deps=deps)
    # Abort flag should be set even though nothing was running.
    assert deps.runner._core._abort.is_set()
