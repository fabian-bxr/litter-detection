from __future__ import annotations

from pydantic_ai import Agent

from litter_agents.agents.models import build_model, output_spec
from litter_agents.config import AgentSettings
from litter_agents.interfaces.mission import SearchAreaSpec

SEARCH_AREA_INSTRUCTIONS = """\
You convert a user's litter-search request into a structured area
specification, relative to the robot's current position.

Coordinate frame: the robot stands at the origin facing +x; +y is to its
left. All values are meters.

Guidelines:
- "around me" / "um mich herum" → a circle centered on the robot
  (center_dx_m = center_dy_m = 0).
- "in front of me" → offset the center forward (positive center_dx_m), e.g.
  "the 5x5 m area in front of me" → rectangle with center_dx_m = 2.5.
- "behind me" → negative center_dx_m; "to my left" → positive center_dy_m.
- "this room", "the corridor" and similar named places cannot be resolved —
  fall back to a circle of radius 5 and say so in the rationale.
- If no size is given at all, default to a circle of radius 5 and say so in
  the rationale.
- rotate_with_robot: true when the user describes the area relative to their
  facing direction (the usual case); false only for explicitly absolute or
  compass-aligned requests.
- Use the rationale to state the assumptions you made, in one short sentence.

The request may be in any language (often English or German).
"""


def build_search_area_agent(settings: AgentSettings) -> Agent[None, SearchAreaSpec]:
    return Agent[None, SearchAreaSpec](
        build_model(settings, settings.text_model_name),
        output_type=output_spec(settings, SearchAreaSpec),
        instructions=SEARCH_AREA_INSTRUCTIONS,
        retries=settings.llm_retries,
    )


async def parse_search_area(
    agent: Agent[None, SearchAreaSpec], prompt: str
) -> SearchAreaSpec:
    result = await agent.run(prompt)
    return result.output
