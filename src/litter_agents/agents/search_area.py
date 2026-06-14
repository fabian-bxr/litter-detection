"""SearchArea agent — turns a natural-language prompt into an AreaSpec.

Example prompts:
    "Search 10 metres around me for litter"
    "Check the corridor in front of me, 6 m deep and 3 m wide"
    "Scan a 5 m radius circle around the robot"
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..config import AgentSettings
from ..mapping.raster import AreaSpec

_SYSTEM_PROMPT = """\
You are a robot mission planner.  The user describes a search area in natural
language.  Your job is to extract a precise geometric specification.

Rules:
- Choose shape = "circle" when the user says "around", "radius", or gives a
  single distance without direction.
- Choose shape = "rectangle" when the user says "in front", "ahead", "corridor",
  or gives separate width and depth/length values.
- Default to "circle" with radius_m = 5.0 if the description is too vague.
- radius_m must be > 0 for circles; width_m and depth_m must both be > 0 for
  rectangles.
- center_dx_m shifts the area forward (positive = forward of robot).  Use 0.0
  unless the user says "in front" — then set center_dx_m = depth_m / 2 so the
  rectangle starts at the robot and extends forward.
- center_dy_m shifts the area laterally (positive = left).  Use 0.0 unless the
  user explicitly mentions an offset to the side.
- interpretation: one short sentence explaining what you understood.
"""


class AreaPlan(BaseModel):
    """Structured output from the SearchArea agent."""

    shape: Literal["circle", "rectangle"]
    radius_m: float | None = None
    width_m: float | None = None
    depth_m: float | None = None
    center_dx_m: float = 0.0
    center_dy_m: float = 0.0
    interpretation: str

    @model_validator(mode="after")
    def _check_fields(self) -> "AreaPlan":
        if self.shape == "circle":
            if self.radius_m is None or self.radius_m <= 0:
                raise ValueError("radius_m must be > 0 for circle")
        else:
            if not (self.width_m and self.depth_m and self.width_m > 0 and self.depth_m > 0):
                raise ValueError("width_m and depth_m must be > 0 for rectangle")
        return self

    def to_area_spec(self) -> AreaSpec:
        return AreaSpec(
            shape=self.shape,
            radius_m=self.radius_m,
            width_m=self.width_m,
            depth_m=self.depth_m,
            center_dx_m=self.center_dx_m,
            center_dy_m=self.center_dy_m,
        )


class SearchAreaAgent:
    def __init__(self, cfg: AgentSettings | None = None) -> None:
        cfg = cfg or AgentSettings()
        provider = OpenAIProvider(
            base_url=f"{cfg.ollama_base_url}/v1",
            api_key=cfg.ollama_api_key,
        )
        model = OpenAIModel(cfg.ollama_text_model, provider=provider)
        self._agent: Agent[None, AreaPlan] = Agent(
            model,
            output_type=AreaPlan,
            system_prompt=_SYSTEM_PROMPT,
            retries=3,
        )

    async def parse(self, prompt: str) -> AreaPlan:
        """Parse a natural-language search description into an AreaPlan."""
        result = await self._agent.run(prompt)
        return result.output
