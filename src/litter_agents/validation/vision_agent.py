"""Pydantic-AI vision agent for litter validation.

Given a JPEG frame, asks gemma4:31b-cloud via Ollama whether the image
shows real litter on the ground and returns a structured result.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..config import AgentSettings


class LitterValidationResult(BaseModel):
    """Structured output from the vision LLM."""

    is_litter: bool
    confidence: float          # 0.0 – 1.0
    description: str           # one short sentence
    category: str | None       # e.g. "plastic bottle", "paper", "can", None if not litter


_SYSTEM_PROMPT = """\
You are a precise litter-detection assistant mounted on a ground robot.
You will be shown a camera frame.  Your task is to decide whether the frame
contains visible litter or rubbish lying on the ground (e.g. plastic bottles,
cans, bags, wrappers, cardboard, cigarette butts, paper, etc.).

Rules:
- Only consider objects that are clearly discarded rubbish, NOT natural debris
  (leaves, sticks, mud), the robot's own shadow, or floor markings.
- Set is_litter=false if you are uncertain.
- confidence should reflect your certainty (0.0 = wild guess, 1.0 = obvious).
- category should be the most specific type you can identify (e.g. "plastic bag"),
  or null if is_litter is false.
- description must be one concise sentence.
"""

_USER_PROMPT = (
    "Examine the attached camera frame and decide whether it shows litter "
    "on the ground.  Return a JSON object matching the requested schema."
)


@dataclass
class VisionAgent:
    """Thin wrapper around the Pydantic-AI agent."""

    _agent: Agent[None, LitterValidationResult]

    @classmethod
    def from_settings(cls, cfg: AgentSettings | None = None) -> "VisionAgent":
        cfg = cfg or AgentSettings()
        provider = OpenAIProvider(
            base_url=f"{cfg.ollama_base_url}/v1",
            api_key=cfg.ollama_api_key,
        )
        model = OpenAIModel(cfg.ollama_vision_model, provider=provider)
        agent: Agent[None, LitterValidationResult] = Agent(
            model,
            output_type=LitterValidationResult,
            system_prompt=_SYSTEM_PROMPT,
            retries=2,
        )
        return cls(_agent=agent)

    async def validate(self, jpeg_bytes: bytes) -> LitterValidationResult:
        """Run the vision LLM on a single JPEG frame."""
        result = await self._agent.run(
            [
                BinaryContent(data=jpeg_bytes, media_type="image/jpeg"),
                _USER_PROMPT,
            ]
        )
        return result.output
