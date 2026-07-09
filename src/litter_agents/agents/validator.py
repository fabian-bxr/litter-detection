from __future__ import annotations

from pydantic_ai import Agent, BinaryContent

from litter_agents.agents.models import build_model, output_spec
from litter_agents.config import AgentSettings
from litter_agents.interfaces.mission import LitterValidation
from litter_agents.validation.worker import ValidateFn, ValidationJob

VALIDATOR_INSTRUCTIONS = """\
You verify detections from a litter-segmentation model running on a quadruped
robot's camera. You receive a close-up crop of the detected region and, when
available, the full scene with the region boxed in red.

Decide whether the crop shows actual litter / trash — a discarded man-made
object lying where it doesn't belong (bottle, wrapper, can, cigarette butt,
paper cup...). It is NOT litter if it is: a shadow, stain or floor texture, a
leaf or other natural debris, furniture or equipment that belongs in the
scene, a permanent fixture, or something a person is actively using.

If it is litter, classify it into exactly one category:
plastic, paper, cardboard, metal, glass, organic, cigarette, textile, other.

Rate your confidence in the overall verdict from 0.0 to 1.0 and describe in
one sentence what you see. When the image is too blurry or small to tell,
answer is_litter=false with low confidence rather than guessing.
"""


def build_validation_agent(settings: AgentSettings) -> Agent[None, LitterValidation]:
    return Agent[None, LitterValidation](
        build_model(settings, settings.vision_model_name),
        output_type=output_spec(settings, LitterValidation),
        instructions=VALIDATOR_INSTRUCTIONS,
        retries=settings.llm_retries,
    )


def make_validate_fn(
    agent: Agent[None, LitterValidation], settings: AgentSettings
) -> ValidateFn:
    """Adapt the agent to the worker's ValidateFn interface."""

    async def validate(job: ValidationJob) -> LitterValidation:
        parts: list = [
            "Image 1 is the close-up crop of the detection.",
            BinaryContent(data=job.crop_jpeg, media_type="image/jpeg"),
        ]
        if job.context_jpeg is not None:
            parts += [
                "Image 2 is the full scene; the detection is boxed in red.",
                BinaryContent(data=job.context_jpeg, media_type="image/jpeg"),
            ]
        result = await agent.run(parts)
        return result.output

    return validate
