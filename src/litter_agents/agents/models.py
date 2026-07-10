from __future__ import annotations

from typing import TypeVar

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import OutputSpec, PromptedOutput, ToolOutput
from pydantic_ai.providers.ollama import OllamaProvider

from litter_agents.config import AgentSettings

T = TypeVar("T")


def build_model(settings: AgentSettings, model_name: str) -> OpenAIChatModel:
    """Ollama Cloud (or self-hosted Ollama) via its OpenAI-compatible endpoint."""
    return OpenAIChatModel(
        model_name,
        provider=OllamaProvider(
            base_url=settings.ollama_base_url,
            # Empty string → None so the OLLAMA_API_KEY env fallback applies.
            api_key=settings.ollama_api_key or None,
        ),
    )


def output_spec(settings: AgentSettings, output_type: type[T]) -> OutputSpec[T]:
    """Structured-output mode for open models.

    Ollama Cloud does not enforce json_schema at generation time (NativeOutput
    raises), so the choice is prompted (schema in the prompt, most robust on
    open vision models) or tool (function calling).
    """
    if settings.agent_output_mode == "tool":
        return ToolOutput(output_type)
    return PromptedOutput(output_type)
