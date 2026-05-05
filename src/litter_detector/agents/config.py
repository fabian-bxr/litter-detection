from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentLLM(BaseModel):
    """LLM endpoint config for one agent.

    Backed by Ollama by default; swap base_url/model per agent.
    """

    base_url: str = "http://localhost:11434/v1"
    model: str = "ministral-3:3b:cloud"
    api_key: str = "ollama"  # Ollama ignores this but pydantic-ai expects something
    temperature: float = 0.2


class AgentsConfig(BaseSettings):
    """Per-agent LLM configuration. Env-overridable via nested delimiter `__`.

    Example:
        AGENTS__MISSION__MODEL=qwen2.5:14b-instruct
        AGENTS__PLANNER__BASE_URL=http://jetson:11434/v1
    """

    model_config = SettingsConfigDict(env_prefix="AGENTS__", env_nested_delimiter="__")

    mission: AgentLLM = Field(default_factory=AgentLLM)
    planner: AgentLLM = Field(default_factory=AgentLLM)
