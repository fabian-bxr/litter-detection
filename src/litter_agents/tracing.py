"""MLflow tracing for the pydantic-ai agents.

Autologs every ``Agent.run()`` — prompt, raw response, token usage, retries and
the parsed structured output — as an MLflow trace. Traces land in the same
``mlflow.db`` the training runs use, under their own experiment, so
``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`` shows both.

Call ``setup_mlflow_tracing()`` once per process at startup. The offline sim has
no LLM calls, so it does not.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from litter_agents.config import AgentSettings

_REPO_ROOT = Path(__file__).parents[2]  # src/litter_agents -> src -> repo root

_configured = False


def tracking_uri(settings: AgentSettings) -> str:
    """Resolve the tracking URI, defaulting to the repo-root sqlite store.

    Absolute so the store does not depend on the CWD a mission is launched from.
    """
    if settings.mlflow_tracking_uri:
        return settings.mlflow_tracking_uri
    return f"sqlite:///{_REPO_ROOT / 'mlflow.db'}"


def setup_mlflow_tracing(settings: AgentSettings | None = None) -> bool:
    """Enable MLflow autologging for pydantic-ai. Idempotent; returns True if active.

    Never raises: an unreachable tracing backend must not take a mission down
    with it.
    """
    global _configured
    settings = settings or AgentSettings()
    if not settings.mlflow_tracing:
        return False
    if _configured:
        return True

    try:
        import mlflow
        import mlflow.pydantic_ai

        uri = tracking_uri(settings)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        mlflow.pydantic_ai.autolog()
    except Exception as exc:  # noqa: BLE001 — tracing is never mission-critical
        logger.warning(f"MLflow tracing disabled: {exc!r}")
        return False

    _configured = True
    logger.info(
        f"MLflow tracing → {uri} (experiment: {settings.mlflow_experiment})"
    )
    return True
