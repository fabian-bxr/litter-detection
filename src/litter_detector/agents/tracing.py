from __future__ import annotations

import os

import mlflow
import mlflow.pydantic_ai
from loguru import logger


def setup_mlflow_tracing(experiment: str = "litter-mission-agent") -> None:
    """Enable MLflow autologging for Pydantic-AI agents.

    Tracking URI honors `MLFLOW_TRACKING_URI`, falling back to the local
    sqlite store used elsewhere in the project.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    mlflow.pydantic_ai.autolog()
    logger.info(f"MLflow tracing enabled — uri={uri}, experiment={experiment}")
