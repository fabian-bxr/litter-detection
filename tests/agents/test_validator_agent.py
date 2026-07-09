import asyncio
import json
import os

import cv2
import numpy as np
import pytest
from pydantic_ai.models.test import TestModel

from litter_agents.agents.validator import build_validation_agent, make_validate_fn
from litter_agents.config import AgentSettings
from litter_agents.interfaces.detections import TrackMsg
from litter_agents.interfaces.mission import LitterValidation
from litter_agents.validation.worker import ValidationJob


def make_job() -> ValidationJob:
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(img, (40, 40), 20, (40, 120, 200), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return ValidationJob(
        track=TrackMsg(
            id=1,
            bbox=(10, 10, 60, 60),
            area_px=1200,
            first_seen_ns=0,
            last_seen_ns=0,
            n_observations=12,
        ),
        crop_jpeg=buf.tobytes(),
        context_jpeg=buf.tobytes(),
        robot_pose=None,
        bearing_rad=0.0,
    )


def test_agent_parses_structured_output(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    settings = AgentSettings()
    agent = build_validation_agent(settings)
    validate = make_validate_fn(agent, settings)
    scripted = json.dumps(
        {
            "is_litter": True,
            "category": "plastic",
            "confidence": 0.85,
            "description": "a crushed plastic bottle",
        }
    )
    with agent.override(model=TestModel(custom_output_text=scripted)):
        result = asyncio.run(validate(make_job()))
    assert isinstance(result, LitterValidation)
    assert result.is_litter and result.category == "plastic"


def test_validation_model_requires_category_for_litter():
    with pytest.raises(ValueError):
        LitterValidation(is_litter=True, confidence=0.9, description="x")
    # Non-litter without category is fine.
    LitterValidation(is_litter=False, confidence=0.5, description="floor stain")


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("OLLAMA_API_KEY"),
    reason="live Ollama Cloud canary; set OLLAMA_API_KEY to run",
)
def test_live_ollama_cloud_vision():
    """Canary for the real endpoint: model name, auth, vision, output parsing."""
    settings = AgentSettings()
    agent = build_validation_agent(settings)
    validate = make_validate_fn(agent, settings)
    result = asyncio.run(validate(make_job()))
    assert isinstance(result, LitterValidation)
    assert 0.0 <= result.confidence <= 1.0
