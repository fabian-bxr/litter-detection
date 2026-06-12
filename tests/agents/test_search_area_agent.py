import asyncio
import json
import math

import numpy as np
import pytest
from pydantic_ai.models.test import TestModel

from litter_agents.agents.search_area import build_search_area_agent, parse_search_area
from litter_agents.config import AgentSettings
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mapping.grid import FREE, GridMap
from litter_agents.mapping.raster import rasterize_area


def scripted_agent(payload: dict):
    agent = build_search_area_agent(AgentSettings(ollama_api_key="test"))
    return agent, TestModel(custom_output_text=json.dumps(payload))


def test_circle_request_end_to_end():
    agent, model = scripted_agent(
        {
            "shape": "circle",
            "radius_m": 10.0,
            "center_dx_m": 0.0,
            "center_dy_m": 0.0,
            "rotate_with_robot": True,
            "rationale": "10 m around the robot",
        }
    )
    with agent.override(model=model):
        spec = asyncio.run(parse_search_area(agent, "Search 10m around me for litter"))
    assert spec.shape == "circle" and spec.radius_m == 10.0

    # The parsed spec rasterizes to the analytic circle area on an open map.
    occ = np.full((500, 500), FREE, dtype=np.int8)
    grid = GridMap(occ=occ, resolution=0.1, origin_x=-25.0, origin_y=-25.0)
    mask = rasterize_area(spec, Pose2D(x=0.0, y=0.0, theta=0.3), grid)
    assert mask.sum() * grid.resolution**2 == pytest.approx(math.pi * 100.0, rel=0.05)


def test_offset_rectangle_request():
    agent, model = scripted_agent(
        {
            "shape": "rectangle",
            "width_m": 5.0,
            "depth_m": 5.0,
            "center_dx_m": 2.5,
            "center_dy_m": 0.0,
            "rotate_with_robot": True,
            "rationale": "5x5 m in front",
        }
    )
    with agent.override(model=model):
        spec = asyncio.run(
            parse_search_area(agent, "Search the 5 by 5 meter area in front of me")
        )
    assert spec.shape == "rectangle"
    assert spec.center_dx_m == 2.5
