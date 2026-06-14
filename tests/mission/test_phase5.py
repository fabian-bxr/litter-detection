"""Tests for Phase 5: SearchArea AreaPlan validation + Reporter formatting."""

import json
import tempfile
from pathlib import Path

import pytest

from litter_agents.agents.search_area import AreaPlan, SearchAreaAgent
from litter_agents.mission.orchestrator import MissionResult
from litter_agents.mission.reporter import print_report, save_report


# ---------------------------------------------------------------------------
# AreaPlan validation (no LLM call — just schema validation)
# ---------------------------------------------------------------------------


def test_area_plan_circle_valid():
    plan = AreaPlan(shape="circle", radius_m=5.0, interpretation="5 m radius")
    spec = plan.to_area_spec()
    assert spec.shape == "circle"
    assert spec.radius_m == 5.0


def test_area_plan_rectangle_valid():
    plan = AreaPlan(
        shape="rectangle",
        width_m=4.0,
        depth_m=6.0,
        center_dx_m=3.0,
        interpretation="4 × 6 m forward corridor",
    )
    spec = plan.to_area_spec()
    assert spec.shape == "rectangle"
    assert spec.width_m == 4.0
    assert spec.depth_m == 6.0
    assert spec.center_dx_m == 3.0


def test_area_plan_circle_missing_radius():
    with pytest.raises(Exception):
        AreaPlan(shape="circle", radius_m=None, interpretation="bad")


def test_area_plan_circle_zero_radius():
    with pytest.raises(Exception):
        AreaPlan(shape="circle", radius_m=0.0, interpretation="bad")


def test_area_plan_rectangle_missing_depth():
    with pytest.raises(Exception):
        AreaPlan(shape="rectangle", width_m=4.0, depth_m=None, interpretation="bad")


def test_search_area_agent_imports():
    agent = SearchAreaAgent.__new__(SearchAreaAgent)
    assert agent is not None


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _make_result(confirmed: bool = True, n_findings: int = 2) -> MissionResult:
    findings = []
    for i in range(n_findings):
        findings.append({
            "id": i + 1,
            "mission_id": "test",
            "run_ts": "2026-06-13T12:00:00",
            "track_id": i + 1,
            "confirmed": int(confirmed),
            "confidence": 0.85,
            "description": "Plastic bottle on floor.",
            "category": "plastic bottle",
            "pose_x": float(i),
            "pose_y": 1.0,
            "pose_theta": 0.0,
            "image_path": None,
            "validated_at": "2026-06-13T12:01:00",
        })
    return MissionResult(
        coverage_fraction=0.82,
        waypoints_visited=10,
        distance_m=34.5,
        duration_s=245.0,
        termination_reason="low_gain",
        findings=findings,
    )


def test_reporter_print_no_error(capsys):
    result = _make_result(confirmed=True, n_findings=2)
    print_report(result)
    out = capsys.readouterr().out
    assert "82.0" in out
    assert "plastic bottle" in out
    assert "confirmed" in out.lower()


def test_reporter_print_no_findings(capsys):
    result = _make_result(n_findings=0)
    print_report(result)
    out = capsys.readouterr().out
    assert "No litter" in out


def test_reporter_save_json():
    result = _make_result(confirmed=True, n_findings=1)
    with tempfile.TemporaryDirectory() as d:
        path = save_report(result, Path(d), "20260613T120000")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["mission_id"] == "20260613T120000"
        assert data["coverage_fraction"] == pytest.approx(0.82)
        assert len(data["findings"]) == 1
