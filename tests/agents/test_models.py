from __future__ import annotations

from litter_detector.agents.models import (
    Candidate,
    MissionStatus,
    NBVParams,
    Pose,
    SearchArea,
)


def test_search_area_round_trip() -> None:
    pose = Pose(x=1.0, y=2.0, theta=0.5)
    area = SearchArea(
        polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
        anchor_pose=pose,
        label="test",
    )
    dump = area.model_dump_json()
    restored = SearchArea.model_validate_json(dump)
    assert restored.anchor_pose.x == 1.0
    assert restored.polygon[2] == (5.0, 5.0)


def test_candidate_score_field() -> None:
    c = Candidate(
        pose=Pose(x=0.0, y=0.0, theta=0.0),
        gain=0.4,
        cost_m=1.0,
        score=0.0,
    )
    assert c.gain == 0.4 and c.cost_m == 1.0


def test_nbv_defaults() -> None:
    p = NBVParams()
    assert p.coverage_target == 0.85
    assert p.h_fov_deg == 70.0


def test_mission_status_default_state() -> None:
    s = MissionStatus(mission_id="abc", state="idle")
    assert s.iteration == 0
    assert s.coverage == 0.0
