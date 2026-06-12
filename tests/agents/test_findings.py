from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.validation.findings import FindingRow, FindingsRepository


def make_row(mission="m1", track_id=1, status="validated", **overrides) -> FindingRow:
    defaults = dict(
        mission_id=mission,
        track_id=track_id,
        status=status,
        category="plastic",
        confidence=0.9,
        description="a crushed bottle",
        robot_pose=Pose2D(x=1.0, y=2.0, theta=0.5),
        bearing_rad=0.1,
        bbox=(10, 20, 30, 40),
        area_px=900,
        n_observations=12,
        first_seen_ns=100,
        last_seen_ns=200,
        validated_at_ns=300,
        image_path="crop.jpg",
        context_image_path="ctx.jpg",
        model_name="gemma3:27b",
        raw_response="{}",
    )
    defaults.update(overrides)
    return FindingRow(**defaults)


def test_roundtrip(tmp_path):
    repo = FindingsRepository(tmp_path / "f.db")
    assert repo.insert_finding(make_row())
    rows = repo.findings("m1")
    assert len(rows) == 1
    row = rows[0]
    assert row.category == "plastic"
    assert row.robot_pose is not None and row.robot_pose.y == 2.0
    assert row.bbox == (10, 20, 30, 40)


def test_unique_constraint_dedups(tmp_path):
    repo = FindingsRepository(tmp_path / "f.db")
    assert repo.insert_finding(make_row())
    assert not repo.insert_finding(make_row(category="metal"))
    assert len(repo.findings("m1")) == 1
    # Same track in another mission is a separate finding.
    assert repo.insert_finding(make_row(mission="m2"))


def test_status_counts_and_filter(tmp_path):
    repo = FindingsRepository(tmp_path / "f.db")
    repo.insert_finding(make_row(track_id=1, status="validated"))
    repo.insert_finding(make_row(track_id=2, status="rejected", category=None))
    repo.insert_finding(make_row(track_id=3, status="error", category=None))
    assert repo.status_counts("m1") == {"validated": 1, "rejected": 1, "error": 1}
    assert [r.track_id for r in repo.findings("m1", status="validated")] == [1]
    assert repo.processed_track_ids("m1") == {1, 2, 3}


def test_missing_pose_roundtrips_as_none(tmp_path):
    repo = FindingsRepository(tmp_path / "f.db")
    repo.insert_finding(make_row(robot_pose=None))
    assert repo.findings("m1")[0].robot_pose is None


def test_mission_lifecycle(tmp_path):
    repo = FindingsRepository(tmp_path / "f.db")
    spec = SearchAreaSpec(shape="circle", radius_m=5.0)
    repo.start_mission("m1", "search 5m around me", spec, started_ns=1)
    repo.finish_mission(
        "m1",
        finished_ns=2,
        coverage_fraction=0.97,
        distance_m=12.3,
        n_waypoints=9,
        n_blocked=1,
        report_json="{}",
    )
    row = repo._conn.execute(
        "SELECT prompt, coverage_fraction, n_waypoints FROM missions"
    ).fetchone()
    assert row == ("search 5m around me", 0.97, 9)
