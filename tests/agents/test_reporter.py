from litter_agents.agents.reporter import build_findings, build_report
from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.validation.findings import FindingRow


def row(track_id: int, category="plastic", x=0.0, y=0.0) -> FindingRow:
    return FindingRow(
        mission_id="m1",
        track_id=track_id,
        status="validated",
        category=category,
        confidence=0.9,
        description="something",
        robot_pose=Pose2D(x=x, y=y, theta=0.0),
        bearing_rad=0.0,
        bbox=(0, 0, 10, 10),
        area_px=100,
        n_observations=12,
        first_seen_ns=0,
        last_seen_ns=0,
        validated_at_ns=0,
        image_path="a.jpg",
        context_image_path=None,
    )


def test_nearby_same_category_flagged_as_duplicate():
    findings = build_findings(
        [
            row(1, x=0.0),
            row(2, x=0.5),  # same category, 0.5 m away
            row(3, x=0.6, category="metal"),  # nearby but different category
            row(4, x=5.0),  # same category, far away
        ]
    )
    by_id = {f.track_id: f for f in findings}
    assert by_id[1].possible_duplicate_of is None
    assert by_id[2].possible_duplicate_of == 1
    assert by_id[3].possible_duplicate_of is None
    assert by_id[4].possible_duplicate_of is None


def test_build_report_counts():
    report = build_report(
        mission_id="m1",
        prompt="search",
        area=SearchAreaSpec(shape="circle", radius_m=5.0),
        coverage_fraction=0.97,
        reachable_target_m2=20.0,
        duration_s=120.0,
        distance_traveled_m=33.0,
        n_waypoints=9,
        n_blocked=1,
        validated=[row(1), row(2, category="glass", x=4.0)],
        status_counts={"validated": 2, "rejected": 3, "error": 1},
    )
    assert len(report.findings) == 2
    assert report.n_rejected == 3
    assert report.n_errors == 1
    assert report.summary_text == ""
