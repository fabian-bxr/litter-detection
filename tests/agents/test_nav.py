from __future__ import annotations

from litter_detector.agents.models import Pose
from litter_detector.agents.tools.nav import FakeNavClient


def test_fake_nav_arrives_by_default() -> None:
    nav = FakeNavClient()
    rid = nav.submit(Pose(x=1.0, y=0.0, theta=0.0))
    result = nav.wait_for_terminal(rid, timeout=0.0)
    assert result is not None
    assert result.state == "arrived_final"
    assert result.final_pose is not None
    assert result.final_pose.x == 1.0


def test_fake_nav_blocked_state() -> None:
    nav = FakeNavClient()
    nav.next_state = "blocked"
    rid = nav.submit(Pose(x=2.0, y=0.0, theta=0.0))
    result = nav.wait_for_terminal(rid, timeout=0.0)
    assert result is not None
    assert result.state == "blocked"
    assert result.final_pose is None
    # Subsequent submits go back to default
    rid2 = nav.submit(Pose(x=3.0, y=0.0, theta=0.0))
    assert nav.wait_for_terminal(rid2, timeout=0.0).state == "arrived_final"


def test_fake_nav_records_targets() -> None:
    nav = FakeNavClient()
    nav.submit(Pose(x=1.0, y=2.0, theta=0.5))
    nav.submit(Pose(x=3.0, y=4.0, theta=0.6))
    assert len(nav.submitted) == 2
    assert nav.submitted[1][1].x == 3.0
