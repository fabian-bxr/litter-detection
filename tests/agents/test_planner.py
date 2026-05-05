from __future__ import annotations

import time

import numpy as np

from litter_detector.agents.models import NBVParams, Pose, SearchArea
from litter_detector.agents.nbv.geometry import rect_around
from litter_detector.agents.planner import PlannerCore, PlannerRunner
from litter_detector.agents.tools.nav import FakeNavClient
from litter_detector.agents.tools.occupancy import FakeOccupancyClient, OccupancyGrid
from litter_detector.agents.tools.pose import FakePoseClient


class _CoupledNav(FakeNavClient):
    """FakeNav that also teleports the FakePoseClient on arrival."""

    def __init__(self, pose: FakePoseClient) -> None:
        super().__init__()
        self._pose = pose

    def submit(self, target: Pose, **kwargs) -> str:
        rid = super().submit(target, **kwargs)
        if self.default_state == "arrived_final" and self.next_state in (None, "arrived_final"):
            self._pose.set(target)
        return rid


def _setup(
    *,
    grid_h: int = 80,
    grid_w: int = 80,
    res: float = 0.1,
    start: Pose | None = None,
) -> tuple[FakePoseClient, FakeOccupancyClient, _CoupledNav]:
    grid = OccupancyGrid(
        data=np.zeros((grid_h, grid_w), dtype=np.int8),
        resolution=res,
        origin_x=0.0,
        origin_y=0.0,
    )
    pose = FakePoseClient(start or Pose(x=4.0, y=4.0, theta=0.0))
    occ = FakeOccupancyClient(grid)
    nav = _CoupledNav(pose)
    return pose, occ, nav


def test_start_fails_when_no_pose_or_grid() -> None:
    occ = FakeOccupancyClient()  # no grid
    pose = FakePoseClient()
    nav = FakeNavClient()
    core = PlannerCore(pose, occ, nav)
    status = core.start(
        SearchArea(
            polygon=rect_around(0, 0, 1, 1),
            anchor_pose=Pose(x=0, y=0, theta=0),
        )
    )
    assert status.state == "failed"


def test_step_completes_a_small_area() -> None:
    pose, occ, nav = _setup(start=Pose(x=4.0, y=4.0, theta=0.0))
    params = NBVParams(
        h_fov_deg=80.0,
        fov_range_m=2.5,
        n_candidates=12,
        coverage_target=0.9,
        max_iterations=20,
        lambda_cost=0.05,
    )
    core = PlannerCore(pose, occ, nav, params=params, rng_seed=7)
    area = SearchArea(
        polygon=rect_around(4.0, 4.0, width=2.0, height=2.0),
        anchor_pose=Pose(x=4.0, y=4.0, theta=0.0),
    )
    status = core.start(area)
    assert status.state == "planning"

    last = status
    for _ in range(params.max_iterations + 1):
        last = core.step()
        if last.state in ("completed", "failed", "blocked", "aborted"):
            break
    assert last.state == "completed", last.last_message
    assert last.coverage >= params.coverage_target
    assert len(nav.submitted) >= 1


def test_step_marks_blocked_when_no_progress() -> None:
    # Tiny FOV + zero candidate radius → no gainful candidates; blocked on first step.
    pose, occ, nav = _setup()
    params = NBVParams(
        h_fov_deg=1.0,
        fov_range_m=0.1,
        n_candidates=0,
        candidate_step_m=0.01,
    )
    core = PlannerCore(pose, occ, nav, params=params)
    area = SearchArea(
        polygon=rect_around(4.0, 4.0, 2.0, 2.0),
        anchor_pose=Pose(x=4.0, y=4.0, theta=0.0),
    )
    core.start(area)
    status = core.step()
    assert status.state == "blocked"


def test_step_handles_grid_resize_mid_mission() -> None:
    pose, occ, nav = _setup(start=Pose(x=4.0, y=4.0, theta=0.0))
    params = NBVParams(
        h_fov_deg=80.0,
        fov_range_m=2.5,
        n_candidates=8,
        coverage_target=0.9,
        max_iterations=10,
    )
    from litter_detector.agents.planner import PlannerCore  # noqa: F401  (already imported)

    core = PlannerCore(pose, occ, nav, params=params, rng_seed=5)
    area = SearchArea(
        polygon=rect_around(4.0, 4.0, 2.0, 2.0),
        anchor_pose=Pose(x=4.0, y=4.0, theta=0.0),
    )
    core.start(area)
    core.step()  # iteration 1 with original grid

    # Swap in a larger grid with shifted origin (simulating frontier expansion).
    larger = OccupancyGrid(
        data=np.zeros((120, 120), dtype=np.int8),
        resolution=0.1,
        origin_x=-2.0,
        origin_y=-2.0,
    )
    occ.set(larger)

    status = core.step()
    # Must NOT fail with "geometry changed" — should rebind and proceed.
    assert status.state in ("planning", "navigating", "completed", "blocked")
    assert "geometry" not in status.last_message


def test_step_handles_nav_failure() -> None:
    pose, occ, nav = _setup()
    nav.default_state = "failed"
    params = NBVParams(coverage_target=0.99, max_iterations=20)
    core = PlannerCore(pose, occ, nav, params=params)
    area = SearchArea(
        polygon=rect_around(4.0, 4.0, 2.0, 2.0),
        anchor_pose=Pose(x=4.0, y=4.0, theta=0.0),
    )
    core.start(area)
    # First few nav failures should be soft retries, not terminal.
    s1 = core.step()
    assert s1.state == "planning", s1.last_message
    # After enough consecutive nav failures the mission gives up.
    for _ in range(10):
        status = core.step()
        if status.state in ("failed", "blocked", "aborted", "completed"):
            break
    assert status.state == "failed"
    assert "nav" in status.last_message


def test_runner_drives_to_completion() -> None:
    pose, occ, nav = _setup()
    params = NBVParams(
        h_fov_deg=80.0,
        fov_range_m=2.5,
        n_candidates=12,
        coverage_target=0.9,
        max_iterations=20,
        lambda_cost=0.05,
    )
    core = PlannerCore(pose, occ, nav, params=params, rng_seed=11)
    runner = PlannerRunner(core)
    area = SearchArea(
        polygon=rect_around(4.0, 4.0, 2.0, 2.0),
        anchor_pose=Pose(x=4.0, y=4.0, theta=0.0),
    )
    runner.start(area)
    runner.join(timeout=5.0)
    final = runner.status()
    assert final.state in ("completed", "blocked")
    if final.state == "completed":
        assert final.coverage >= params.coverage_target


def test_abort_stops_runner() -> None:
    pose, occ, nav = _setup()
    # Use a slow-arriving nav so abort lands during a step.
    nav.default_state = "arrived_final"
    params = NBVParams(coverage_target=0.99, max_iterations=100)
    core = PlannerCore(pose, occ, nav, params=params)
    runner = PlannerRunner(core)
    area = SearchArea(
        polygon=rect_around(4.0, 4.0, 4.0, 4.0),
        anchor_pose=Pose(x=4.0, y=4.0, theta=0.0),
    )
    runner.start(area)
    time.sleep(0.05)
    runner.abort()
    runner.join(timeout=5.0)
    final = runner.status()
    assert final.state in ("aborted", "completed", "blocked")
