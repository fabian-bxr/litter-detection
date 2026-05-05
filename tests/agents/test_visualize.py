from __future__ import annotations

import time

import numpy as np

from litter_detector.agents.models import Candidate, NBVParams, Pose, SearchArea
from litter_detector.agents.nbv.geometry import polygon_mask, rect_around
from litter_detector.agents.nbv.visualize import render_debug_jpeg
from litter_detector.agents.planner import PlannerCore
from litter_detector.agents.tools.nav import FakeNavClient
from litter_detector.agents.tools.occupancy import FakeOccupancyClient, OccupancyGrid
from litter_detector.agents.tools.pose import FakePoseClient


def _grid() -> OccupancyGrid:
    return OccupancyGrid(
        data=np.zeros((40, 40), dtype=np.int8),
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
    )


def test_render_debug_jpeg_returns_jpeg_bytes() -> None:
    grid = _grid()
    polygon = rect_around(2.0, 2.0, 2.0, 2.0)
    target = polygon_mask(grid, polygon) & grid.free_mask()
    seen = np.zeros_like(target)
    pose = Pose(x=2.0, y=2.0, theta=0.5)
    candidates = [
        Candidate(pose=Pose(x=2.5, y=2.0, theta=0.0), gain=0.1, cost_m=0.5, score=0.05),
        Candidate(pose=Pose(x=2.0, y=2.5, theta=1.5), gain=0.2, cost_m=0.5, score=0.18),
    ]
    blob = render_debug_jpeg(
        grid=grid,
        seen_mask=seen,
        target_mask=target,
        polygon=polygon,
        pose=pose,
        candidates=candidates,
        chosen=candidates[1],
        iteration=1,
        coverage=0.0,
    )
    assert isinstance(blob, bytes)
    assert len(blob) > 1000
    # JPEG SOI magic
    assert blob[:2] == b"\xff\xd8"
    assert blob[-2:] == b"\xff\xd9"


def test_planner_publishes_debug_jpegs() -> None:
    grid = _grid()
    pose = FakePoseClient(Pose(x=2.0, y=2.0, theta=0.0))
    occ = FakeOccupancyClient(grid)

    class _CoupledNav(FakeNavClient):
        def submit(self, target: Pose, **kwargs) -> str:
            rid = super().submit(target, **kwargs)
            pose.set(target)
            return rid

    class _Sink:
        def __init__(self) -> None:
            self.payloads: list[bytes] = []

        def put(self, payload: bytes) -> None:
            self.payloads.append(payload)

    nav = _CoupledNav()
    sink = _Sink()
    params = NBVParams(
        h_fov_deg=80.0,
        fov_range_m=2.0,
        n_candidates=6,
        coverage_target=0.85,
        max_iterations=8,
        lambda_cost=0.05,
    )
    core = PlannerCore(
        pose, occ, nav, params=params, rng_seed=3, debug_publisher=sink
    )
    area = SearchArea(
        polygon=rect_around(2.0, 2.0, 2.0, 2.0),
        anchor_pose=Pose(x=2.0, y=2.0, theta=0.0),
    )
    core.start(area)
    for _ in range(params.max_iterations):
        s = core.step()
        if s.state in ("completed", "failed", "blocked", "aborted"):
            break

    # Periodic debug publisher runs at ~2 Hz on its own thread; give it a tick.
    deadline = time.monotonic() + 2.0
    while not sink.payloads and time.monotonic() < deadline:
        time.sleep(0.05)
    core.stop_periodic_debug()

    assert len(sink.payloads) >= 1
    for blob in sink.payloads:
        assert blob[:2] == b"\xff\xd8"
