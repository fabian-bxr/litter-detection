from __future__ import annotations

from pathlib import Path

import numpy as np

from litter_detector.agents.models import Candidate, NBVParams, Pose, SearchArea
from litter_detector.agents.nbv.geometry import polygon_mask, rect_around
from litter_detector.agents.nbv.visualize import render_debug_png
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


def test_render_debug_png_writes_file(tmp_path: Path) -> None:
    grid = _grid()
    polygon = rect_around(2.0, 2.0, 2.0, 2.0)
    target = polygon_mask(grid, polygon) & grid.free_mask()
    seen = np.zeros_like(target)
    pose = Pose(x=2.0, y=2.0, theta=0.5)
    candidates = [
        Candidate(pose=Pose(x=2.5, y=2.0, theta=0.0), gain=0.1, cost_m=0.5, score=0.05),
        Candidate(pose=Pose(x=2.0, y=2.5, theta=1.5), gain=0.2, cost_m=0.5, score=0.18),
    ]
    out = tmp_path / "iter_001.png"
    render_debug_png(
        out_path=out,
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
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_planner_writes_debug_pngs(tmp_path: Path) -> None:
    grid = _grid()
    pose = FakePoseClient(Pose(x=2.0, y=2.0, theta=0.0))
    occ = FakeOccupancyClient(grid)

    class _CoupledNav(FakeNavClient):
        def submit(self, target: Pose, **kwargs) -> str:
            rid = super().submit(target, **kwargs)
            pose.set(target)
            return rid

    nav = _CoupledNav()
    params = NBVParams(
        h_fov_deg=80.0,
        fov_range_m=2.0,
        n_candidates=6,
        coverage_target=0.85,
        max_iterations=8,
        lambda_cost=0.05,
    )
    core = PlannerCore(pose, occ, nav, params=params, rng_seed=3, debug_dir=tmp_path)
    area = SearchArea(
        polygon=rect_around(2.0, 2.0, 2.0, 2.0),
        anchor_pose=Pose(x=2.0, y=2.0, theta=0.0),
    )
    core.start(area)
    for _ in range(params.max_iterations):
        s = core.step()
        if s.state in ("completed", "failed", "blocked", "aborted"):
            break

    pngs = list(tmp_path.rglob("iter_*.png"))
    assert len(pngs) >= 1
