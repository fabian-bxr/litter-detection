"""Integration: the full exploration loop with FakeNav on the real lab map."""

import asyncio
from pathlib import Path

import pytest

from litter_agents.hunter.explore import explore
from litter_agents.hunter.params import HunterParams
from litter_agents.hunter.planner import ExplorationPlanner
from litter_agents.interfaces.mission import SearchAreaSpec
from litter_agents.mapping.provider import FileMapProvider
from litter_agents.mapping.raster import rasterize_area
from litter_agents.sim.fake_nav import FakeNav, FakePoseSource
from litter_agents.sim.sim_main import default_start

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS = HunterParams()


def run_mission(blocked_discs=None, radius_m=4.0):
    async def _run():
        grid = await FileMapProvider(REPO_ROOT / "my_lab_grid.yaml").load()
        start = default_start(grid, PARAMS.robot_radius_m)
        spec = SearchAreaSpec(shape="circle", radius_m=radius_m)
        target = rasterize_area(spec, start, grid)
        planner = ExplorationPlanner(grid, target, PARAMS, start)
        pose_source = FakePoseSource(start)
        nav = FakeNav(
            pose_source,
            blocked_discs=blocked_discs or [],
            on_tick=planner.coverage.update,
        )
        planner.coverage.update(start)
        stats = await explore(
            planner,
            nav,
            pose_source,
            max_speed=0.6,
            max_waypoints=80,
            max_duration_s=120.0,
            blocked_wait_s=0.0,
            replan_idle_s=0.0,
        )
        return planner, stats

    return asyncio.run(_run())


def test_covers_lab_map():
    planner, stats = run_mission()
    assert stats.stop_reason in ("coverage_target_reached", "no_information_gain")
    assert planner.coverage.fraction() >= PARAMS.coverage_target_fraction
    assert stats.n_waypoints <= 60


def test_blocked_disc_is_handled_and_mission_still_ends():
    planner, stats_free = run_mission()
    # Drop an invisible obstacle onto the first leg of the unblocked run.
    assert stats_free.waypoints
    bx, by = stats_free.waypoints[0]
    planner, stats = run_mission(blocked_discs=[(bx, by, 0.5)])
    assert stats.n_blocked >= 1
    assert planner.n_blocked >= 1
    assert len(planner.blacklist) >= 1
    assert len(planner.dynamic) == 1
    assert stats.stop_reason in (
        "coverage_target_reached",
        "no_information_gain",
        "max_waypoints",
    )
    # Coverage still mostly achieved despite the obstacle.
    assert planner.coverage.fraction() >= 0.6


@pytest.mark.parametrize("radius", [2.0, 6.0])
def test_various_radii_terminate(radius):
    _, stats = run_mission(radius_m=radius)
    assert stats.stop_reason in ("coverage_target_reached", "no_information_gain")
