from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from litter_agents.hunter.navigator import NavInterface, NavResult, PoseLatest
from litter_agents.hunter.planner import ExplorationPlanner


@dataclass
class ExplorationStats:
    n_waypoints: int = 0
    n_blocked: int = 0
    stop_reason: str = ""
    waypoints: list[tuple[float, float]] = field(default_factory=list)


async def explore(
    planner: ExplorationPlanner,
    nav: NavInterface,
    pose_source: PoseLatest,
    *,
    max_speed: float,
    max_waypoints: int = 200,
    max_duration_s: float = 1800.0,
    no_gain_limit: int = 3,
    blocked_wait_s: float = 2.5,
    replan_idle_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
) -> ExplorationStats:
    """The main search loop: replan → goto → handle blocks, until covered.

    Shared verbatim between the real mission (ZenohNavClient + live pose) and
    the offline sim (FakeNav + FakePoseSource). Coverage updates happen
    outside this loop — from live poses, never pre-credited from the plan.
    """
    stats = ExplorationStats()
    started = clock()
    n_no_gain = 0
    while True:
        if planner.done():
            stats.stop_reason = "coverage_target_reached"
            break
        if clock() - started > max_duration_s:
            stats.stop_reason = "max_duration"
            break
        if stats.n_waypoints >= max_waypoints:
            stats.stop_reason = "max_waypoints"
            break

        pose = pose_source.latest
        if pose is None:
            raise RuntimeError("exploration started without a robot pose")

        candidate = planner.next_waypoint(pose)
        if candidate is None:
            n_no_gain += 1
            if n_no_gain >= no_gain_limit:
                stats.stop_reason = "no_information_gain"
                break
            # Coverage may still be absorbing recent movement — idle briefly.
            await asyncio.sleep(replan_idle_s)
            continue
        n_no_gain = 0

        stats.n_waypoints += 1
        stats.waypoints.append((candidate.target.x, candidate.target.y))
        logger.info(
            "Waypoint {}: ({:.2f}, {:.2f}) dist {:.2f} m, est. gain {:.2f} m², "
            "coverage {:.0%}",
            stats.n_waypoints,
            candidate.target.x,
            candidate.target.y,
            candidate.distance_m,
            candidate.gain_m2,
            planner.coverage.fraction(),
        )
        result, last_pose = await nav.goto(candidate.target, max_speed)
        if result is not NavResult.ARRIVED:
            stats.n_blocked += 1
            logger.warning("Goto ended {} short of ({:.2f}, {:.2f})",
                           result, candidate.target.x, candidate.target.y)
            if blocked_wait_s > 0:
                # Give the robodog executor time to retreat off the obstacle.
                await asyncio.sleep(blocked_wait_s)
            planner.register_block(
                last_pose or candidate.target,
                candidate.target,
                pose_source.latest or last_pose or pose,
            )
    logger.info(
        "Exploration finished: {} ({} waypoints, {} blocked, coverage {:.0%})",
        stats.stop_reason,
        stats.n_waypoints,
        stats.n_blocked,
        planner.coverage.fraction(),
    )
    return stats
