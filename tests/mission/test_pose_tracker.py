import asyncio
import math
from datetime import datetime, timezone

import pytest

from litter_agents.config import ROBODOG_POSE_TOPIC
from litter_agents.interfaces.robodog import OdometryState
from litter_agents.mission.pose_tracker import ZenohPoseTracker


def odo(x: float, y: float, yaw: float, ts: float) -> OdometryState:
    return OdometryState(
        x=x,
        y=y,
        quaternion=[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)],
        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
    )


def test_latest_and_distance(bridge):
    async def run():
        tracker = ZenohPoseTracker(bridge)
        assert tracker.latest is None
        bridge.push(ROBODOG_POSE_TOPIC, odo(0.0, 0.0, 0.0, 100.0))
        bridge.push(ROBODOG_POSE_TOPIC, odo(3.0, 4.0, math.pi / 2, 101.0))
        pose = await tracker.wait_first(1.0)
        assert pose.x == pytest.approx(3.0)
        assert pose.theta == pytest.approx(math.pi / 2)
        assert tracker.distance_traveled == pytest.approx(5.0)

    asyncio.run(run())


def test_pose_at_matches_nearest_timestamp(bridge):
    async def run():
        tracker = ZenohPoseTracker(bridge)
        for i in range(10):
            bridge.push(ROBODOG_POSE_TOPIC, odo(float(i), 0.0, 0.0, 100.0 + i))
        # 100.4 s is nearest to the i=0..9 sample at 100.0 s? No: 100.4 → i=0.
        match = tracker.pose_at(int(100.4e9))
        assert match is not None and match.x == pytest.approx(0.0)
        match = tracker.pose_at(int(105.6e9))
        assert match is not None and match.x == pytest.approx(6.0)

    asyncio.run(run())


def test_pose_at_falls_back_to_latest_when_stale(bridge):
    async def run():
        tracker = ZenohPoseTracker(bridge)
        bridge.push(ROBODOG_POSE_TOPIC, odo(7.0, 0.0, 0.0, 100.0))
        # Query 1 hour away from any buffered pose.
        match = tracker.pose_at(int(3700.0e9))
        assert match is not None and match.x == pytest.approx(7.0)

    asyncio.run(run())


def test_wait_first_times_out(bridge):
    async def run():
        tracker = ZenohPoseTracker(bridge)
        with pytest.raises(asyncio.TimeoutError):
            await tracker.wait_first(0.05)

    asyncio.run(run())
