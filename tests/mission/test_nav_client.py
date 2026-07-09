import asyncio

from litter_agents.config import (
    NAV_REQUEST_TOPIC,
    NAV_STATUS_TOPIC,
    AgentSettings,
)
from litter_agents.hunter.navigator import NavResult
from litter_agents.interfaces.robodog import (
    NavigationState,
    NavigationStatus,
    Pose2D,
)
from litter_agents.mission.nav_client import ZenohNavClient


class StaticPose:
    latest = Pose2D(x=0.0, y=0.0, theta=0.0)
    distance_traveled = 0.0

    async def wait_first(self, timeout: float) -> Pose2D:
        return self.latest

    def pose_at(self, wall_ts_ns: int) -> Pose2D:
        return self.latest


def status(state: NavigationState, request_id: str, pose=None) -> NavigationStatus:
    return NavigationStatus(state=state, request_id=request_id, current_pose=pose)


def make_client(bridge, **settings_overrides):
    settings = AgentSettings(nav_status_timeout_s=0.5, **settings_overrides)
    return ZenohNavClient(bridge, StaticPose(), settings)


def script_responses(bridge, states, pose=None):
    """On each nav/request, stream the scripted states echoing its request_id."""

    def on_publish(key, model):
        if key != NAV_REQUEST_TOPIC:
            return
        for st in states:
            bridge.push(NAV_STATUS_TOPIC, status(st, model.request_id, pose))

    bridge.on_publish = on_publish


def test_arrival(bridge):
    async def run():
        client = make_client(bridge)
        script_responses(
            bridge,
            [NavigationState.FOLLOWING, NavigationState.ARRIVED_FINAL],
            pose=Pose2D(x=2.0, y=0.0, theta=0.1),
        )
        result, last = await client.goto(Pose2D(x=2.0, y=0.0, theta=0.0), 0.4)
        assert result is NavResult.ARRIVED
        assert last is not None and last.x == 2.0
        # The published request has one must-stop segment with aligned heading.
        key, request = bridge.published[0]
        assert key == NAV_REQUEST_TOPIC
        seg = request.segments[0]
        assert seg.must_stop and seg.orientation_at_target == 0.0

    asyncio.run(run())


def test_blocked(bridge):
    async def run():
        client = make_client(bridge)
        stall = Pose2D(x=1.0, y=0.0, theta=0.0)
        script_responses(bridge, [NavigationState.BLOCKED], pose=stall)
        result, last = await client.goto(Pose2D(x=2.0, y=0.0, theta=0.0), 0.4)
        assert result is NavResult.BLOCKED
        assert last is not None and last.x == 1.0

    asyncio.run(run())


def test_stale_request_ids_are_ignored(bridge):
    async def run():
        client = make_client(bridge)

        def on_publish(key, model):
            if key != NAV_REQUEST_TOPIC:
                return
            # A stale terminal status first — must not resolve the goto.
            bridge.push(
                NAV_STATUS_TOPIC, status(NavigationState.ARRIVED_FINAL, "old-req")
            )
            bridge.push(
                NAV_STATUS_TOPIC,
                status(NavigationState.BLOCKED, model.request_id),
            )

        bridge.on_publish = on_publish
        result, _ = await client.goto(Pose2D(x=1.0, y=0.0, theta=0.0), 0.4)
        assert result is NavResult.BLOCKED

    asyncio.run(run())


def test_status_silence_times_out_and_halts(bridge):
    async def run():
        client = make_client(bridge)
        # No responses scripted at all.
        result, _ = await client.goto(Pose2D(x=1.0, y=0.0, theta=0.0), 0.4)
        assert result is NavResult.TIMEOUT
        # goto request + halt request were published.
        assert len(bridge.published) == 2
        halt_req = bridge.published[1][1]
        assert halt_req.segments[0].target.x == 0.0  # current pose

    asyncio.run(run())
