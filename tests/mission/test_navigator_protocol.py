"""Verify ZenohNavClient satisfies NavInterface (structural subtyping check)."""

from litter_agents.hunter.navigator import NavInterface, ZenohNavClient


def test_zenoh_nav_client_satisfies_protocol():
    assert issubclass(ZenohNavClient, NavInterface)
