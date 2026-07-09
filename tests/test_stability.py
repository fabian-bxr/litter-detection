from __future__ import annotations

import json

from litter_detector.stability import _parse_angular_velocity


def test_parses_gyro_list():
    payload = json.dumps({"gyro": [0.1, -0.2, 0.05]}).encode()
    assert _parse_angular_velocity(payload) == (0.1, -0.2, 0.05)


def test_parses_angular_velocity_list():
    payload = json.dumps({"angular_velocity": [1.0, 0.0, -1.0]}).encode()
    assert _parse_angular_velocity(payload) == (1.0, 0.0, -1.0)


def test_parses_wx_wy_wz_keys():
    payload = json.dumps({"wx": 0.5, "wy": 1.5, "wz": -0.5}).encode()
    assert _parse_angular_velocity(payload) == (0.5, 1.5, -0.5)


def test_returns_none_for_unknown_schema():
    payload = json.dumps({"foo": "bar"}).encode()
    assert _parse_angular_velocity(payload) is None


def test_returns_none_for_garbage_payload():
    assert _parse_angular_velocity(b"\x00\x01not-json") is None


def test_returns_none_when_values_not_numeric():
    payload = json.dumps({"gyro": ["a", "b", "c"]}).encode()
    assert _parse_angular_velocity(payload) is None
