"""IMU-based stability gate for the detector.

Subscribes to a Zenoh topic carrying the robot's angular velocity and exposes
`is_stable()` for the detector loop to skip frames captured while the robot is
shaking too hard for the model to be useful (motion blur dominates).

Expected payload (JSON): one of
    {"gyro": [x, y, z]}            # rad/s, body frame
    {"angular_velocity": [x, y, z]}
    {"wx": x, "wy": y, "wz": z}

If your Go2 publishes a different schema, adjust `_parse_angular_velocity`.

Disabled by default — set `stability_imu_topic` in Settings to enable.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

import zenoh
from loguru import logger


def _parse_angular_velocity(payload: bytes) -> tuple[float, float, float] | None:
    """Best-effort extraction of (wx, wy, wz) in rad/s from a JSON payload."""
    try:
        data: Any = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("gyro", "angular_velocity"):
        v = data.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            try:
                return float(v[0]), float(v[1]), float(v[2])
            except (TypeError, ValueError):
                return None
    if all(k in data for k in ("wx", "wy", "wz")):
        try:
            return float(data["wx"]), float(data["wy"]), float(data["wz"])
        except (TypeError, ValueError):
            return None
    return None


class StabilityGate:
    """Tracks angular-velocity magnitude from an IMU stream over Zenoh.

    Thread-safety: `is_stable()` is safe to call from the detector loop while
    the Zenoh callback writes the latest sample from its own thread.
    """

    def __init__(
        self,
        session: zenoh.Session,
        topic: str,
        max_angular_velocity: float,
        sample_timeout_s: float = 0.5,
    ) -> None:
        """
        Args:
            session: an already-open Zenoh session (shared with the detector).
            topic: Zenoh key expression to subscribe to. Empty string =
                disabled (`is_stable` always returns True).
            max_angular_velocity: ||ω|| above this (rad/s) marks the frame as
                unstable and the detector should skip it.
            sample_timeout_s: if no IMU sample has arrived within this many
                seconds, assume the IMU stream is dead and stop gating
                (fail-open: don't strand the detector when the IMU drops).
        """
        self.topic = topic
        self.max_angular_velocity = max_angular_velocity
        self.sample_timeout_s = sample_timeout_s
        self._lock = threading.Lock()
        self._latest_magnitude: float = 0.0
        self._latest_ts: float = 0.0
        self._enabled = bool(topic)
        self._subscriber: zenoh.Subscriber | None = None
        if self._enabled:
            self._subscriber = session.declare_subscriber(topic, self._on_sample)
            logger.info(
                f"StabilityGate enabled on '{topic}' "
                f"(max ω = {max_angular_velocity:.2f} rad/s)"
            )

    def _on_sample(self, sample: zenoh.Sample) -> None:
        omega = _parse_angular_velocity(bytes(sample.payload))
        if omega is None:
            return
        mag = math.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)
        with self._lock:
            self._latest_magnitude = mag
            self._latest_ts = time.monotonic()

    def is_stable(self) -> bool:
        """True iff the most recent IMU reading is below the threshold.

        Fail-open when the gate is disabled or the IMU stream has gone stale.
        """
        if not self._enabled:
            return True
        with self._lock:
            mag = self._latest_magnitude
            ts = self._latest_ts
        if ts == 0.0 or (time.monotonic() - ts) > self.sample_timeout_s:
            return True
        return mag <= self.max_angular_velocity

    @property
    def latest_magnitude(self) -> float:
        with self._lock:
            return self._latest_magnitude

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.undeclare()
            self._subscriber = None
