from __future__ import annotations

import math

from litter_detector.agents.models import Pose
from litter_detector.agents.tools.pose import FakePoseClient, quaternion_to_yaw


def test_quaternion_to_yaw_identity() -> None:
    assert abs(quaternion_to_yaw(0, 0, 0, 1)) < 1e-9


def test_quaternion_to_yaw_90deg() -> None:
    # 90° rotation about Z: (qx,qy,qz,qw) = (0, 0, sin(45°), cos(45°))
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    yaw = quaternion_to_yaw(0.0, 0.0, s, c)
    assert abs(yaw - math.pi / 2) < 1e-6


def test_fake_pose_client() -> None:
    fake = FakePoseClient()
    assert fake.get(timeout=0.0) is None
    p = Pose(x=1.0, y=2.0, theta=0.3)
    fake.set(p)
    assert fake.get(timeout=0.0) == p
