import asyncio

import cv2
import numpy as np
import pytest

from litter_agents.config import AgentSettings
from litter_agents.interfaces.detections import TrackedMsg, TrackMsg
from litter_agents.interfaces.mission import LitterValidation
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.validation.findings import FindingsRepository
from litter_agents.validation.worker import DetectionValidationWorker

FRAME_TOPIC = "litter/frame"
TRACKED_TOPIC = "litter/tracked"


class StaticPose:
    latest = Pose2D(x=1.0, y=2.0, theta=0.0)
    distance_traveled = 0.0

    async def wait_first(self, timeout: float) -> Pose2D:
        return self.latest

    def pose_at(self, wall_ts_ns: int) -> Pose2D:
        return self.latest


def jpeg_frame(color=(0, 0, 255), w=640, h=480) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def track(track_id=1, bbox=(100, 100, 64, 64), area_px=2000, n_obs=12) -> TrackMsg:
    return TrackMsg(
        id=track_id,
        bbox=bbox,
        area_px=area_px,
        first_seen_ns=100,
        last_seen_ns=200,
        n_observations=n_obs,
    )


ACCEPT = LitterValidation(
    is_litter=True, category="plastic", confidence=0.9, description="a bottle"
)
REJECT = LitterValidation(is_litter=False, confidence=0.7, description="a shadow")


def make_settings(tmp_path, **overrides) -> AgentSettings:
    defaults = dict(
        findings_db_path=str(tmp_path / "findings.db"),
        findings_dir=str(tmp_path / "missions"),
        llm_timeout_s=1.0,
        llm_retry_backoff_s=0.0,
        validation_concurrency=1,
    )
    defaults.update(overrides)
    return AgentSettings(**defaults)


def run_worker(bridge, tmp_path, validate, pushes, settings=None):
    """Start the worker, deliver scripted messages, drain, return the repo."""

    async def _run():
        s = settings or make_settings(tmp_path)
        repo = FindingsRepository(s.findings_db_path)
        worker = DetectionValidationWorker(
            bridge, StaticPose(), repo, validate, s, mission_id="m1"
        )
        worker.start()
        for key, value in pushes:
            bridge.push(key, value)
        await asyncio.sleep(0.05)
        await worker.stop(drain_timeout_s=5.0)
        return repo

    return asyncio.run(_run())


def test_validated_finding_with_images(bridge, tmp_path):
    calls = []

    async def validate(job):
        calls.append(job)
        return ACCEPT

    repo = run_worker(
        bridge,
        tmp_path,
        validate,
        [
            (FRAME_TOPIC, (1000, jpeg_frame())),
            (TRACKED_TOPIC, TrackedMsg(timestamp_ns=1000, tracks=[track()])),
        ],
    )
    rows = repo.findings("m1")
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "validated"
    assert row.category == "plastic"
    assert row.robot_pose is not None and row.robot_pose.x == 1.0
    assert row.image_path and row.context_image_path
    from pathlib import Path

    assert Path(row.image_path).exists()
    assert Path(row.context_image_path).exists()
    # Crop has padded bbox dimensions (64 px + 2×15% ≈ 83).
    crop = cv2.imread(row.image_path)
    assert 70 <= crop.shape[0] <= 100
    assert len(calls) == 1


def test_rejected_is_persisted(bridge, tmp_path):
    async def validate(job):
        return REJECT

    repo = run_worker(
        bridge,
        tmp_path,
        validate,
        [
            (FRAME_TOPIC, (1000, jpeg_frame())),
            (TRACKED_TOPIC, TrackedMsg(timestamp_ns=1000, tracks=[track()])),
        ],
    )
    assert repo.status_counts("m1") == {"rejected": 1}


def test_failing_llm_writes_error_row_after_retry(bridge, tmp_path):
    attempts = []

    async def validate(job):
        attempts.append(1)
        raise RuntimeError("ollama down")

    repo = run_worker(
        bridge,
        tmp_path,
        validate,
        [
            (FRAME_TOPIC, (1000, jpeg_frame())),
            (TRACKED_TOPIC, TrackedMsg(timestamp_ns=1000, tracks=[track()])),
        ],
    )
    assert repo.status_counts("m1") == {"error": 1}
    assert len(attempts) == 2  # initial + one retry


def test_readiness_gates(bridge, tmp_path):
    async def validate(job):
        return ACCEPT

    not_ready = [
        track(track_id=1, n_obs=5),  # too few observations
        track(track_id=2, bbox=(100, 100, 20, 64)),  # too narrow
        track(track_id=3, area_px=100),  # too small a blob
        track(track_id=4, bbox=(0, 100, 64, 64)),  # touches the border
        track(track_id=5, bbox=(600, 440, 64, 64)),  # exceeds the border
    ]
    repo = run_worker(
        bridge,
        tmp_path,
        validate,
        [
            (FRAME_TOPIC, (1000, jpeg_frame())),
            (TRACKED_TOPIC, TrackedMsg(timestamp_ns=1000, tracks=not_ready)),
        ],
    )
    assert repo.findings("m1") == []


def test_track_validated_only_once(bridge, tmp_path):
    calls = []

    async def validate(job):
        calls.append(job.track.id)
        return ACCEPT

    repo = run_worker(
        bridge,
        tmp_path,
        validate,
        [
            (FRAME_TOPIC, (1000, jpeg_frame())),
            (TRACKED_TOPIC, TrackedMsg(timestamp_ns=1000, tracks=[track()])),
            (TRACKED_TOPIC, TrackedMsg(timestamp_ns=1001, tracks=[track(n_obs=20)])),
        ],
    )
    assert len(repo.findings("m1")) == 1
    assert calls == [1]


def test_frame_pairing_by_attachment(bridge, tmp_path):
    seen_crops = []

    async def validate(job):
        seen_crops.append(job.crop_jpeg)
        return ACCEPT

    red = jpeg_frame(color=(0, 0, 255))
    blue = jpeg_frame(color=(255, 0, 0))
    run_worker(
        bridge,
        tmp_path,
        validate,
        [
            (FRAME_TOPIC, (1000, red)),
            (FRAME_TOPIC, (2000, blue)),
            # Tracked message belongs to the red frame, not the newest one.
            (TRACKED_TOPIC, TrackedMsg(timestamp_ns=1000, tracks=[track()])),
        ],
    )
    assert len(seen_crops) == 1
    crop = cv2.imdecode(np.frombuffer(seen_crops[0], np.uint8), cv2.IMREAD_COLOR)
    b, g, r = crop[10, 10]
    assert r > 200 and b < 50  # red frame was used


def test_frame_pairing_falls_back_to_newest(bridge, tmp_path):
    seen_crops = []

    async def validate(job):
        seen_crops.append(job.crop_jpeg)
        return ACCEPT

    run_worker(
        bridge,
        tmp_path,
        validate,
        [
            (FRAME_TOPIC, (None, jpeg_frame(color=(0, 0, 255)))),
            (FRAME_TOPIC, (None, jpeg_frame(color=(255, 0, 0)))),
            (TRACKED_TOPIC, TrackedMsg(timestamp_ns=1234, tracks=[track()])),
        ],
    )
    crop = cv2.imdecode(np.frombuffer(seen_crops[0], np.uint8), cv2.IMREAD_COLOR)
    b, g, r = crop[10, 10]
    assert b > 200 and r < 50  # newest (blue) frame was used


def test_no_frames_means_no_job(bridge, tmp_path):
    async def validate(job):
        pytest.fail("must not be called")

    repo = run_worker(
        bridge,
        tmp_path,
        validate,
        [(TRACKED_TOPIC, TrackedMsg(timestamp_ns=1000, tracks=[track()]))],
    )
    assert repo.findings("m1") == []
