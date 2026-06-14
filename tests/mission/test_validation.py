"""Tests for Phase 4: FindingsDB and ValidationWorker frame-buffer matching."""

import asyncio
import tempfile
import time
from collections import deque
from pathlib import Path

import pytest

from litter_agents.validation.findings_db import FindingRecord, FindingsDB


# ---------------------------------------------------------------------------
# FindingsDB
# ---------------------------------------------------------------------------


async def test_findings_db_insert_and_query():
    with tempfile.TemporaryDirectory() as d:
        db = FindingsDB(Path(d) / "test.db")
        await db.init()

        rec = FindingRecord(
            mission_id="m1",
            run_ts="2026-06-13T10:00:00Z",
            track_id=7,
            confirmed=True,
            confidence=0.87,
            description="Plastic bottle on floor.",
            category="plastic bottle",
            pose_x=1.0,
            pose_y=2.0,
            pose_theta=0.0,
            image_path=None,
        )
        row_id = await db.insert(rec)
        assert row_id == 1

        rows = await db.query_mission("m1")
        assert len(rows) == 1
        assert rows[0]["track_id"] == 7
        assert rows[0]["confirmed"] == 1
        assert rows[0]["category"] == "plastic bottle"


async def test_findings_db_filters_by_mission():
    with tempfile.TemporaryDirectory() as d:
        db = FindingsDB(Path(d) / "test.db")
        await db.init()

        for mid, tid in [("mission-A", 1), ("mission-B", 2), ("mission-A", 3)]:
            await db.insert(
                FindingRecord(
                    mission_id=mid,
                    run_ts="2026-06-13T10:00:00Z",
                    track_id=tid,
                    confirmed=False,
                    confidence=0.1,
                    description="Test.",
                    category=None,
                    pose_x=0.0,
                    pose_y=0.0,
                    pose_theta=0.0,
                    image_path=None,
                )
            )

        a_rows = await db.query_mission("mission-A")
        b_rows = await db.query_mission("mission-B")
        assert len(a_rows) == 2
        assert len(b_rows) == 1


# ---------------------------------------------------------------------------
# Frame buffer nearest-match (isolated from Zenoh)
# ---------------------------------------------------------------------------


def _nearest_frame_from_buf(
    buf: deque[tuple[int, bytes]], ts_ns: int
) -> bytes | None:
    """Reproduce ValidationWorker._nearest_frame logic for testing."""
    if not buf:
        return None
    best_bytes, best_dt = None, float("inf")
    for recv_ns, jpeg in buf:
        dt = abs(recv_ns - ts_ns)
        if dt < best_dt:
            best_dt, best_bytes = dt, jpeg
    return best_bytes


def test_frame_buffer_nearest_match():
    buf: deque[tuple[int, bytes]] = deque(maxlen=10)
    now = time.time_ns()
    # Three frames: 100 ms apart
    for i in range(3):
        buf.append((now + i * 100_000_000, bytes([i])))

    # Query at exactly the second frame's timestamp
    result = _nearest_frame_from_buf(buf, now + 100_000_000)
    assert result == bytes([1])


def test_frame_buffer_empty_returns_none():
    buf: deque[tuple[int, bytes]] = deque(maxlen=10)
    assert _nearest_frame_from_buf(buf, time.time_ns()) is None


def test_frame_buffer_picks_closest_not_first():
    buf: deque[tuple[int, bytes]] = deque(maxlen=10)
    now = time.time_ns()
    buf.append((now, b"early"))
    buf.append((now + 900_000_000, b"late"))

    # A query at now+800ms should match the "late" frame (100ms away) over "early" (800ms away)
    result = _nearest_frame_from_buf(buf, now + 800_000_000)
    assert result == b"late"
