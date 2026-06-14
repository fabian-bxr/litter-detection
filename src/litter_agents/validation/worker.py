"""ValidationWorker — buffers camera frames and validates litter detections.

Flow:
  1. Subscribes to litter/frame (JPEG) → ring-buffer (maxlen=60 ≈ 2 s at 30 Hz)
  2. Subscribes to litter/tracked (JSON TrackedMsg) → queue
  3. For each TrackedMsg, for each track with n_observations >= min_obs
     and not already validated:
       a. Find the nearest buffered JPEG to track.first_seen_ns
       b. Optionally save the JPEG to disk
       c. Call VisionAgent.validate(jpeg)
       d. Write FindingRecord to FindingsDB
       e. Optionally call the user-supplied callback (e.g. for logging)
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Callable, Awaitable

from ..config import AgentSettings
from ..interfaces.detections import TrackedMsg
from ..interfaces.robodog import Pose2D
from ..zenoh_bridge import AsyncZenoh
from .findings_db import FindingRecord, FindingsDB
from .vision_agent import LitterValidationResult, VisionAgent

FRAME_TOPIC = "litter/frame"
TRACKED_TOPIC = "litter/tracked"


class ValidationWorker:
    """Runs concurrently with the exploration loop.

    Args:
        bridge:       Shared AsyncZenoh bridge.
        db:           Open FindingsDB.
        mission_id:   Unique string for this mission run.
        run_ts:       ISO-8601 mission start timestamp.
        pose_fn:      Callable that returns the robot's current Pose2D.
        cfg:          AgentSettings.
        on_finding:   Optional async callback invoked after each validated finding.
        min_obs:      Minimum observations before a track is sent to the LLM.
        frame_buffer_size: Max JPEG frames kept in memory.
    """

    def __init__(
        self,
        bridge: AsyncZenoh,
        db: FindingsDB,
        mission_id: str,
        run_ts: str,
        pose_fn: "Callable[[], Pose2D]",
        cfg: AgentSettings | None = None,
        on_finding: "Callable[[FindingRecord, LitterValidationResult], Awaitable[None]] | None" = None,
        min_obs: int = 10,
        frame_buffer_size: int = 60,
    ) -> None:
        self._db = db
        self._mission_id = mission_id
        self._run_ts = run_ts
        self._pose_fn = pose_fn
        self._cfg = cfg or AgentSettings()
        self._on_finding = on_finding
        self._min_obs = min_obs

        # Ring buffer: (received_ns, jpeg_bytes)
        self._frame_buf: deque[tuple[int, bytes]] = deque(maxlen=frame_buffer_size)
        self._frame_q = bridge.subscribe_queue(FRAME_TOPIC, maxsize=200)
        self._tracked_q = bridge.subscribe_queue(TRACKED_TOPIC, maxsize=200)

        self._validated_ids: set[int] = set()
        self._agent = VisionAgent.from_settings(self._cfg)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="validation-worker")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        frame_consumer = asyncio.create_task(self._consume_frames(), name="val-frames")
        track_consumer = asyncio.create_task(self._consume_tracks(), name="val-tracks")
        try:
            await asyncio.gather(frame_consumer, track_consumer)
        except asyncio.CancelledError:
            frame_consumer.cancel()
            track_consumer.cancel()
            await asyncio.gather(frame_consumer, track_consumer, return_exceptions=True)
            raise

    async def _consume_frames(self) -> None:
        while True:
            raw = await self._frame_q.get()
            self._frame_buf.append((time.time_ns(), raw))

    async def _consume_tracks(self) -> None:
        while True:
            raw = await self._tracked_q.get()
            try:
                msg = TrackedMsg.model_validate_json(raw)
            except Exception:
                continue
            for track in msg.tracks:
                if track.id in self._validated_ids:
                    continue
                if track.n_observations < self._min_obs:
                    continue
                self._validated_ids.add(track.id)
                # Fire-and-forget per track so one slow LLM call doesn't
                # block processing of subsequent detections.
                asyncio.create_task(
                    self._validate_track(track.id, track.first_seen_ns),
                    name=f"validate-{track.id}",
                )

    async def _validate_track(self, track_id: int, first_seen_ns: int) -> None:
        jpeg = self._nearest_frame(first_seen_ns)
        if jpeg is None:
            return

        pose = self._pose_fn()

        # Optionally save the trigger frame to disk
        image_path: str | None = None
        images_dir = self._cfg.mission_images_path
        if images_dir:
            images_dir = Path(images_dir) / self._mission_id
            images_dir.mkdir(parents=True, exist_ok=True)
            img_file = images_dir / f"track_{track_id}_{first_seen_ns}.jpg"
            img_file.write_bytes(jpeg)
            image_path = str(img_file)

        try:
            result = await self._agent.validate(jpeg)
        except Exception as exc:
            # LLM failure → store as unconfirmed with note
            result = type("R", (), {  # lightweight fallback object
                "is_litter": False,
                "confidence": 0.0,
                "description": f"LLM error: {exc}",
                "category": None,
            })()  # type: ignore[assignment]

        rec = FindingRecord(
            mission_id=self._mission_id,
            run_ts=self._run_ts,
            track_id=track_id,
            confirmed=result.is_litter,
            confidence=result.confidence,
            description=result.description,
            category=result.category,
            pose_x=pose.x,
            pose_y=pose.y,
            pose_theta=pose.theta,
            image_path=image_path,
        )
        await self._db.insert(rec)

        if self._on_finding:
            await self._on_finding(rec, result)  # type: ignore[arg-type]

    def _nearest_frame(self, ts_ns: int) -> bytes | None:
        if not self._frame_buf:
            return None
        best_bytes, best_dt = None, float("inf")
        for recv_ns, jpeg in self._frame_buf:
            dt = abs(recv_ns - ts_ns)
            if dt < best_dt:
                best_dt, best_bytes = dt, jpeg
        return best_bytes

    @property
    def validated_count(self) -> int:
        return len(self._validated_ids)
