from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import zenoh
from loguru import logger

from litter_agents.config import AgentSettings, repo_path
from litter_agents.interfaces.detections import TrackedMsg, TrackMsg
from litter_agents.interfaces.mission import LitterValidation
from litter_agents.interfaces.robodog import Pose2D
from litter_agents.mission.pose_tracker import PoseSource
from litter_agents.validation import crops
from litter_agents.validation.findings import (
    FindingRow,
    FindingsRepository,
    validation_to_raw,
)
from litter_agents.zenoh_bridge import Bridge

# (attachment timestamp_ns | None, jpeg bytes)
FrameMsg = tuple[int | None, bytes]


@dataclass
class ValidationJob:
    track: TrackMsg
    crop_jpeg: bytes
    context_jpeg: bytes | None
    robot_pose: Pose2D | None
    bearing_rad: float


ValidateFn = Callable[[ValidationJob], Awaitable[LitterValidation]]


def decode_frame(sample: zenoh.Sample) -> FrameMsg:
    ts: int | None = None
    attachment = sample.attachment
    if attachment is not None:
        try:
            ts = int(attachment.to_bytes().decode())
        except (ValueError, UnicodeDecodeError):
            ts = None
    return ts, sample.payload.to_bytes()


def decode_tracked(sample: zenoh.Sample) -> TrackedMsg:
    return TrackedMsg.model_validate_json(sample.payload.to_bytes())


class DetectionValidationWorker:
    """Turns stable tracks into validated findings.

    Subscribes to ``litter/tracked`` + ``litter/frame``; when a track passes
    the readiness gate it is cropped from the matching frame and queued for
    the (slow) vision agent. Zenoh handlers only enqueue — every LLM call runs
    in one of ``validation_concurrency`` consumer coroutines, so a burst of
    detections can never stall the subscribers. Each track id is judged once
    per mission; negatives are persisted as ``rejected`` so they never
    re-queue.
    """

    def __init__(
        self,
        bridge: Bridge,
        pose_source: PoseSource,
        repo: FindingsRepository,
        validate: ValidateFn,
        settings: AgentSettings,
        mission_id: str,
        *,
        model_name: str | None = None,
    ) -> None:
        self._pose_source = pose_source
        self._repo = repo
        self._validate = validate
        self._settings = settings
        self._mission_id = mission_id
        self._model_name = model_name or settings.vision_model_name
        self._images_dir = repo_path(settings.findings_dir) / mission_id / "findings"

        self._frames: deque[FrameMsg] = deque(maxlen=30)
        self._jobs: asyncio.Queue[ValidationJob] = asyncio.Queue(
            maxsize=settings.validation_queue_size
        )
        self._processed: set[int] = repo.processed_track_ids(mission_id)
        self._in_flight: set[int] = set()
        self._tasks: list[asyncio.Task] = []

        topics = settings.topics()
        self._tracked_queue = bridge.subscribe_queue(
            topics.detection.tracked, decode_tracked, maxsize=8
        )
        bridge.subscribe(topics.detection.frame, decode_frame, self._frames.append)

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._produce(), name="validation-producer")]
        self._tasks += [
            asyncio.create_task(self._consume(), name=f"validation-consumer-{i}")
            for i in range(self._settings.validation_concurrency)
        ]

    async def stop(self, drain_timeout_s: float = 90.0) -> None:
        """Stop intake, let queued jobs finish (bounded), then cancel."""
        if not self._tasks:
            return
        self._tasks[0].cancel()  # producer
        try:
            await asyncio.wait_for(self._jobs.join(), drain_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "Validation queue not drained after {} s; {} jobs dropped",
                drain_timeout_s,
                self._jobs.qsize(),
            )
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    # ── intake ──────────────────────────────────────────────────────────────

    def _frame_for(self, timestamp_ns: int) -> bytes | None:
        for ts, payload in reversed(self._frames):
            if ts == timestamp_ns:
                return payload
        # No attachment match (detector without the attachment patch, or the
        # frame already rotated out): newest frame is at most one frame off
        # and the crop padding absorbs the bbox drift.
        return self._frames[-1][1] if self._frames else None

    def _ready(self, track: TrackMsg, frame_shape: tuple[int, ...]) -> bool:
        s = self._settings
        x, y, w, h = track.bbox
        height, width = frame_shape[:2]
        margin = s.validation_border_margin_px
        return (
            track.n_observations >= s.validation_min_observations
            and w >= s.validation_min_bbox_px
            and h >= s.validation_min_bbox_px
            and track.area_px >= s.validation_min_area_px
            # Border-touching boxes crop partial objects — wait for a better view.
            and x >= margin
            and y >= margin
            and x + w <= width - margin
            and y + h <= height - margin
        )

    async def _produce(self) -> None:
        while True:
            msg = await self._tracked_queue.get()
            for track in msg.tracks:
                if track.id in self._processed or track.id in self._in_flight:
                    continue
                payload = self._frame_for(msg.timestamp_ns)
                if payload is None:
                    continue
                frame = crops.decode_jpeg(payload)
                if frame is None or not self._ready(track, frame.shape):
                    continue
                job = ValidationJob(
                    track=track,
                    crop_jpeg=crops.encode_jpeg(
                        crops.crop_with_padding(
                            frame, track.bbox, self._settings.validation_crop_pad
                        )
                    ),
                    context_jpeg=(
                        crops.encode_jpeg(crops.context_image(frame, track.bbox))
                        if self._settings.validation_send_context
                        else None
                    ),
                    robot_pose=self._pose_source.pose_at(track.last_seen_ns),
                    bearing_rad=crops.camera_bearing_rad(
                        track.bbox,
                        frame.shape[1],
                        math.radians(self._settings.camera_fov_deg),
                    ),
                )
                self._in_flight.add(track.id)
                if self._jobs.full():
                    dropped = self._jobs.get_nowait()
                    self._jobs.task_done()
                    self._in_flight.discard(dropped.track.id)
                    logger.warning(
                        "Validation queue full — dropped oldest job (track {})",
                        dropped.track.id,
                    )
                self._jobs.put_nowait(job)

    # ── consumption ─────────────────────────────────────────────────────────

    async def _consume(self) -> None:
        while True:
            job = await self._jobs.get()
            try:
                await self._process(job)
            except Exception:
                logger.opt(exception=True).error(
                    "Validation job for track {} crashed", job.track.id
                )
            finally:
                self._in_flight.discard(job.track.id)
                self._processed.add(job.track.id)
                self._jobs.task_done()

    async def _run_validation(self, job: ValidationJob) -> LitterValidation:
        try:
            return await asyncio.wait_for(
                self._validate(job), self._settings.llm_timeout_s
            )
        except Exception as first_error:
            # CancelledError is BaseException in 3.11 and propagates untouched.
            logger.warning(
                "Validation of track {} failed ({}); retrying once",
                job.track.id,
                type(first_error).__name__,
            )
            await asyncio.sleep(self._settings.llm_retry_backoff_s)
            return await asyncio.wait_for(
                self._validate(job), self._settings.llm_timeout_s
            )

    async def _process(self, job: ValidationJob) -> None:
        track = job.track
        try:
            validation = await self._run_validation(job)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error("Validation of track {} gave up: {}", track.id, error)
            self._insert(job, status="error", validation=None)
            return
        status = "validated" if validation.is_litter else "rejected"
        self._insert(job, status=status, validation=validation)
        logger.info(
            "Track {} {}: {} ({:.0%})",
            track.id,
            status,
            validation.category or validation.description[:60],
            validation.confidence,
        )

    def _save_images(self, job: ValidationJob) -> tuple[str, str | None]:
        self._images_dir.mkdir(parents=True, exist_ok=True)
        crop_path = self._images_dir / f"track_{job.track.id}_crop.jpg"
        crop_path.write_bytes(job.crop_jpeg)
        ctx_path = None
        if job.context_jpeg is not None:
            ctx_path = self._images_dir / f"track_{job.track.id}_ctx.jpg"
            ctx_path.write_bytes(job.context_jpeg)
        return str(crop_path), str(ctx_path) if ctx_path else None

    def _insert(
        self, job: ValidationJob, *, status: str, validation: LitterValidation | None
    ) -> None:
        crop_path, ctx_path = self._save_images(job)
        self._repo.insert_finding(
            FindingRow(
                mission_id=self._mission_id,
                track_id=job.track.id,
                status=status,
                category=validation.category if validation else None,
                confidence=validation.confidence if validation else None,
                description=validation.description if validation else None,
                robot_pose=job.robot_pose,
                bearing_rad=job.bearing_rad,
                bbox=job.track.bbox,
                area_px=job.track.area_px,
                n_observations=job.track.n_observations,
                first_seen_ns=job.track.first_seen_ns,
                last_seen_ns=job.track.last_seen_ns,
                validated_at_ns=time.time_ns(),
                image_path=crop_path,
                context_image_path=ctx_path,
                model_name=self._model_name,
                raw_response=validation_to_raw(validation) if validation else None,
            )
        )
