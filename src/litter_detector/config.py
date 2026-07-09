from __future__ import annotations

import os
import zenoh

import msgspec
from pydantic_settings import BaseSettings


class DetectionTopics(msgspec.Struct, frozen=True):
    frame: str  # Original camera frame used for detection
    mask: str  # Binary mask of detected litter
    masked_frame: str  # Camera frame with litter mask applied
    detections: str  # Detections JSON
    tracked: str  # Tracked objects with stable IDs JSON


class CameraTopics(msgspec.Struct, frozen=True):
    go2_camera: str  # Go2's camera frame
    frame: str  # Post-processed camera frame from selected source


class Topics(msgspec.Struct, frozen=True):
    detection: DetectionTopics
    camera: CameraTopics


TOPICS = Topics(
    detection=DetectionTopics(
        frame="litter/frame",
        mask="litter/mask",
        masked_frame="litter/masked_frame",
        detections="litter/detection",
        tracked="litter/tracked",
    ),
    camera=CameraTopics(go2_camera="robodog/sensors/go2_camera", frame="camera/frame"),
)


def _build_zenoh_config() -> zenoh.Config:
    """Build a zenoh.Config programmatically from settings."""
    endpoint = os.environ.get("ZENOH_ROUTER_ENDPOINT", "tcp/127.0.0.1:7447")
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
    return cfg


class Settings(BaseSettings):
    frame_width: int = 1280
    frame_height: int = 720
    otel_endpoint: str = "http://localhost:4317"
    source: str = "webcam"
    id: int | None = None

    # ── Detector postprocessing ─────────────────────────────────────────────
    detector_prob_threshold: float = 0.5
    # EWMA over the last frame's probability map: smoothed = a*prev + (1-a)*new.
    # 0.0 disables smoothing entirely; ~0.5 ≈ 2-frame memory; 0.7 ≈ 3–4 frames.
    detector_temporal_alpha: float = 0.5
    # Morphological closing on the binary mask (kernel size in px).
    # 1 or less disables; 5–7 works well for typical litter blobs.
    detector_morph_close_kernel: int = 5

    # ── Tracker ─────────────────────────────────────────────────────────────
    registry_db_path: str = "runs/objects.db"
    tracker_min_area_px: int = 50
    # Tuned for handheld / Go2-mounted camera: while walking, bboxes jump
    # between frames and detection confidence drops on blurred frames, so a
    # lower IoU gate, lower confidence floor and longer max_age keep tracks
    # alive across short detection gaps.
    tracker_iou_threshold: float = 0.2
    tracker_iou_threshold_low: float = 0.2
    # ByteTrack high/low confidence split. Detections at or above this go
    # into the first (track-spawning) matching pass; weaker detections can
    # only rescue an existing track.
    tracker_det_high_thresh: float = 0.7
    tracker_max_age: int = 75
    tracker_min_hits: int = 3
    tracker_count_min_observations: int = 10
    tracker_mask_erode_kernel: int = 3
    tracker_min_confidence: float = 0.45
    # Appearance-aware matching. 0.0 disables the colour-histogram tiebreaker
    # (pure IoU). 0.2–0.4 helps when nearby objects swap IDs under blur.
    tracker_appearance_weight: float = 0.3
    tracker_appearance_alpha: float = 0.9

    # ── Stability gate (IMU) ────────────────────────────────────────────────
    # Empty string disables the gate. Set to e.g. "robodog/sensors/imu" to
    # have the detector skip frames captured while the robot is shaking.
    stability_imu_topic: str = ""
    stability_max_angular_velocity: float = 1.0  # rad/s

    @staticmethod
    def topics() -> Topics:
        return TOPICS

    @staticmethod
    def zenoh_config() -> zenoh.Config:
        """Builds a Zenoh client config from settings."""
        return _build_zenoh_config()
