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

    @staticmethod
    def topics() -> Topics:
        return TOPICS

    @staticmethod
    def zenoh_config() -> zenoh.Config:
        """Builds a Zenoh client config from settings."""
        return _build_zenoh_config()
