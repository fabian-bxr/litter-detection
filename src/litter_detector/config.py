from __future__ import annotations

import os
import zenoh

from typing import Dict
import msgspec
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DetectionTopics(msgspec.Struct, frozen=True):
    frames: str
    detections: str
    go2_camera: str


TOPICS = DetectionTopics(
    frames="litter/frames",
    detections="litter/detections",
    go2_camera="robodog/sensors/go2_camera",
)


def _build_zenoh_config() -> zenoh.Config:
    """Build a zenoh.Config programmatically from settings."""
    endpoint = os.environ.get("ZENOH_ROUTER_ENDPOINT", "tcp/127.0.0.1:7447")
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
    return cfg


class Settings(BaseSettings):
    frame_width: int = 640
    frame_height: int = 480

    @staticmethod
    def topics() -> DetectionTopics:
        return TOPICS

    @staticmethod
    def zenoh_config() -> zenoh.Config:
        """Builds a Zenoh client config from settings."""
        return _build_zenoh_config()
