"""Pure cv2 helpers for preparing detection images for the vision agent."""

from __future__ import annotations

import cv2
import numpy as np


def decode_jpeg(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def crop_with_padding(
    frame: np.ndarray, bbox: tuple[int, int, int, int], pad_frac: float
) -> np.ndarray:
    """Crop bbox plus a fractional margin, clamped to the frame."""
    x, y, w, h = bbox
    pad_x = int(round(w * pad_frac))
    pad_y = int(round(h * pad_frac))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame.shape[1], x + w + pad_x)
    y1 = min(frame.shape[0], y + h + pad_y)
    return frame[y0:y1, x0:x1]


def context_image(
    frame: np.ndarray, bbox: tuple[int, int, int, int], target_width: int = 640
) -> np.ndarray:
    """Downscaled full frame with the detection boxed — scene context for the LLM."""
    x, y, w, h = bbox
    img = frame.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    if img.shape[1] > target_width:
        scale = target_width / img.shape[1]
        img = cv2.resize(img, (target_width, int(img.shape[0] * scale)))
    return img


def camera_bearing_rad(
    bbox: tuple[int, int, int, int], frame_width: int, fov_rad: float
) -> float:
    """Horizontal bearing of the bbox center: 0 = straight ahead, +left.

    Sign convention matches the robot frame (+y left): an object on the image's
    left half has positive bearing.
    """
    cx = bbox[0] + bbox[2] / 2.0
    return -(cx / frame_width - 0.5) * fov_rad
