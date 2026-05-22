from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coords (top-left origin)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass(frozen=True, eq=False)
class Detection:
    """A single per-frame observation extracted from the segmentation mask.

    `appearance` is an optional L2-normalised feature vector (HSV colour
    histogram in the current setup) used by appearance-aware trackers as a
    matching tiebreaker. Equality is intentionally identity-based (eq=False)
    so the optional ndarray field doesn't break dataclass-generated __eq__.
    """

    bbox: BBox
    area_px: int  # mask pixel area (not bbox area)
    confidence: float = 1.0  # mean sigmoid prob inside the blob, or 1.0 if probs not supplied
    appearance: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Track:
    """A confirmed tracked object — what gets published and persisted."""

    id: int
    bbox: BBox
    area_px: int
    first_seen_ns: int
    last_seen_ns: int
    n_observations: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "bbox": [self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h],
            "area_px": self.area_px,
            "first_seen_ns": self.first_seen_ns,
            "last_seen_ns": self.last_seen_ns,
            "n_observations": self.n_observations,
        }
