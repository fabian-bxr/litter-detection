"""Pydantic models of the litter_detector JSON payloads on ``litter/tracked``."""

from __future__ import annotations

from pydantic import BaseModel


class TrackMsg(BaseModel):
    """One confirmed track, as serialized by Track.to_dict()."""

    id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h — pixel coords, top-left origin
    area_px: int  # connected-component area, not bbox area
    first_seen_ns: int
    last_seen_ns: int
    n_observations: int

    @property
    def cx(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2.0

    @property
    def cy(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2.0


class TrackedMsg(BaseModel):
    timestamp_ns: int
    tracks: list[TrackMsg]
