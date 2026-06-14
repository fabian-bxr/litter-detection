"""Pydantic models for litter/tracked Zenoh topic."""

from __future__ import annotations

from pydantic import BaseModel


class TrackMsg(BaseModel):
    id: int
    bbox: list[int]          # [x, y, w, h] in pixels
    area_px: float
    first_seen_ns: int
    last_seen_ns: int
    n_observations: int


class TrackedMsg(BaseModel):
    timestamp_ns: int
    tracks: list[TrackMsg]
