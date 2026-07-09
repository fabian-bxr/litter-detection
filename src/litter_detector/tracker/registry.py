"""SQLite-backed registry of tracked objects.

One row per stable track ID. Updated every frame the track is observed; the
first-seen timestamp is preserved across updates so we can reconstruct full
object histories from the database alone after the run.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from litter_detector.tracker.types import BBox, Track

_SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY,
    first_seen_ns INTEGER NOT NULL,
    last_seen_ns INTEGER NOT NULL,
    n_observations INTEGER NOT NULL,
    last_bbox_x INTEGER NOT NULL,
    last_bbox_y INTEGER NOT NULL,
    last_bbox_w INTEGER NOT NULL,
    last_bbox_h INTEGER NOT NULL,
    last_area_px INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_last_seen ON objects(last_seen_ns);
"""

_UPSERT = """
INSERT INTO objects (id, first_seen_ns, last_seen_ns, n_observations,
                     last_bbox_x, last_bbox_y, last_bbox_w, last_bbox_h, last_area_px)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    last_seen_ns = excluded.last_seen_ns,
    n_observations = excluded.n_observations,
    last_bbox_x = excluded.last_bbox_x,
    last_bbox_y = excluded.last_bbox_y,
    last_bbox_w = excluded.last_bbox_w,
    last_bbox_h = excluded.last_bbox_h,
    last_area_px = excluded.last_area_px
"""


class ObjectRegistry:
    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is safe here: detector loop is single-threaded;
        # this just prevents SQLite from rejecting the handle if a future caller
        # accesses it from a callback thread.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert(self, track: Track) -> None:
        self._conn.execute(_UPSERT, _row_from_track(track))
        self._conn.commit()

    def upsert_all(self, tracks: Iterable[Track]) -> None:
        rows = [_row_from_track(t) for t in tracks]
        if not rows:
            return
        self._conn.executemany(_UPSERT, rows)
        self._conn.commit()

    def get(self, track_id: int) -> Track | None:
        row = self._conn.execute(
            "SELECT id, first_seen_ns, last_seen_ns, n_observations, "
            "last_bbox_x, last_bbox_y, last_bbox_w, last_bbox_h, last_area_px "
            "FROM objects WHERE id = ?",
            (track_id,),
        ).fetchone()
        return _track_from_row(row) if row else None

    def all(self) -> list[Track]:
        rows = self._conn.execute(
            "SELECT id, first_seen_ns, last_seen_ns, n_observations, "
            "last_bbox_x, last_bbox_y, last_bbox_w, last_bbox_h, last_area_px "
            "FROM objects ORDER BY id"
        ).fetchall()
        return [_track_from_row(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


def _row_from_track(t: Track) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        t.id,
        t.first_seen_ns,
        t.last_seen_ns,
        t.n_observations,
        t.bbox.x,
        t.bbox.y,
        t.bbox.w,
        t.bbox.h,
        t.area_px,
    )


def _track_from_row(row: tuple[int, ...]) -> Track:
    return Track(
        id=int(row[0]),
        first_seen_ns=int(row[1]),
        last_seen_ns=int(row[2]),
        n_observations=int(row[3]),
        bbox=BBox(x=int(row[4]), y=int(row[5]), w=int(row[6]), h=int(row[7])),
        area_px=int(row[8]),
    )
