"""SQLite-backed registry of tracked objects (YOLO detector).

One row per stable track ID, upserted every frame the track is observed. The
first-seen timestamp is preserved across updates, so full object histories can
be reconstructed from the database alone after a run. Schema matches the
`detection_tracking` branch's ObjectRegistry, so existing tooling/queries work.

Works directly with the detector's track dicts
(``{id, bbox:[x,y,w,h], area_px, first_seen_ns, last_seen_ns, n_observations}``),
so it carries no dependency on the hand-rolled tracker's Track/BBox types.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

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
        # check_same_thread=False is safe here: the detector loop is single
        # threaded; this just avoids SQLite rejecting the handle if a future
        # caller touches it from a callback thread.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert_all(self, tracks: Iterable[dict]) -> None:
        rows = [_row(t) for t in tracks]
        if not rows:
            return
        self._conn.executemany(_UPSERT, rows)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _row(t: dict) -> tuple[int, int, int, int, int, int, int, int, int]:
    x, y, w, h = t["bbox"]
    return (
        int(t["id"]),
        int(t["first_seen_ns"]),
        int(t["last_seen_ns"]),
        int(t["n_observations"]),
        int(x), int(y), int(w), int(h),
        int(t["area_px"]),
    )
