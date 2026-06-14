"""SQLite findings database.

Schema is created on first connect; all writes are synchronous (sqlite3)
executed via asyncio.to_thread so they don't block the event loop.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_DDL = """
CREATE TABLE IF NOT EXISTS findings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id       TEXT    NOT NULL,
    run_ts           TEXT    NOT NULL,
    track_id         INTEGER NOT NULL,
    confirmed        INTEGER NOT NULL DEFAULT 0,
    confidence       REAL,
    description      TEXT,
    category         TEXT,
    pose_x           REAL,
    pose_y           REAL,
    pose_theta       REAL,
    image_path       TEXT,
    validated_at     TEXT    NOT NULL
);
"""


@dataclass
class FindingRecord:
    mission_id: str
    run_ts: str
    track_id: int
    confirmed: bool
    confidence: float
    description: str
    category: str | None
    pose_x: float
    pose_y: float
    pose_theta: float
    image_path: str | None


def _sync_init(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(_DDL)
        con.commit()
    finally:
        con.close()


def _sync_insert(path: str, rec: FindingRecord) -> int:
    con = sqlite3.connect(path)
    try:
        cur = con.execute(
            """
            INSERT INTO findings
                (mission_id, run_ts, track_id, confirmed, confidence,
                 description, category, pose_x, pose_y, pose_theta,
                 image_path, validated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.mission_id,
                rec.run_ts,
                rec.track_id,
                int(rec.confirmed),
                rec.confidence,
                rec.description,
                rec.category,
                rec.pose_x,
                rec.pose_y,
                rec.pose_theta,
                rec.image_path,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        con.commit()
        return cur.lastrowid or 0
    finally:
        con.close()


def _sync_query_all(path: str, mission_id: str) -> list[dict]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "SELECT * FROM findings WHERE mission_id = ? ORDER BY id",
            (mission_id,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        con.close()


class FindingsDB:
    """Async wrapper around the sync SQLite helpers."""

    def __init__(self, db_path: Path) -> None:
        self._path = str(db_path)

    async def init(self) -> None:
        await asyncio.to_thread(_sync_init, self._path)

    async def insert(self, rec: FindingRecord) -> int:
        return await asyncio.to_thread(_sync_insert, self._path, rec)

    async def query_mission(self, mission_id: str) -> list[dict]:
        return await asyncio.to_thread(_sync_query_all, self._path, mission_id)
