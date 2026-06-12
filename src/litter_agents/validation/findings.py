"""SQLite store for validated litter findings and mission records.

Same conventions as litter_detector.tracker.registry: stdlib sqlite3, schema
created on open, repo-root-relative default path. One row per (mission,
track); the UNIQUE constraint is the dedup primary key — tracker ids reset
across detector restarts, mission scoping absorbs that.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from litter_agents.interfaces.mission import LitterValidation, SearchAreaSpec
from litter_agents.interfaces.robodog import Pose2D

_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('validated', 'rejected', 'error')),
    category TEXT,
    confidence REAL,
    description TEXT,
    model_name TEXT,
    robot_x REAL,
    robot_y REAL,
    robot_theta REAL,
    bearing_rad REAL,
    bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER,
    area_px INTEGER,
    n_observations INTEGER,
    first_seen_ns INTEGER,
    last_seen_ns INTEGER,
    validated_at_ns INTEGER,
    image_path TEXT,
    context_image_path TEXT,
    raw_response TEXT,
    UNIQUE(mission_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_findings_mission ON findings(mission_id);

CREATE TABLE IF NOT EXISTS missions (
    mission_id TEXT PRIMARY KEY,
    prompt TEXT,
    area_spec_json TEXT,
    started_ns INTEGER,
    finished_ns INTEGER,
    coverage_fraction REAL,
    distance_m REAL,
    n_waypoints INTEGER,
    n_blocked INTEGER,
    report_json TEXT
);
"""


@dataclass(frozen=True)
class FindingRow:
    mission_id: str
    track_id: int
    status: str  # validated | rejected | error
    category: str | None
    confidence: float | None
    description: str | None
    robot_pose: Pose2D | None
    bearing_rad: float
    bbox: tuple[int, int, int, int]
    area_px: int
    n_observations: int
    first_seen_ns: int
    last_seen_ns: int
    validated_at_ns: int
    image_path: str | None
    context_image_path: str | None
    model_name: str | None = None
    raw_response: str | None = None


class FindingsRepository:
    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        # Single event loop writes; check_same_thread=False only keeps SQLite
        # from rejecting the handle if a shutdown hook touches it elsewhere.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── findings ────────────────────────────────────────────────────────────

    def insert_finding(self, row: FindingRow) -> bool:
        """Insert; returns False when this (mission, track) already exists."""
        pose = row.robot_pose
        try:
            self._conn.execute(
                """INSERT INTO findings (
                    mission_id, track_id, status, category, confidence,
                    description, model_name, robot_x, robot_y, robot_theta,
                    bearing_rad, bbox_x, bbox_y, bbox_w, bbox_h, area_px,
                    n_observations, first_seen_ns, last_seen_ns,
                    validated_at_ns, image_path, context_image_path, raw_response
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row.mission_id,
                    row.track_id,
                    row.status,
                    row.category,
                    row.confidence,
                    row.description,
                    row.model_name,
                    pose.x if pose else None,
                    pose.y if pose else None,
                    pose.theta if pose else None,
                    row.bearing_rad,
                    *row.bbox,
                    row.area_px,
                    row.n_observations,
                    row.first_seen_ns,
                    row.last_seen_ns,
                    row.validated_at_ns,
                    row.image_path,
                    row.context_image_path,
                    row.raw_response,
                ),
            )
        except sqlite3.IntegrityError:
            return False
        self._conn.commit()
        return True

    def processed_track_ids(self, mission_id: str) -> set[int]:
        rows = self._conn.execute(
            "SELECT track_id FROM findings WHERE mission_id = ?", (mission_id,)
        ).fetchall()
        return {int(r[0]) for r in rows}

    def findings(self, mission_id: str, status: str | None = None) -> list[FindingRow]:
        query = (
            "SELECT mission_id, track_id, status, category, confidence, description,"
            " model_name, robot_x, robot_y, robot_theta, bearing_rad,"
            " bbox_x, bbox_y, bbox_w, bbox_h, area_px, n_observations,"
            " first_seen_ns, last_seen_ns, validated_at_ns, image_path,"
            " context_image_path, raw_response FROM findings WHERE mission_id = ?"
        )
        args: list = [mission_id]
        if status is not None:
            query += " AND status = ?"
            args.append(status)
        rows = self._conn.execute(query + " ORDER BY track_id", args).fetchall()
        return [_finding_from_row(r) for r in rows]

    def status_counts(self, mission_id: str) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM findings WHERE mission_id = ? GROUP BY status",
            (mission_id,),
        ).fetchall()
        return {str(status): int(n) for status, n in rows}

    # ── missions ────────────────────────────────────────────────────────────

    def start_mission(
        self,
        mission_id: str,
        prompt: str,
        area_spec: SearchAreaSpec | None,
        started_ns: int,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO missions (mission_id, prompt, area_spec_json,"
            " started_ns) VALUES (?,?,?,?)",
            (
                mission_id,
                prompt,
                area_spec.model_dump_json() if area_spec else None,
                started_ns,
            ),
        )
        self._conn.commit()

    def finish_mission(
        self,
        mission_id: str,
        *,
        finished_ns: int,
        coverage_fraction: float,
        distance_m: float,
        n_waypoints: int,
        n_blocked: int,
        report_json: str,
    ) -> None:
        self._conn.execute(
            "UPDATE missions SET finished_ns=?, coverage_fraction=?, distance_m=?,"
            " n_waypoints=?, n_blocked=?, report_json=? WHERE mission_id=?",
            (
                finished_ns,
                coverage_fraction,
                distance_m,
                n_waypoints,
                n_blocked,
                report_json,
                mission_id,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _finding_from_row(r: tuple) -> FindingRow:
    pose = (
        Pose2D(x=r[7], y=r[8], theta=r[9])
        if r[7] is not None and r[8] is not None and r[9] is not None
        else None
    )
    return FindingRow(
        mission_id=str(r[0]),
        track_id=int(r[1]),
        status=str(r[2]),
        category=r[3],
        confidence=r[4],
        description=r[5],
        model_name=r[6],
        robot_pose=pose,
        bearing_rad=float(r[10]),
        bbox=(int(r[11]), int(r[12]), int(r[13]), int(r[14])),
        area_px=int(r[15]),
        n_observations=int(r[16]),
        first_seen_ns=int(r[17]),
        last_seen_ns=int(r[18]),
        validated_at_ns=int(r[19]),
        image_path=r[20],
        context_image_path=r[21],
        raw_response=r[22],
    )


def validation_to_raw(validation: LitterValidation) -> str:
    return json.dumps(validation.model_dump())
