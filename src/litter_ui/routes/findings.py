from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from litter_agents.validation.findings import FindingRow, FindingsRepository
from litter_ui.deps import _REPO_ROOT, get_repo

router = APIRouter()


class FindingUpdate(BaseModel):
    category: str | None = None
    status: str | None = None


# ── Missions ─────────────────────────────────────────────────────────────────


@router.get("/missions")
def list_missions(repo: FindingsRepository = Depends(get_repo)) -> list[dict]:
    return repo.list_missions()


# ── Findings per mission ──────────────────────────────────────────────────────


@router.get("/missions/{mission_id}/findings")
def list_findings(
    mission_id: str,
    status: str | None = Query(default=None),
    repo: FindingsRepository = Depends(get_repo),
) -> list[dict]:
    return [_to_dict(f) for f in repo.findings(mission_id, status=status)]


@router.get("/missions/{mission_id}/findings/{track_id}")
def get_finding(
    mission_id: str,
    track_id: int,
    repo: FindingsRepository = Depends(get_repo),
) -> dict:
    finding = repo.get_finding(mission_id, track_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    return _to_dict(finding)


# ── CRUD on individual findings ───────────────────────────────────────────────


@router.delete("/findings/{mission_id}/{track_id}", status_code=204)
def delete_finding(
    mission_id: str,
    track_id: int,
    repo: FindingsRepository = Depends(get_repo),
) -> None:
    if not repo.delete_finding(mission_id, track_id):
        raise HTTPException(status_code=404, detail="Finding not found")


@router.patch("/findings/{mission_id}/{track_id}")
def update_finding(
    mission_id: str,
    track_id: int,
    body: FindingUpdate,
    repo: FindingsRepository = Depends(get_repo),
) -> dict:
    if body.category is None and body.status is None:
        raise HTTPException(status_code=422, detail="Provide at least one of: category, status")
    if not repo.update_finding(mission_id, track_id, category=body.category, status=body.status):
        raise HTTPException(status_code=404, detail="Finding not found")
    finding = repo.get_finding(mission_id, track_id)
    assert finding is not None
    return _to_dict(finding)


# ── Images ────────────────────────────────────────────────────────────────────


@router.get("/findings/{mission_id}/{track_id}/image")
def get_finding_image(
    mission_id: str,
    track_id: int,
    image_type: Annotated[str, Query(alias="type", pattern="^(crop|context)$")] = "crop",
    repo: FindingsRepository = Depends(get_repo),
) -> FileResponse:
    finding = repo.get_finding(mission_id, track_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    raw_path = finding.image_path if image_type == "crop" else finding.context_image_path
    if not raw_path:
        raise HTTPException(status_code=404, detail="Image not stored for this finding")
    path = Path(raw_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return FileResponse(str(path), media_type="image/jpeg")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_dict(f: FindingRow) -> dict:
    return {
        "mission_id": f.mission_id,
        "track_id": f.track_id,
        "status": f.status,
        "category": f.category,
        "confidence": f.confidence,
        "description": f.description,
        "model_name": f.model_name,
        "robot_x": f.robot_pose.x if f.robot_pose else None,
        "robot_y": f.robot_pose.y if f.robot_pose else None,
        "bearing_rad": f.bearing_rad,
        "bbox": list(f.bbox),
        "area_px": f.area_px,
        "n_observations": f.n_observations,
        "validated_at_ns": f.validated_at_ns,
        "image_path": f.image_path,
        "context_image_path": f.context_image_path,
    }
