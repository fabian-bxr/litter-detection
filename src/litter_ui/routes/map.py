from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from litter_agents.config import AgentSettings

router = APIRouter(prefix="/api/map")

_REPO_ROOT = Path(__file__).parents[3]  # src/litter_ui/routes -> src/litter_ui -> src -> repo root


def _resolve_map() -> tuple[Path, dict]:
    """Return (png_path, yaml_meta). Raises HTTPException on missing files."""
    yaml_path = Path(AgentSettings().map_yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = _REPO_ROOT / yaml_path
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="Map YAML not found")
    with yaml_path.open() as f:
        meta: dict = yaml.safe_load(f)
    png_path = Path(meta["image"])
    if not png_path.is_absolute():
        png_path = yaml_path.parent / png_path
    return png_path, meta


@router.get("/image")
def get_map_image() -> FileResponse:
    png_path, _ = _resolve_map()
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="Map image not found")
    return FileResponse(str(png_path), media_type="image/png")


@router.get("/config")
def get_map_config() -> dict:
    png_path, meta = _resolve_map()
    width_px: int | None = None
    height_px: int | None = None
    if png_path.exists():
        with Image.open(png_path) as img:
            width_px, height_px = img.size  # (width, height) in pixels

    origin: list[float] = meta.get("origin", [0.0, 0.0, 0.0])
    return {
        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "origin_theta": float(origin[2]),
        "resolution": float(meta.get("resolution", 0.05)),
        "width_px": width_px,
        "height_px": height_px,
    }
