from __future__ import annotations

import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image

from litter_agents.config import AgentSettings
from litter_agents.mapping.provider import MapServerProvider, build_map_provider

router = APIRouter(prefix="/api/map")


async def _load_map_server() -> tuple[dict, bytes]:
    """Fetch (yaml_meta, png_bytes) for the configured map source.

    Honours ``AgentSettings.map_source`` — ``file`` reads the local map_server
    YAML+PNG, ``mola`` pulls the live costmap from the MOLA SLAM control API for
    the session the robot is localizing against (so the UI shows the map that
    matches the robot pose, not a stale local file).
    """
    settings = AgentSettings()
    provider = build_map_provider(settings)
    if not isinstance(provider, MapServerProvider):
        raise HTTPException(
            status_code=400,
            detail=f"map_source={settings.map_source!r} does not serve an image",
        )
    try:
        return await provider.fetch_map_server()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:  # network / API errors from the MOLA provider
        raise HTTPException(
            status_code=502, detail=f"map fetch failed ({settings.map_source}): {e}"
        ) from e


@router.get("/image")
async def get_map_image() -> Response:
    _, png_bytes = await _load_map_server()
    return Response(content=png_bytes, media_type="image/png")


@router.get("/coverage.png")
async def get_coverage_overlay() -> Response:
    """Latest exploration-coverage overlay, aligned to the static map image.

    A transparent RGBA PNG the frontend stacks on top of ``/api/map/image``.
    404 while no mission/sim is producing coverage. The frontend refetches this
    (cache-busted by the ``overlay_seq`` it receives on ``/ws/state``).
    """
    import litter_ui.zenoh_state as state

    if state.coverage_overlay_png is None:
        raise HTTPException(status_code=404, detail="no coverage overlay yet")
    return Response(
        content=state.coverage_overlay_png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/config")
async def get_map_config() -> dict:
    meta, png_bytes = await _load_map_server()
    with Image.open(io.BytesIO(png_bytes)) as img:
        width_px, height_px = img.size  # (width, height) in pixels

    origin: list[float] = meta.get("origin", [0.0, 0.0, 0.0])
    return {
        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "origin_theta": float(origin[2]) if len(origin) > 2 else 0.0,
        "resolution": float(meta.get("resolution", 0.05)),
        "width_px": width_px,
        "height_px": height_px,
    }
