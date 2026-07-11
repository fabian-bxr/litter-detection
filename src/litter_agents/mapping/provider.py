from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import httpx
import numpy as np
import yaml
from loguru import logger

from litter_agents.interfaces.robodog import OccupancyGrid
from litter_agents.mapping.grid import FREE, OCCUPIED, UNKNOWN, GridMap

if TYPE_CHECKING:
    from litter_agents.config import AgentSettings


class MapProvider(ABC):
    """Source of the static map.

    The file-based provider is the current default; the same interface lets a
    Zenoh- or REST-served map replace it without touching any consumer.
    """

    @abstractmethod
    async def load(self) -> GridMap: ...


def _grid_from_map_server(meta: dict, img: np.ndarray, *, source: str) -> GridMap:
    """Build a GridMap from ROS map_server metadata + a grayscale image.

    Shared by the file and MOLA-REST providers so both apply identical
    thresholding. Follows map_server semantics: occupancy probability
    p = (255 - v) / 255 (or v / 255 when ``negate``), occupied if
    p > occupied_thresh, free if p < free_thresh, else unknown. The PNG's top
    row is the map's max-y edge, so the image is flipped vertically to get row
    index increasing with +y.
    """
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    if len(origin) > 2 and abs(float(origin[2])) > 1e-9:
        raise NotImplementedError(f"map origin yaw {origin[2]} != 0 is not supported")
    mode = meta.get("mode", "trinary")
    if mode != "trinary":
        raise NotImplementedError(f"map mode {mode!r} not supported (trinary only)")

    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.196))
    if int(meta.get("negate", 0)):
        p_occ = img.astype(np.float32) / 255.0
    else:
        p_occ = (255.0 - img.astype(np.float32)) / 255.0

    occ = np.full(img.shape, UNKNOWN, dtype=np.int8)
    occ[p_occ > occupied_thresh] = OCCUPIED
    occ[p_occ < free_thresh] = FREE
    occ = np.flipud(occ).copy()

    grid = GridMap(
        occ=occ,
        resolution=float(meta["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
    )
    logger.info(
        "Loaded map {} ({}x{} cells @ {} m, {} free / {} occupied / {} unknown)",
        source,
        grid.width,
        grid.height,
        grid.resolution,
        int(grid.free_mask().sum()),
        int(grid.occupied_mask().sum()),
        int(grid.unknown_mask().sum()),
    )
    return grid


class MapServerProvider(MapProvider):
    """A map delivered as ROS map_server artifacts: YAML metadata + grayscale PNG.

    Subclasses only supply :meth:`fetch_map_server` (where the bytes come from);
    decoding and thresholding into a :class:`GridMap` is shared. Consumers that
    need the image itself (e.g. the web UI) can call ``fetch_map_server``
    directly instead of ``load``.
    """

    @abstractmethod
    async def fetch_map_server(self) -> tuple[dict, bytes]:
        """Return (parsed YAML metadata, PNG bytes) for the current map."""

    async def load(self) -> GridMap:
        meta, png_bytes = await self.fetch_map_server()
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("map image is not a decodable PNG")
        return _grid_from_map_server(meta, img, source=self._source_label(meta))

    def _source_label(self, meta: dict) -> str:
        return str(meta.get("image", "map"))


class FileMapProvider(MapServerProvider):
    """Loads a ROS map_server-style map off disk: grayscale PNG + YAML metadata.

    This is the format MOLA's mm2grid emits.
    """

    def __init__(self, yaml_path: str | Path) -> None:
        self.yaml_path = Path(yaml_path)

    async def fetch_map_server(self) -> tuple[dict, bytes]:
        meta = yaml.safe_load(self.yaml_path.read_text())
        image_path = Path(meta["image"])
        if not image_path.is_absolute():
            image_path = self.yaml_path.parent / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"could not read map image {image_path}")
        return meta, image_path.read_bytes()

    def _source_label(self, meta: dict) -> str:
        return Path(meta["image"]).name


class MolaMapProvider(MapServerProvider):
    """Static map served by the robodog-digipro MOLA SLAM control API.

    MOLA's ``build-grid`` turns a mapping session into a ROS map_server 2D
    costmap (a grayscale PNG + YAML). This provider fetches those artifacts over
    REST — the same format ``FileMapProvider`` reads off disk — from::

        GET {base_url}/maps/{session}/grid.yaml   # metadata
        GET {base_url}/maps/{session}/grid.png    # occupancy image

    When ``session`` is None the session MOLA is currently localizing against
    (``GET /status``) is used, so the map frame matches the robot pose; if
    nothing is localized it falls back to the most recently modified session.
    With ``build_if_missing`` the costmap is (re)built via ``POST .../build-grid``
    if it isn't there yet (e.g. right after a mapping run).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8088",
        session: str | None = None,
        *,
        build_if_missing: bool = False,
        build_params: dict | None = None,
        timeout_s: float = 30.0,
        build_timeout_s: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.build_if_missing = build_if_missing
        self.build_params = build_params or {}
        self.timeout_s = timeout_s
        self.build_timeout_s = build_timeout_s
        self._client = client

    async def fetch_map_server(self) -> tuple[dict, bytes]:
        if self._client is not None:
            return await self._fetch_with(self._client)
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout_s
        ) as client:
            return await self._fetch_with(client)

    async def _fetch_with(self, client: httpx.AsyncClient) -> tuple[dict, bytes]:
        session = self.session or await self._resolve_session(client)
        meta_text, png_bytes = await self._fetch_grid(client, session)
        meta = yaml.safe_load(meta_text)
        meta["_mola_session"] = session  # for the source label; ignored downstream
        return meta, png_bytes

    def _source_label(self, meta: dict) -> str:
        return f"mola:{meta.get('_mola_session')}/grid.png"

    async def _resolve_session(self, client: httpx.AsyncClient) -> str:
        """Pick the session to load when the caller didn't name one.

        Prefer the map MOLA is currently localizing against — the robot's pose
        is only valid in that map's frame, so any other session would render the
        wrong map. Fall back to the most recently modified session only when
        nothing is localized.
        """
        active = await self._active_session(client)
        if active:
            logger.info("MOLA map session from active localization: {}", active)
            return active
        latest = await self._latest_session(client)
        logger.warning(
            "MOLA is not localizing — falling back to newest session {!r}; its "
            "map frame may not match the robot pose. Set --mola-session to be "
            "explicit.",
            latest,
        )
        return latest

    async def _active_session(self, client: httpx.AsyncClient) -> str | None:
        """The session MOLA currently has open (localizing/mapping), if any.

        Best-effort: returns None if the control API has no /status route or
        isn't reachable, so session resolution can fall back to /maps.
        """
        try:
            status = (await self._get(client, "/status")).json()
        except httpx.HTTPError:
            return None
        session = status.get("session")
        return str(session) if session else None

    async def _latest_session(self, client: httpx.AsyncClient) -> str:
        maps = (await self._get(client, "/maps")).json()
        if not maps:
            raise RuntimeError(
                f"MOLA API at {self.base_url} reports no maps — run a mapping "
                "session (and build-grid) first, or set a session name"
            )
        latest = max(maps, key=lambda m: m.get("mtime", 0))
        return str(latest["session"])

    async def _fetch_grid(
        self, client: httpx.AsyncClient, session: str
    ) -> tuple[str, bytes]:
        try:
            return await self._get_grid(client, session)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404 and self.build_if_missing:
                logger.info(
                    "MOLA costmap missing for session {} — building it via "
                    "build-grid (this can take a while)",
                    session,
                )
                await self._build_grid(client, session)
                return await self._get_grid(client, session)
            raise

    async def _get_grid(
        self, client: httpx.AsyncClient, session: str
    ) -> tuple[str, bytes]:
        meta_text = (await self._get(client, f"/maps/{session}/grid.yaml")).text
        png = (await self._get(client, f"/maps/{session}/grid.png")).content
        return meta_text, png

    async def _build_grid(self, client: httpx.AsyncClient, session: str) -> None:
        body = {"floor_z": 0.0, "min_h": 0.1, "max_h": 1.5}
        body.update(self.build_params)
        resp = await client.post(
            f"/maps/{session}/build-grid", json=body, timeout=self.build_timeout_s
        )
        resp.raise_for_status()

    async def _get(self, client: httpx.AsyncClient, path: str) -> httpx.Response:
        resp = await client.get(path)
        resp.raise_for_status()
        return resp


class ZenohMapProvider(MapProvider):
    """Map served as a robodog OccupancyGrid message.

    Takes the fetch coroutine instead of a Zenoh session so the class stays
    transport-agnostic; the mission wiring supplies e.g. a one-shot subscriber
    on ``robodog/map/occupancy``.
    """

    def __init__(self, fetch: Callable[[], Awaitable[OccupancyGrid]]) -> None:
        self._fetch = fetch

    async def load(self) -> GridMap:
        return GridMap.from_occupancy_grid(await self._fetch())


def build_map_provider(settings: AgentSettings) -> MapProvider:
    """Construct the map provider selected by ``settings.map_source``.

    ``file`` and ``mola`` are self-contained. ``zenoh`` needs a live fetch
    coroutine (a Zenoh session), so it can't be built from settings alone —
    construct :class:`ZenohMapProvider` explicitly and pass it to the
    ``MissionController`` instead.
    """
    from litter_agents.config import repo_path

    src = settings.map_source
    if src == "file":
        # Relative YAML paths are repo-root-relative, not CWD-relative.
        return FileMapProvider(repo_path(settings.map_yaml_path))
    if src == "mola":
        return MolaMapProvider(
            settings.mola_api_url,
            session=settings.mola_map_session or None,
            build_if_missing=settings.mola_build_grid,
            build_params={
                "floor_z": settings.mola_grid_floor_z,
                "min_h": settings.mola_grid_min_h,
                "max_h": settings.mola_grid_max_h,
            },
        )
    if src == "zenoh":
        raise NotImplementedError(
            "map_source='zenoh' needs a live Zenoh fetch coroutine; construct "
            "ZenohMapProvider(fetch) directly and pass it to MissionController"
        )
    raise ValueError(f"unknown map_source: {src!r}")
