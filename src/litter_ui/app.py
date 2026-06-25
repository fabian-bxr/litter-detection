from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import litter_ui.zenoh_state as zenoh_state
from litter_ui.routes import findings as findings_routes
from litter_ui.routes import map as map_routes
from litter_ui.routes import missions as missions_routes
from litter_ui.routes import ws as ws_routes

_REPO_ROOT = Path(__file__).parents[2]  # src/litter_ui -> src -> repo root


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await zenoh_state.startup()
    yield
    zenoh_state.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="Litter Detection UI", version="0.1.0", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(findings_routes.router, prefix="/api")
    app.include_router(map_routes.router)
    app.include_router(missions_routes.router, prefix="/api/mission")
    app.include_router(ws_routes.router)

    @app.get("/health", tags=["infra"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    ui_dist = _REPO_ROOT / "ui" / "dist"
    if ui_dist.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    return app


def main() -> None:
    uvicorn.run(create_app(), host="0.0.0.0", port=8080)
