from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import litter_ui.zenoh_state as state

router = APIRouter()


@router.websocket("/ws/camera")
async def ws_camera(ws: WebSocket) -> None:
    await ws.accept()

    if state.camera_queue is None:
        await ws.send_text(json.dumps({"error": "zenoh_unavailable"}))
        await ws.close()
        return

    try:
        while True:
            try:
                frame: bytes = await asyncio.wait_for(
                    state.camera_queue.get(), timeout=5.0
                )
                await ws.send_bytes(frame)
            except asyncio.TimeoutError:
                # Keep connection alive while no frames arrive.
                await ws.send_text(json.dumps({"keepalive": True}))
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass


@router.websocket("/ws/state")
async def ws_state(ws: WebSocket) -> None:
    await ws.accept()

    if state.az is None:
        await ws.send_text(json.dumps({"error": "zenoh_unavailable"}))
        await ws.close()
        return

    q: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    state.state_subscribers.add(q)
    try:
        # Send whatever we already know so the client renders immediately.
        await ws.send_text(json.dumps(state.state_snapshot()))

        while True:
            msg = await q.get()
            await ws.send_text(msg)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        state.state_subscribers.discard(q)
