"""Thread-safe Zenoh ↔ asyncio bridge.

Zenoh callbacks fire on Zenoh's internal threads — never call asyncio from
them directly.  All deliveries go through loop.call_soon_threadsafe so that
consumer coroutines work with ordinary `await queue.get()`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import zenoh
from pydantic import BaseModel


class AsyncZenoh:
    """Thin asyncio wrapper around a Zenoh session.

    Usage::

        bridge = AsyncZenoh(loop, "tcp/127.0.0.1:7447")
        q = bridge.subscribe_latest("robodog/localization/pose")
        raw = await q.get()
        ...
        await bridge.close()
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, endpoint: str) -> None:
        self._loop = loop
        self._subs: list[zenoh.Subscriber] = []
        self._pubs: dict[str, zenoh.Publisher] = {}

        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"client"')
        cfg.insert_json5("connect/endpoints", f'["{endpoint}"]')
        self._session = zenoh.open(cfg)

    # ------------------------------------------------------------------
    # Subscribe helpers
    # ------------------------------------------------------------------

    def _safe_schedule(self, fn: "Callable[[], None]") -> None:
        """Schedule fn on the event loop; silently drops if the loop is closed."""
        try:
            self._loop.call_soon_threadsafe(fn)
        except RuntimeError:
            pass  # loop already closed during shutdown — discard message

    def subscribe_latest(self, key: str) -> asyncio.Queue[bytes]:
        """Single-slot queue — only the newest message is kept."""
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

        def _cb(sample: zenoh.Sample) -> None:
            payload = bytes(sample.payload)

            def _put() -> None:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                q.put_nowait(payload)

            self._safe_schedule(_put)

        self._subs.append(self._session.declare_subscriber(key, _cb))
        return q

    def subscribe_queue(self, key: str, maxsize: int = 200) -> asyncio.Queue[bytes]:
        """FIFO queue — messages are not dropped (old ones block if full)."""
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=maxsize)

        def _cb(sample: zenoh.Sample) -> None:
            payload = bytes(sample.payload)
            self._safe_schedule(lambda: q.put_nowait(payload))

        self._subs.append(self._session.declare_subscriber(key, _cb))
        return q

    def subscribe_with_attachment(
        self, key: str
    ) -> asyncio.Queue[tuple[bytes, bytes | None]]:
        """Single-slot queue that also captures the sample attachment (e.g. timestamp)."""
        q: asyncio.Queue[tuple[bytes, bytes | None]] = asyncio.Queue(maxsize=1)

        def _cb(sample: zenoh.Sample) -> None:
            payload = bytes(sample.payload)
            att = bytes(sample.attachment) if sample.attachment is not None else None

            def _put() -> None:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                q.put_nowait((payload, att))

            self._safe_schedule(_put)

        self._subs.append(self._session.declare_subscriber(key, _cb))
        return q

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    def _publisher(self, key: str) -> zenoh.Publisher:
        if key not in self._pubs:
            self._pubs[key] = self._session.declare_publisher(
                key, encoding=zenoh.Encoding.APPLICATION_JSON
            )
        return self._pubs[key]

    def publish_json(self, key: str, msg: BaseModel) -> None:
        """Publish a Pydantic model as JSON. Safe to call from the asyncio thread."""
        self._publisher(key).put(msg.model_dump_json().encode())

    def publish_bytes(self, key: str, data: bytes, encoding: Any = None) -> None:
        enc = encoding or zenoh.Encoding.APPLICATION_OCTET_STREAM
        if key not in self._pubs:
            self._pubs[key] = self._session.declare_publisher(key, encoding=enc)
        self._pubs[key].put(data)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        for sub in self._subs:
            sub.undeclare()
        for pub in self._pubs.values():
            pub.undeclare()
        self._session.close()
