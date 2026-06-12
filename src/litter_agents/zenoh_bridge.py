"""Asyncio-friendly Zenoh wrapper.

Zenoh callbacks fire on zenoh runtime threads; everything here immediately
hands decoded values to the asyncio event loop via ``call_soon_threadsafe``,
so all consumer code runs single-threaded in the loop and needs no locks.
Decoding failures are logged and dropped — a malformed message must never
take down a subscriber.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

import zenoh
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T")


class Bridge(Protocol):
    """Pub/sub surface the mission components depend on.

    AsyncZenoh is the real implementation; tests substitute an in-memory one.
    """

    def subscribe(
        self,
        key: str,
        decode: Callable[[zenoh.Sample], T],
        handler: Callable[[T], None],
    ) -> None: ...

    def subscribe_latest(
        self, key: str, decode: Callable[[zenoh.Sample], T]
    ) -> "LatestValue[T]": ...

    def subscribe_queue(
        self, key: str, decode: Callable[[zenoh.Sample], T], maxsize: int = 32
    ) -> asyncio.Queue[T]: ...

    def publish_json(self, key: str, model: BaseModel) -> None: ...


class LatestValue(Generic[T]):
    """Single-slot holder: newest value wins. Mutated only on the event loop."""

    def __init__(self) -> None:
        self._value: T | None = None
        self._event = asyncio.Event()

    def _set(self, value: T) -> None:
        self._value = value
        self._event.set()

    @property
    def latest(self) -> T | None:
        return self._value

    async def wait_first(self, timeout: float) -> T:
        await asyncio.wait_for(self._event.wait(), timeout)
        assert self._value is not None
        return self._value


class AsyncZenoh:
    """Owns the Zenoh session and bridges its callbacks into the event loop."""

    def __init__(self, config: zenoh.Config) -> None:
        self._session = zenoh.open(config)
        self._loop = asyncio.get_running_loop()
        self._publishers: dict[str, zenoh.Publisher] = {}
        self._subscribers: list[zenoh.Subscriber] = []

    def subscribe(
        self,
        key: str,
        decode: Callable[[zenoh.Sample], T],
        handler: Callable[[T], None],
    ) -> None:
        """Decode on the zenoh thread (cheap), handle on the event loop."""

        def callback(sample: zenoh.Sample) -> None:
            try:
                value = decode(sample)
            except Exception:
                logger.opt(exception=True).warning("Failed to decode message on {}", key)
                return
            self._loop.call_soon_threadsafe(handler, value)

        # Keep a reference; a garbage-collected subscriber stops receiving.
        self._subscribers.append(self._session.declare_subscriber(key, callback))

    def subscribe_latest(
        self, key: str, decode: Callable[[zenoh.Sample], T]
    ) -> LatestValue[T]:
        holder: LatestValue[T] = LatestValue()
        self.subscribe(key, decode, holder._set)
        return holder

    def subscribe_queue(
        self, key: str, decode: Callable[[zenoh.Sample], T], maxsize: int = 32
    ) -> asyncio.Queue[T]:
        queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)

        def put(value: T) -> None:
            if queue.full():  # drop-oldest
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(value)

        self.subscribe(key, decode, put)
        return queue

    def publish_json(self, key: str, model: BaseModel) -> None:
        pub = self._publishers.get(key)
        if pub is None:
            pub = self._session.declare_publisher(
                key, encoding=zenoh.Encoding.APPLICATION_JSON
            )
            self._publishers[key] = pub
        pub.put(model.model_dump_json())

    def close(self) -> None:
        self._session.close()
