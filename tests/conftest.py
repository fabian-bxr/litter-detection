"""In-memory Bridge implementation for mission-component tests (no zenoh)."""

import asyncio

import pytest
from pydantic import BaseModel


class MemoryBridge:
    """Implements litter_agents.zenoh_bridge.Bridge without a network.

    ``push(key, value)`` delivers an already-decoded value to subscribers
    (the decode function is bypassed); published models are recorded in
    ``published`` and also handed to any ``on_publish`` hook so tests can
    script the robot's reactions.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self.published: list[tuple[str, BaseModel]] = []
        self.on_publish = None

    def subscribe(self, key, decode, handler) -> None:
        self._handlers.setdefault(key, []).append(handler)

    def subscribe_latest(self, key, decode):
        from litter_agents.zenoh_bridge import LatestValue

        holder = LatestValue()
        self.subscribe(key, decode, holder._set)
        return holder

    def subscribe_queue(self, key, decode, maxsize: int = 32) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

        def put(value):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(value)

        self.subscribe(key, decode, put)
        return queue

    def publish_json(self, key: str, model: BaseModel) -> None:
        self.published.append((key, model))
        if self.on_publish:
            self.on_publish(key, model)

    def push(self, key: str, value) -> None:
        for handler in self._handlers.get(key, []):
            handler(value)


@pytest.fixture
def bridge() -> MemoryBridge:
    return MemoryBridge()
