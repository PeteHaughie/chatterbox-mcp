import asyncio
import json
import time
import uuid
import threading
from collections import deque

SAMPLE_RATE = 24000


class StreamSession:
    """One generation stream — uses thread-safe deque + polling async iter."""

    def __init__(self, metadata: dict):
        self.id: str = uuid.uuid4().hex
        self.state: str = "pending"
        self.error: str | None = None
        self.created_at: float = time.time()
        self.metadata: dict = {
            "sample_rate": SAMPLE_RATE,
            "format": "raw_pcm_s16le",
            **metadata,
        }
        self._queue: deque = deque()

    def push(self, event: str, data: dict):
        self._queue.append({"event": event, "data": data})

    def end(self):
        self._queue.append(None)

    async def __aiter__(self):
        while True:
            while not self._queue:
                await asyncio.sleep(0.05)
            msg = self._queue.popleft()
            if msg is None:
                break
            yield msg
