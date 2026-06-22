import time
import threading

from .session import StreamSession


_STREAM_TTL = 120  # seconds before an orphaned session is reaped


class StreamManager:
    """Manages lifecycle of all active stream sessions."""

    def __init__(self):
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def create(self, metadata: dict) -> StreamSession:
        session = StreamSession(metadata)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> StreamSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def reap_expired(self):
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if now - s.created_at > _STREAM_TTL
            ]
            for sid in expired:
                self._sessions.pop(sid, None)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)
