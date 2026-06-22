import asyncio
import json
import socket
import threading
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .manager import StreamManager


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StreamSSEServer:
    """Single persistent SSE HTTP server for all active streams."""

    def __init__(self, manager: StreamManager):
        self.manager = manager
        self.port: int = 0
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _handle_sse(self, request):
        session_id = request.path_params["session_id"]
        session = self.manager.get(session_id)
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=404)

        session.state = "streaming"

        async def event_stream():
            yield f"event: metadata\ndata: {json.dumps(session.metadata)}\n\n"
            try:
                async for msg in session:
                    yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"
            except Exception:
                yield f"event: error\ndata: {json.dumps({'message': 'stream error'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _handle_health(self, request):
        return JSONResponse({"ok": True, "active_sessions": self.manager.active_count})

    def build_app(self) -> Starlette:
        return Starlette(routes=[
            Route("/stream/{session_id:str}", self._handle_sse),
            Route("/health", self._handle_health),
        ])

    def start(self):
        """Start uvicorn on a free port in a daemon thread."""
        if self._thread is not None:
            return

        self.port = _find_free_port()
        app = self.build_app()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            lifespan="off",
        )
        self._server = uvicorn.Server(config=config)

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
            self._thread = None
            self._server = None
