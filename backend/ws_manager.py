"""
WebSocket connection manager.
Maintains the set of connected dashboard clients and broadcasts
JSON messages to all of them.

broadcast_sync() is safe to call from background threads via
run_coroutine_threadsafe once the event loop is registered.
"""

import json
import asyncio
import logging
from typing import Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the running asyncio event loop (called at startup)."""
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)
        logger.info("WebSocket client connected. Total: %d", len(self.connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(self.connections))

    async def broadcast(self, message: dict) -> None:
        """Send a JSON message to every connected client."""
        if not self.connections:
            return
        payload = json.dumps(message, default=str)
        dead: Set[WebSocket] = set()
        for ws in list(self.connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.connections.discard(ws)

    def broadcast_sync(self, message: dict) -> None:
        """
        Thread-safe broadcast — call this from non-async contexts
        (e.g. subprocess reader threads).
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)


# Module-level singleton
ws_manager = WebSocketManager()
