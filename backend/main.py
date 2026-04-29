"""
FastAPI control plane — entry point.

Start with:
    uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0

Endpoints:
    GET  /api/health
    GET  /api/cluster
    POST /api/jobs/start
    POST /api/jobs/stop
    GET  /api/jobs/status
    POST /api/diagnostics/run
    GET  /api/diagnostics/latest
    WS   /ws
"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .state import app_state
from .ws_manager import ws_manager
from .process_manager import start_job, stop_job
from .gpu_info import run_diagnostics as _run_diag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Register event loop for thread-safe WS broadcasting; cleanup on shutdown."""
    ws_manager.set_loop(asyncio.get_running_loop())
    logger.info("Control plane started — WebSocket hub ready")
    yield
    # Shutdown: terminate any running processes
    logger.info("Shutting down — stopping any running training jobs")
    stop_job()


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Distributed Training Control Plane",
    description="FastAPI backend bridging the React dashboard to train.py",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────

class TrainingConfigRequest(BaseModel):
    role: str = "master"
    rank: int = Field(default=0, ge=0)
    worldSize: int = Field(default=1, ge=1)
    masterAddr: str = "127.0.0.1"
    masterPort: str = "29500"
    backend: str = "gloo"
    epochs: int = Field(default=5, ge=1)
    batchSize: int = Field(default=64, ge=1)
    learningRate: float = Field(default=0.001, gt=0)
    initMethod: str = "env://"
    dataDir: str = "./data"
    saveDir: str = "./checkpoints"


# ── Health ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "timestamp": int(time.time() * 1000),
        "active_connections": len(ws_manager.connections),
        "is_running": app_state.is_running,
    }


# ── Cluster state ─────────────────────────────────────────────────────

@app.get("/api/cluster")
async def get_cluster():
    return app_state.get_snapshot()


# ── Jobs ──────────────────────────────────────────────────────────────

@app.post("/api/jobs/start")
async def start_training(config: TrainingConfigRequest):
    if app_state.is_running:
        return {
            "success": False,
            "error": "A training job is already running. Stop it first.",
        }
    try:
        job_id = start_job(config.model_dump())
        return {"success": True, "job_id": job_id}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@app.post("/api/jobs/stop")
async def stop_training():
    stop_job()
    return {"success": True}


@app.get("/api/jobs/status")
async def job_status():
    return {
        "is_running": app_state.is_running,
        "job_id": app_state.job_id,
        "node_count": len(app_state.nodes),
        "current_epoch": app_state.training.get("currentEpoch", 0),
    }


# ── Diagnostics ───────────────────────────────────────────────────────

@app.post("/api/diagnostics/run")
async def run_diagnostics():
    """Run gpu_test.py in a thread pool (blocking) and broadcast result."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_diag)
    app_state.diagnostics_result = result
    await ws_manager.broadcast({"type": "diagnostics_result", "result": result})
    return result


@app.get("/api/diagnostics/latest")
async def get_diagnostics():
    if app_state.diagnostics_result is None:
        return {"error": "No diagnostics have been run yet. POST /api/diagnostics/run first."}
    return app_state.diagnostics_result


# ── WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Push current state immediately so the client shows real data on connect
        snapshot = app_state.get_snapshot()
        await websocket.send_json({
            "type": "init",
            "nodes": snapshot["nodes"],
            "training": snapshot["training"],
            "logs": snapshot["logs"],
            "terminalLines": snapshot["terminalLines"],
            "timestamp": int(time.time() * 1000),
        })

        # Keep connection alive; handle pings and client disconnects
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Proactive heartbeat so load-balancers / proxies don't drop the connection
                await websocket.send_json(
                    {"type": "heartbeat", "timestamp": int(time.time() * 1000)}
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
