"""
Subprocess lifecycle manager.

Responsibilities:
  - Launch train.py processes (one per local rank)
  - Read stdout in background threads
  - Parse log + metric lines via log_parser
  - Update AppState and broadcast via ws_manager
  - Kill processes on stop_job()
"""

import subprocess
import threading
import sys
import uuid
import time
import os
from pathlib import Path
from typing import Dict

from .state import app_state
from .log_parser import parse_line
from .ws_manager import ws_manager

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Stdout reader thread ──────────────────────────────────────────────

def _read_stdout(proc: subprocess.Popen, rank: int) -> None:
    """
    Runs in a daemon thread.
    Reads process stdout line-by-line, parses each line, updates state,
    and broadcasts to WebSocket clients.
    """
    try:
        for raw_line in iter(proc.stdout.readline, ""):
            if not raw_line:
                break

            log_entry, metric_data = parse_line(raw_line)

            # ── Structured log ─────────────────────────────────────
            if log_entry:
                stored = app_state.add_log(log_entry)
                ws_manager.broadcast_sync({
                    "type": "log",
                    "entry": stored,
                })

            # ── Metric update ──────────────────────────────────────
            if metric_data:
                mtype = metric_data.get("type")

                if mtype == "batch":
                    app_state.update_batch_metric(
                        rank=metric_data.get("rank", rank),
                        epoch=metric_data.get("epoch", 0),
                        batch=metric_data.get("batch", 0),
                        total_batches=metric_data.get("total_batches", 0),
                        loss=metric_data.get("loss", 0.0),
                        acc=metric_data.get("acc", 0.0),
                        throughput=metric_data.get("throughput", 0.0),
                    )
                elif mtype == "epoch":
                    app_state.update_epoch_metric(
                        rank=metric_data.get("rank", rank),
                        epoch=metric_data.get("epoch", 0),
                        total_epochs=metric_data.get("total_epochs", 0),
                        loss=metric_data.get("loss", 0.0),
                        acc=metric_data.get("acc", 0.0),
                        elapsed=metric_data.get("elapsed", 0.0),
                    )

                # Broadcast updated cluster + training state
                snapshot = app_state.get_snapshot()
                ws_manager.broadcast_sync({
                    "type": "cluster_state",
                    "nodes": snapshot["nodes"],
                    "training": snapshot["training"],
                })

    except Exception as exc:
        err_entry = app_state.add_log({
            "timestamp": int(time.time() * 1000),
            "rank": rank,
            "level": "error",
            "message": f"[Backend] Stdout reader error for rank {rank}: {exc}",
        })
        ws_manager.broadcast_sync({"type": "log", "entry": err_entry})

    finally:
        # ── Process finished ───────────────────────────────────────
        ret_code = proc.wait()
        success = ret_code == 0

        app_state.mark_node_status(rank, "idle" if success else "failed")

        finish_msg = (
            f"[Rank {rank}] Process exited (code {ret_code}) — Training complete! 🎉"
            if success else
            f"[Rank {rank}] Process exited with error code {ret_code}"
        )
        finish_entry = app_state.add_log({
            "timestamp": int(time.time() * 1000),
            "rank": rank,
            "level": "success" if success else "error",
            "message": finish_msg,
        })
        ws_manager.broadcast_sync({"type": "log", "entry": finish_entry})

        # Check if ALL tracked processes are done
        with app_state._lock:
            all_done = all(p.poll() is not None for p in app_state.processes.values())

        if all_done:
            app_state.finish_job(success=True)
            snapshot = app_state.get_snapshot()
            ws_manager.broadcast_sync({
                "type": "job_complete",
                "success": success,
                "nodes": snapshot["nodes"],
                "training": snapshot["training"],
            })


# ── Public API ────────────────────────────────────────────────────────

def start_job(config: dict) -> str:
    """
    Launch a training process for the given config.
    Returns a job_id UUID string.
    """
    # Stop any running job first
    stop_job()

    job_id = str(uuid.uuid4())
    rank = config["rank"]

    # Build command
    python = sys.executable
    script = str(REPO_ROOT / "train.py")

    cmd = [
        python, script,
        "--rank",        str(rank),
        "--world-size",  str(config.get("worldSize", 1)),
        "--master-addr", config.get("masterAddr", "127.0.0.1"),
        "--master-port", str(config.get("masterPort", "29500")),
        "--backend",     config.get("backend", "gloo"),
        "--epochs",      str(config.get("epochs", 5)),
        "--batch-size",  str(config.get("batchSize", 64)),
        "--lr",          str(config.get("learningRate", 0.001)),
        "--init-method", config.get("initMethod", "env://"),
        "--data-dir",    config.get("dataDir", "./data"),
        "--save-dir",    config.get("saveDir", "./checkpoints"),
    ]

    # Force unbuffered Python output so we get lines in real-time
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,          # line-buffered
            env=env,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        app_state.finish_job(success=False)
        raise RuntimeError(f"Failed to launch train.py: {exc}") from exc

    # Register state
    app_state.init_job(config)
    app_state.job_id = job_id
    with app_state._lock:
        app_state.processes[rank] = proc

    # Post startup log
    startup_entry = app_state.add_log({
        "timestamp": int(time.time() * 1000),
        "rank": rank,
        "level": "info",
        "message": (
            f"[Backend] Launched train.py — rank={rank}, "
            f"world_size={config.get('worldSize',1)}, "
            f"backend={config.get('backend','gloo')}, "
            f"epochs={config.get('epochs',5)}"
        ),
    })
    ws_manager.broadcast_sync({"type": "log", "entry": startup_entry})

    # Broadcast initial "training" cluster state
    snapshot = app_state.get_snapshot()
    ws_manager.broadcast_sync({
        "type": "cluster_state",
        "nodes": snapshot["nodes"],
        "training": snapshot["training"],
    })

    # Spawn stdout reader thread
    reader = threading.Thread(
        target=_read_stdout,
        args=(proc, rank),
        daemon=True,
        name=f"reader-rank-{rank}",
    )
    reader.start()

    return job_id


def stop_job() -> None:
    """Terminate all running training processes."""
    with app_state._lock:
        procs: Dict[int, subprocess.Popen] = dict(app_state.processes)

    for rank, proc in procs.items():
        if proc.poll() is None:          # still running
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass

    with app_state._lock:
        app_state.processes.clear()
        app_state.is_running = False
        app_state.training["isRunning"] = False
        for node in app_state.nodes.values():
            node["status"] = "idle"

    # Log and broadcast
    stop_entry = app_state.add_log({
        "timestamp": int(time.time() * 1000),
        "rank": 0,
        "level": "warning",
        "message": "[Rank 0] Training stopped by user",
    })
    ws_manager.broadcast_sync({"type": "log", "entry": stop_entry})

    snapshot = app_state.get_snapshot()
    ws_manager.broadcast_sync({
        "type": "cluster_state",
        "nodes": snapshot["nodes"],
        "training": snapshot["training"],
    })
