"""
Shared in-memory state for the distributed training control plane.
Thread-safe singleton — accessed from both asyncio (FastAPI) and
background threads (subprocess stdout readers).
"""

import threading
import time
import socket
from typing import Dict, List, Optional, Any


MAX_LOGS = 500
MAX_TERMINAL_LINES = 1500


def _get_local_hostname() -> str:
    return socket.gethostname()


def _make_initial_node(rank: int = 0, world_size: int = 1) -> dict:
    return {
        "rank": rank,
        "ip": "127.0.0.1",
        "hostname": _get_local_hostname(),
        "status": "idle",
        "isMaster": rank == 0,
        "gpu": None,
        "backend": "gloo",
        "currentEpoch": 0,
        "totalEpochs": 0,
        "currentBatch": 0,
        "totalBatches": 0,
        "throughput": 0,
        "lastHeartbeat": int(time.time() * 1000),
        "retryCount": 0,
    }


def _make_initial_training() -> dict:
    return {
        "isRunning": False,
        "currentEpoch": 0,
        "totalEpochs": 0,
        "globalLoss": 0,
        "globalAccuracy": 0,
        "elapsedTime": 0,
        "epochHistory": [],
        "perNodeMetrics": [],
    }


class AppState:
    """Single source of truth for all runtime state."""

    def __init__(self):
        self._lock = threading.Lock()
        self.nodes: Dict[int, dict] = {}
        self.training: dict = _make_initial_training()
        self.logs: List[dict] = []
        self.terminal_lines: List[dict] = []
        self.processes: Dict[int, Any] = {}   # rank -> subprocess.Popen
        self.is_running: bool = False
        self.job_id: Optional[str] = None
        self.diagnostics_result: Optional[dict] = None
        self._log_counter: int = 0
        self._terminal_counter: int = 0
        self._job_start_time: float = 0

    # ── Logs ──────────────────────────────────────────────────────────

    def _next_log_id(self) -> str:
        """Must be called with lock held."""
        self._log_counter += 1
        return f"log-{self._log_counter}"

    def _next_terminal_id(self) -> str:
        """Must be called with lock held."""
        self._terminal_counter += 1
        return f"term-{self._terminal_counter}"

    def add_log(self, entry: dict) -> dict:
        """Add a log entry (assigns id), return the stored entry."""
        with self._lock:
            entry = dict(entry)
            entry["id"] = self._next_log_id()
            self.logs.append(entry)
            if len(self.logs) > MAX_LOGS:
                self.logs = self.logs[-MAX_LOGS:]
        return entry

    def add_terminal_line(self, entry: dict) -> dict:
        """Add a raw terminal line (assigns id), return the stored entry."""
        with self._lock:
            entry = dict(entry)
            entry["id"] = self._next_terminal_id()
            self.terminal_lines.append(entry)
            if len(self.terminal_lines) > MAX_TERMINAL_LINES:
                self.terminal_lines = self.terminal_lines[-MAX_TERMINAL_LINES:]
        return entry

    # ── Snapshots ─────────────────────────────────────────────────────

    def get_snapshot(self) -> dict:
        """Thread-safe atomic snapshot of current state."""
        with self._lock:
            return {
                "nodes": list(self.nodes.values()),
                "training": dict(self.training),
                "logs": list(self.logs),
                "terminalLines": list(self.terminal_lines),
                "isRunning": self.is_running,
            }

    # ── Job lifecycle ─────────────────────────────────────────────────

    def init_job(self, config: dict) -> None:
        """Reset state and prepare for a new training run."""
        with self._lock:
            self.is_running = True
            self._job_start_time = time.time()
            self.training = _make_initial_training()
            self.training["isRunning"] = True
            self.training["totalEpochs"] = config.get("epochs", 1)
            self.terminal_lines = []

            world_size = config.get("worldSize", 1)
            local_rank = config.get("rank", 0)
            backend = config.get("backend", "gloo")
            master_addr = config.get("masterAddr", "127.0.0.1")

            self.nodes = {}
            for r in range(world_size):
                node = _make_initial_node(r, world_size)
                node["backend"] = backend
                node["totalEpochs"] = config.get("epochs", 1)
                node["status"] = "training" if r == local_rank else "idle"
                node["ip"] = master_addr if r == 0 else "—"
                self.nodes[r] = node

            self.training["perNodeMetrics"] = [
                {
                    "rank": r,
                    "loss": 0,
                    "accuracy": 0,
                    "throughput": 0,
                    "batchesCompleted": 0,
                }
                for r in range(world_size)
            ]

    def update_batch_metric(
        self,
        rank: int,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        acc: float,
        throughput: float,
    ) -> None:
        with self._lock:
            if rank in self.nodes:
                self.nodes[rank]["currentEpoch"] = max(0, epoch - 1)
                self.nodes[rank]["currentBatch"] = batch
                self.nodes[rank]["totalBatches"] = total_batches
                self.nodes[rank]["throughput"] = round(throughput, 1)
                self.nodes[rank]["lastHeartbeat"] = int(time.time() * 1000)
                self.nodes[rank]["status"] = "training"

            for m in self.training["perNodeMetrics"]:
                if m["rank"] == rank:
                    m["loss"] = round(loss, 4)
                    m["accuracy"] = round(acc, 1)
                    m["throughput"] = round(throughput, 1)
                    m["batchesCompleted"] = batch
                    break

            metrics = self.training["perNodeMetrics"]
            if metrics:
                self.training["globalLoss"] = round(
                    sum(m["loss"] for m in metrics) / len(metrics), 4
                )
                self.training["globalAccuracy"] = round(
                    sum(m["accuracy"] for m in metrics) / len(metrics), 1
                )
            self.training["currentEpoch"] = max(0, epoch - 1)
            self.training["elapsedTime"] = time.time() - self._job_start_time

    def update_epoch_metric(
        self,
        rank: int,
        epoch: int,
        total_epochs: int,
        loss: float,
        acc: float,
        elapsed: float,
        throughput: float | None = None,
        avg_batch_time: float | None = None,
        max_gpu_mem_mb: float | None = None,
    ) -> None:
        with self._lock:
            if rank in self.nodes:
                self.nodes[rank]["currentEpoch"] = epoch
                self.nodes[rank]["currentBatch"] = self.nodes[rank].get("totalBatches", 0)
            self.training["currentEpoch"] = epoch
            self.training["globalLoss"] = round(loss, 4)
            self.training["globalAccuracy"] = round(acc, 1)
            self.training["elapsedTime"] = elapsed
            entry = {
                "epoch": epoch,
                "loss": round(loss, 4),
                "accuracy": round(acc, 1),
                "timestamp": int(time.time() * 1000),
            }
            if throughput is not None:
                entry["throughput"] = round(throughput, 1)
            if avg_batch_time is not None:
                entry["avgBatchTime"] = round(avg_batch_time, 4)
            if max_gpu_mem_mb is not None:
                entry["maxGpuMemMb"] = round(max_gpu_mem_mb, 1)
            self.training["epochHistory"].append(entry)

    def mark_node_status(self, rank: int, status: str) -> None:
        with self._lock:
            if rank in self.nodes:
                self.nodes[rank]["status"] = status

    def finish_job(self, success: bool = True) -> None:
        with self._lock:
            self.is_running = False
            self.training["isRunning"] = False
            for node in self.nodes.values():
                node["status"] = "idle" if success else "failed"

    def reset(self) -> None:
        with self._lock:
            self.nodes = {}
            self.training = _make_initial_training()
            self.logs = []
            self.terminal_lines = []
            self.is_running = False
            self.job_id = None


# ── Module-level singleton ────────────────────────────────────────────

app_state = AppState()
