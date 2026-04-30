"""
Runs gpu_test.py in a subprocess and returns parsed GPU/system diagnostics.
"""

import subprocess
import sys
import re
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_diagnostics() -> dict:
    """
    Execute gpu_test.py and parse its stdout into a structured dict.
    Safe to call from a thread-pool executor (blocking I/O only).
    """
    gpu_test = REPO_ROOT / "gpu_test.py"

    if not gpu_test.exists():
        return _error_result("gpu_test.py not found in repository root")

    try:
        proc = subprocess.run(
            [sys.executable, str(gpu_test)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return _error_result("Diagnostics timed out after 60s")
    except Exception as exc:
        return _error_result(str(exc))

    parsed = _parse_output(output)
    parsed["timestamp"] = int(time.time() * 1000)
    parsed["raw_output"] = output[:10_000]    # cap at 10 KB
    return parsed


# ── Parsers ───────────────────────────────────────────────────────────

def _error_result(reason: str) -> dict:
    return {
        "timestamp": int(time.time() * 1000),
        "error": reason,
        "raw_output": "",
        "torch_cuda": False,
        "cuda_version": "N/A",
        "torch_version": "N/A",
        "gpu_count": 0,
        "gpu_names": [],
        "cudnn_enabled": False,
        "cudnn_version": "N/A",
        "is_wsl": False,
        "nvidia_smi_available": False,
    }


def _parse_output(output: str) -> dict:
    result = {
        "error": None,
        "torch_cuda": False,
        "cuda_version": "N/A",
        "torch_version": "N/A",
        "gpu_count": 0,
        "gpu_names": [],
        "cudnn_enabled": False,
        "cudnn_version": "N/A",
        "is_wsl": False,
        "nvidia_smi_available": False,
    }

    lower = output.lower()

    # ── CUDA availability ─────────────────────────────────────────────
    if re.search(r'cuda available\s*[:=]\s*true', lower):
        result["torch_cuda"] = True
    elif re.search(r'cuda is available', lower):
        result["torch_cuda"] = True

    # ── CUDA version ──────────────────────────────────────────────────
    m = re.search(r'cuda version\s*[:=]\s*(\S+)', output, re.IGNORECASE)
    if m:
        result["cuda_version"] = m.group(1).strip("'\"")

    # ── Torch version ─────────────────────────────────────────────────
    m = re.search(r'torch version\s*[:=]\s*(\S+)', output, re.IGNORECASE)
    if m:
        result["torch_version"] = m.group(1).strip("'\"")

    # ── GPU count ─────────────────────────────────────────────────────
    m = re.search(r'gpu count\s*[:=]\s*(\d+)', output, re.IGNORECASE)
    if m:
        result["gpu_count"] = int(m.group(1))

    # ── GPU names (various formats) ───────────────────────────────────
    names: list[str] = []
    # "GPU 0: NVIDIA RTX 4090" style
    names += re.findall(r'GPU\s+\d+\s*[:=]\s*([^\n\r(]+)', output)
    # Bare model names if above matched nothing
    if not names:
        names += re.findall(
            r'((?:GeForce|RTX|GTX|Tesla|Quadro|A\d{2,3}|H\d{2,3})\s+[\w\s\-]+?)(?:\n|$|\()',
            output,
        )
    result["gpu_names"] = [n.strip() for n in names[:8]]

    # ── cuDNN ─────────────────────────────────────────────────────────
    if re.search(r'cudnn.{0,10}enabled\s*[:=]\s*true', lower):
        result["cudnn_enabled"] = True
    m = re.search(r'cudnn version\s*[:=]\s*(\S+)', output, re.IGNORECASE)
    if m:
        result["cudnn_version"] = m.group(1).strip("'\"")
        result["cudnn_enabled"] = True

    # ── WSL detection ─────────────────────────────────────────────────
    if re.search(r'microsoft|wsl\d?|windows subsystem', lower):
        result["is_wsl"] = True

    # ── nvidia-smi ────────────────────────────────────────────────────
    if re.search(r'nvidia-smi', lower) and not re.search(r'not found|failed|error', lower[:300]):
        result["nvidia_smi_available"] = True

    return result
