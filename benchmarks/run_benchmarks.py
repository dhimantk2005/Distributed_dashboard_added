"""Benchmark runner for single-GPU and distributed runs.

Examples:
  python benchmarks/run_benchmarks.py single \
    --script train_single_gpu.py --trials 3 --epochs 1 --batch-size 64

  python benchmarks/run_benchmarks.py distributed \
    --script train_resnet.py --rank 0 --world-size 3 --master-addr 192.168.1.40 \
    --master-port 29500 --ifname tailscale0 --backend nccl --trials 3 --epochs 1 --batch-size 64

  python benchmarks/run_benchmarks.py batchsize \
    --script train_resnet.py --rank 0 --world-size 3 --master-addr 192.168.1.40 \
    --master-port 29500 --ifname eth2 --backend nccl --epochs 1 --batch-sizes 32,64,128,256
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], env: dict[str, str] | None) -> int:
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _write_manifest(manifest_path: Path, record: dict) -> None:
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _build_common_args(args: argparse.Namespace) -> list[str]:
    return [
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--log-interval", str(args.log_interval),
    ]


def _build_metrics_file(metrics_dir: Path, label: str, trial: int, rank: int, batch_size: int) -> Path:
    name = f"{label}_trial{trial}_rank{rank}_bs{batch_size}.jsonl"
    return metrics_dir / name


def run_single(args: argparse.Namespace) -> int:
    metrics_dir = Path(args.metrics_dir)
    logs_dir = Path(args.logs_dir)
    _ensure_dir(metrics_dir)
    _ensure_dir(logs_dir)

    manifest_path = metrics_dir / "runs.jsonl"
    label = args.label or "single_gpu"

    for trial in range(1, args.trials + 1):
        metrics_file = _build_metrics_file(metrics_dir, label, trial, 0, args.batch_size)
        cmd = [
            "python", args.script,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--log-interval", str(args.log_interval),
            "--metrics-file", str(metrics_file),
        ]

        record = {
            "run_id": f"{label}_{_timestamp()}_trial{trial}",
            "label": label,
            "mode": "single",
            "trial": trial,
            "rank": 0,
            "world_size": 1,
            "batch_size": args.batch_size,
            "metrics_file": str(metrics_file),
            "command": cmd,
        }
        _write_manifest(manifest_path, record)

        code = _run(cmd, env=None)
        if code != 0:
            return code

    return 0


def run_distributed(args: argparse.Namespace) -> int:
    metrics_dir = Path(args.metrics_dir)
    logs_dir = Path(args.logs_dir)
    _ensure_dir(metrics_dir)
    _ensure_dir(logs_dir)

    manifest_path = metrics_dir / "runs.jsonl"
    label = args.label or f"dist_{args.ifname}"

    env = os.environ.copy()
    if args.ifname:
        env["NCCL_SOCKET_IFNAME"] = args.ifname
        env["GLOO_SOCKET_IFNAME"] = args.ifname

    for trial in range(1, args.trials + 1):
        metrics_file = _build_metrics_file(metrics_dir, label, trial, args.rank, args.batch_size)
        cmd = [
            "python", args.script,
            "--rank", str(args.rank),
            "--world-size", str(args.world_size),
            "--master-addr", args.master_addr,
            "--master-port", str(args.master_port),
            "--backend", args.backend,
            "--ifname", args.ifname,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--log-interval", str(args.log_interval),
            "--metrics-file", str(metrics_file),
        ]

        record = {
            "run_id": f"{label}_{_timestamp()}_trial{trial}",
            "label": label,
            "mode": "distributed",
            "trial": trial,
            "rank": args.rank,
            "world_size": args.world_size,
            "batch_size": args.batch_size,
            "ifname": args.ifname,
            "metrics_file": str(metrics_file),
            "command": cmd,
        }
        _write_manifest(manifest_path, record)

        code = _run(cmd, env=env)
        if code != 0:
            return code

    return 0


def run_batchsize(args: argparse.Namespace) -> int:
    metrics_dir = Path(args.metrics_dir)
    logs_dir = Path(args.logs_dir)
    _ensure_dir(metrics_dir)
    _ensure_dir(logs_dir)

    manifest_path = metrics_dir / "runs.jsonl"
    label = args.label or f"batchsize_{args.ifname}"

    env = os.environ.copy()
    if args.ifname:
        env["NCCL_SOCKET_IFNAME"] = args.ifname
        env["GLOO_SOCKET_IFNAME"] = args.ifname

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    for batch_size in batch_sizes:
        for trial in range(1, args.trials + 1):
            metrics_file = _build_metrics_file(metrics_dir, label, trial, args.rank, batch_size)
            cmd = [
                "python", args.script,
                "--rank", str(args.rank),
                "--world-size", str(args.world_size),
                "--master-addr", args.master_addr,
                "--master-port", str(args.master_port),
                "--backend", args.backend,
                "--ifname", args.ifname,
                "--epochs", str(args.epochs),
                "--batch-size", str(batch_size),
                "--lr", str(args.lr),
                "--log-interval", str(args.log_interval),
                "--metrics-file", str(metrics_file),
            ]

            record = {
                "run_id": f"{label}_{_timestamp()}_trial{trial}_bs{batch_size}",
                "label": label,
                "mode": "batchsize",
                "trial": trial,
                "rank": args.rank,
                "world_size": args.world_size,
                "batch_size": batch_size,
                "ifname": args.ifname,
                "metrics_file": str(metrics_file),
                "command": cmd,
            }
            _write_manifest(manifest_path, record)

            code = _run(cmd, env=env)
            if code != 0:
                return code

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--script", required=True, help="Training script to run")
    base.add_argument("--epochs", type=int, default=1)
    base.add_argument("--batch-size", type=int, default=64)
    base.add_argument("--lr", type=float, default=1e-3)
    base.add_argument("--log-interval", type=int, default=50)
    base.add_argument("--trials", type=int, default=3)
    base.add_argument("--metrics-dir", default="benchmarks/metrics")
    base.add_argument("--logs-dir", default="benchmarks/logs")
    base.add_argument("--label", default=None)

    dist = argparse.ArgumentParser(add_help=False, parents=[base])
    dist.add_argument("--rank", type=int, required=True)
    dist.add_argument("--world-size", type=int, required=True)
    dist.add_argument("--master-addr", required=True)
    dist.add_argument("--master-port", type=int, default=29500)
    dist.add_argument("--backend", default="nccl")
    dist.add_argument("--ifname", required=True)

    single = sub.add_parser("single", parents=[base])
    single.set_defaults(func=run_single)

    distributed = sub.add_parser("distributed", parents=[dist])
    distributed.set_defaults(func=run_distributed)

    batchsize = sub.add_parser("batchsize", parents=[dist])
    batchsize.add_argument("--batch-sizes", required=True, help="Comma-separated list, e.g. 32,64,128")
    batchsize.set_defaults(func=run_batchsize)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
