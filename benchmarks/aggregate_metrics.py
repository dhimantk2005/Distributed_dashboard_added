"""Aggregate METRIC jsonl files into summary stats.

Example:
  python benchmarks/aggregate_metrics.py --metrics-glob "benchmarks/metrics/*rank0*.jsonl" --out benchmarks/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path


def _read_epoch_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "epoch":
                rows.append(obj)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-glob", required=True)
    parser.add_argument("--out", default="benchmarks/summary.csv")
    args = parser.parse_args()

    paths = [Path(p) for p in glob.glob(args.metrics_glob)]
    if not paths:
        print("No metrics files found.")
        return 1

    rows: list[dict] = []
    for path in paths:
        epochs = _read_epoch_metrics(path)
        if not epochs:
            continue
        throughput = [float(e.get("throughput", 0.0)) for e in epochs]
        elapsed = [float(e.get("elapsed", 0.0)) for e in epochs]
        acc = [float(e.get("acc", 0.0)) for e in epochs]
        loss = [float(e.get("loss", 0.0)) for e in epochs]
        avg_batch = [float(e.get("avg_batch_time", 0.0)) for e in epochs]
        mem = [float(e.get("max_gpu_mem_mb", 0.0)) for e in epochs if e.get("max_gpu_mem_mb") is not None]

        rows.append({
            "file": str(path),
            "epochs": len(epochs),
            "throughput_mean": _mean(throughput),
            "throughput_std": _std(throughput),
            "elapsed_mean": _mean(elapsed),
            "elapsed_std": _std(elapsed),
            "acc_mean": _mean(acc),
            "acc_std": _std(acc),
            "loss_mean": _mean(loss),
            "loss_std": _std(loss),
            "avg_batch_time_mean": _mean(avg_batch),
            "avg_batch_time_std": _std(avg_batch),
            "max_gpu_mem_mb_mean": _mean(mem) if mem else 0.0,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} summaries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
