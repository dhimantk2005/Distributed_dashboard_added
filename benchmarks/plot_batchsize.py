"""Plot batch size vs throughput from a CSV summary.

Example:
  python benchmarks/plot_batchsize.py --csv benchmarks/summary.csv --out benchmarks/batchsize.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="benchmarks/batchsize.png")
    parser.add_argument("--label", default="Ethernet")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib is required for plotting.")
        print(exc)
        return 1

    batch_sizes = []
    throughputs = []

    with open(args.csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_path = row.get("file", "")
            # Infer batch size from filename pattern: *_bs<value>.jsonl
            bs = None
            if "_bs" in file_path:
                try:
                    tail = file_path.split("_bs", 1)[1]
                    bs = int(tail.split(".jsonl")[0])
                except Exception:
                    bs = None
            if bs is None:
                continue
            batch_sizes.append(bs)
            throughputs.append(float(row.get("throughput_mean", 0.0)))

    if not batch_sizes:
        print("No batch sizes found in CSV.")
        return 1

    pairs = sorted(zip(batch_sizes, throughputs), key=lambda x: x[0])
    batch_sizes = [p[0] for p in pairs]
    throughputs = [p[1] for p in pairs]

    plt.figure(figsize=(7, 4))
    plt.plot(batch_sizes, throughputs, marker="o", label=args.label)
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (samples/s)")
    plt.title("Batch size vs throughput")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Wrote plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
