Benchmark Workflow
==================

Overview
--------
This folder provides scripts to run benchmarks and aggregate metrics across:
- single GPU
- Tailscale distributed
- Ethernet distributed
- batch size sweep on Ethernet

All training scripts already write METRIC json lines to a metrics file.


1) Single GPU
------------
Example (CNN):
  python benchmarks/run_benchmarks.py single \
    --script train_single_gpu.py --trials 3 --epochs 1 --batch-size 64

Example (ResNet):
  python benchmarks/run_benchmarks.py single \
    --script train_single_gpu_resnet.py --trials 3 --epochs 1 --batch-size 64


2) Distributed Tailscale
------------------------
Run on each node with the appropriate rank. Use the same master address/port.

Rank 0:
  python benchmarks/run_benchmarks.py distributed \
    --script train_resnet.py --rank 0 --world-size 3 --master-addr 192.168.1.40 \
    --master-port 29500 --ifname tailscale0 --backend nccl --trials 3 --epochs 1 --batch-size 64

Rank 1 (example):
  python benchmarks/run_benchmarks.py distributed \
    --script train_resnet.py --rank 1 --world-size 3 --master-addr 192.168.1.40 \
    --master-port 29500 --ifname tailscale0 --backend nccl --trials 3 --epochs 1 --batch-size 64


3) Distributed Ethernet
-----------------------
Same as above, but set --ifname to your Ethernet interface (e.g. eth2).


4) Batch size sweep (Ethernet only)
-----------------------------------
Run on each node with the same ranks and settings:
  python benchmarks/run_benchmarks.py batchsize \
    --script train_resnet.py --rank 0 --world-size 3 --master-addr 192.168.1.40 \
    --master-port 29500 --ifname eth2 --backend nccl --epochs 1 --batch-sizes 32,64,128,256


5) Aggregate metrics
--------------------
Aggregate rank0 metrics files into a CSV summary:
  python benchmarks/aggregate_metrics.py \
    --metrics-glob "benchmarks/metrics/*rank0*.jsonl" \
    --out benchmarks/summary.csv


6) Plot batch size graph (Ethernet)
-----------------------------------
  python benchmarks/plot_batchsize.py \
    --csv benchmarks/summary.csv --out benchmarks/batchsize.png


Notes
-----
- Use unique metrics files per rank to avoid collisions (the runner does this).
- For distributed comparisons, only use rank 0 epoch metrics for summaries.
- If you want network stats, run ping/iperf3 separately and add to your report.
