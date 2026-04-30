"""
Distributed ML Training across laptops via Tailscale
=====================================================
Usage (run on EVERY machine):

  Master node (rank 0):
    python train.py --rank 0 --world-size 2 --master-addr <your-tailscale-ip>

  Worker node (rank 1):
    python train.py --rank 1 --world-size 2 --master-addr <master-tailscale-ip>

Tip: Find your Tailscale IP with `tailscale ip -4`
"""

import os
import argparse
import time
import sys
import json
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ─── Config ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Distributed MNIST training")
    p.add_argument("--rank",        type=int,   required=True,          help="Rank of this process (0 = master)")
    p.add_argument("--world-size",  type=int,   required=True,          help="Total number of machines/processes")
    p.add_argument("--master-addr", type=str,   required=True,          help="Tailscale IP of the master node")
    p.add_argument("--master-port", type=str,   default="29500",        help="Port to use (default: 29500)")
    p.add_argument("--backend",     type=str,   default="nccl",         help="Distributed backend: gloo (CPU) or nccl (GPU)")
    p.add_argument("--epochs",      type=int,   default=1,              help="Number of training epochs")
    p.add_argument("--batch-size",  type=int,   default=64,             help="Per-node batch size")
    p.add_argument("--lr",          type=float, default=1e-3,           help="Learning rate")
    p.add_argument("--data-dir",    type=str,   default="./data",       help="Where to download MNIST")
    p.add_argument("--save-dir",    type=str,   default="./checkpoints",help="Where to save model checkpoints")
    p.add_argument("--log-interval",type=int,   default=50,             help="Log every N batches")
    p.add_argument("--ifname", type=str, default="tailscale0",                  help="Network interface for distributed comms (e.g., tailscale0, eth0)")
    p.add_argument("--init-timeout-min", type=int, default=5,            help="Process-group init timeout in minutes")
    p.add_argument("--init-retries", type=int, default=8,                help="Worker retry attempts for transient init failures")
    p.add_argument("--retry-delay", type=float, default=3.0,             help="Seconds to wait between worker init retries")
    p.add_argument("--init-method", type=str, default="env://",         help="Init method: 'env://', 'tcp://' or 'file://<path>'")
    p.add_argument("--metrics-file", type=str, default="metrics.jsonl", help="Write METRIC lines to this file (per process)")
    return p.parse_args()


def emit_metric(metric: dict, metrics_file: str) -> None:
    line = json.dumps(metric)
    print(f"METRIC: {line}", flush=True)
    if metrics_file:
        os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
        with open(metrics_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ─── Setup / Teardown ────────────────────────────────────────────────────────

def setup(rank, world_size, master_addr, master_port, backend, timeout_min, init_retries, retry_delay, ifname, init_method):
    # Determine initialization method
    if init_method.startswith("file://"):
        store_file = init_method.replace("file://", "")
        os.makedirs(os.path.dirname(store_file) or ".", exist_ok=True)
        init_url = init_method
        print(f"[Rank {rank}] Using file-based store: {store_file}")
    elif init_method == "tcp://":
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        init_url = init_method
        print(f"[Rank {rank}] Using TCP store at {master_addr}:{master_port}")
    else:
        # Default env://
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        init_url = "env://"

    # Force a routable NIC for multi-machine runs so Gloo/NCCL do not pick loopback.
    if ifname and init_url != "file://":
        os.environ["GLOO_SOCKET_IFNAME"] = ifname
        os.environ["NCCL_SOCKET_IFNAME"] = ifname
        os.environ["TP_SOCKET_IFNAME"] = ifname
        print(f"[Rank {rank}] Using network interface: {ifname}")

    attempts = max(1, init_retries if rank != 0 else 1)
    timeout = timedelta(minutes=timeout_min)

    for attempt in range(1, attempts + 1):
        if init_url.startswith("file://"):
            print(f"[Rank {rank}] Waiting for rendezvous on file store (attempt {attempt}/{attempts}) ...")
        else:
            print(
                f"[Rank {rank}] Connecting to master at {master_addr}:{master_port} "
                f"(attempt {attempt}/{attempts}, timeout={timeout_min}m) ..."
            )
        try:
            dist.init_process_group(
                backend=backend,
                init_method=init_url,
                rank=rank,
                world_size=world_size,
                timeout=timeout,
            )
            print(f"[Rank {rank}] Connected! ({world_size} nodes total)")
            return
        except RuntimeError as e:
            err = str(e)
            err_lower = err.lower()
            transient = any(s in err_lower for s in [
                "connection refused",
                "timed out",
                "failed to recv, got 0 bytes",
                "connection reset",
                "connection closed",
                "broken pipe",
            ])

            if "Address already in use" in err or "Address in use" in err:
                print(f"[Rank {rank}] ERROR: Port {master_port} already in use!")
                print(f"         Try a different port or kill existing process")
                raise
            elif "Connection refused" in err or "refused" in err_lower:
                print(f"[Rank {rank}] ERROR: Connection refused by master {master_addr}:{master_port}")
                print(f"         - Is master running? (python train.py --rank 0 ...)")
                print(f"         - Correct --master-addr? Current: {master_addr}")
                print(f"         - Firewall blocking port {master_port}?")
                print(f"         - On Windows: netsh advfirewall firewall show rule name=all")
            elif "timed out" in err_lower:
                print(f"[Rank {rank}] ERROR: Connection timeout after {timeout_min} minutes")
                print(f"         - Network unreachable between {os.environ['MASTER_ADDR']} and this node")
                print(f"         - On Tailscale: run 'tailscale ping {master_addr}'")
                print(f"         - Check: tailscale status | both nodes connected?")
            elif "gloo" in err_lower or "backend" in err_lower:
                print(f"[Rank {rank}] ERROR: Backend initialization failed: {e}")
                print(f"         - Selected backend: {backend}")
                print(f"         - Try: --backend gloo (CPU) or --backend nccl (GPU)")
            else:
                print(f"[Rank {rank}] ERROR: Failed to initialize distributed process group:")
                print(f"         {e}")

            if rank != 0 and transient and attempt < attempts:
                print(f"[Rank {rank}] Retry in {retry_delay:.1f}s (master may still be starting or restarting)")
                time.sleep(retry_delay)
                continue
            raise
        except Exception as e:
            print(f"[Rank {rank}] UNEXPECTED ERROR during setup: {type(e).__name__}: {e}")
            raise


def cleanup():
    dist.destroy_process_group()


# ─── Model ───────────────────────────────────────────────────────────────────

class SimpleNet(nn.Module):
    """Small CNN for CIFAR-10 — swap this out for your own model."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, loss, save_dir, rank):
    """Only the master (rank 0) saves checkpoints."""
    if rank != 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save({
        "epoch":      epoch,
        "model":      model.module.state_dict(),   # .module unwraps DDP
        "optimizer":  optimizer.state_dict(),
        "loss":       loss,
    }, path)
    print(f"[Rank 0] Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ─── Training loop ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, sampler, optimizer, loss_fn, device, epoch_idx, rank, log_interval, metrics_file):
    model.train()
    sampler.set_epoch(epoch_idx)  # Ensures different shuffling per epoch across all nodes

    total_loss    = 0.0
    correct       = 0
    total         = 0
    last_throughput = 0.0
    batch_time_sum = 0.0
    batch_count = 0

    # Disable tqdm when stdout is not a terminal (e.g. launched via subprocess)
    headless = not sys.stderr.isatty()
    progress = tqdm(loader, desc=f"Rank {rank} Epoch {epoch_idx + 1}",
                    position=rank, leave=True, disable=headless)

    for batch_idx, (data, target) in enumerate(progress):
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_t0 = time.time()

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss   = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_elapsed   = time.time() - batch_t0
        last_throughput = data.size(0) / max(batch_elapsed, 1e-9)
        batch_time_sum += batch_elapsed
        batch_count += 1

        total_loss += loss.item()
        pred        = output.argmax(dim=1)
        correct    += pred.eq(target).sum().item()
        total      += target.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(loader):
            avg_loss = total_loss / (batch_idx + 1)
            acc      = 100.0 * correct / total
            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%")

            metric = {
                "type": "batch",
                "rank": rank,
                "epoch": epoch_idx + 1,
                "batch": batch_idx + 1,
                "total_batches": len(loader),
                "loss": round(avg_loss, 4),
                "acc": round(acc, 1),
                "throughput": round(last_throughput, 1),
            }
            emit_metric(metric, metrics_file)

    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total
    total_samples = total
    return avg_loss, avg_acc, total_samples, batch_time_sum, batch_count


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.world_size < 1:
        print("ERROR: --world-size must be >= 1")
        sys.exit(2)
    if args.rank < 0 or args.rank >= args.world_size:
        print(f"ERROR: --rank must be in [0, {args.world_size - 1}], got {args.rank}")
        sys.exit(2)
    if args.world_size == 1:
        print(f"[Rank 0] Single-node mode: using gloo backend, world_size=1")
        args.backend = "gloo"

    # Pick device — GPU if available and backend supports it, else CPU
    if args.backend == "nccl" and torch.cuda.is_available():
        device = torch.device("cuda", args.rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        if args.backend == "nccl":
            print(f"[Rank {args.rank}] WARNING: nccl requested but no CUDA found, falling back to gloo+cpu")
            args.backend = "gloo"

    print(f"[Rank {args.rank}] Using device: {device}")

    # Init distributed process group
    try:
        setup(
            args.rank,
            args.world_size,
            args.master_addr,
            args.master_port,
            args.backend,
            args.init_timeout_min,
            args.init_retries,
            args.retry_delay,
            args.ifname,
            args.init_method,
        )
    except Exception as e:
        print(f"[Rank {args.rank}] FATAL: Could not establish distributed training")
        print(f"[Rank {args.rank}] Cleanup and exit")
        sys.exit(1)

    try:
        # ── Data ──────────────────────────────────────────────────────────────────
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),  # CIFAR-10 mean/std
        ])

        print(f"[Rank {args.rank}] Loading CIFAR-10 dataset from {args.data_dir}...")
        dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        print(f"[Rank {args.rank}] Dataset loaded: {len(dataset)} samples")

        # DistributedSampler splits data across nodes so no node trains on the same samples
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=2,          # parallel data loading
            pin_memory=(device.type == "cuda"),
        )

        # ── Model ─────────────────────────────────────────────────────────────────
        model   = SimpleNet().to(device)
        if device.type == "cuda":
            model = DDP(model, device_ids=[device.index])
        else:
            model = DDP(model)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        # ── Training ──────────────────────────────────────────────────────────────
        if args.rank == 0:
            print(f"\n{'='*55}")
            print(f"  Distributed Training — {args.world_size} node(s)")
            print(f"  Epochs: {args.epochs} | Batch/node: {args.batch_size} | LR: {args.lr}")
            print(f"  Effective batch size: {args.batch_size * args.world_size}")
            print(f"{'='*55}\n")

        for epoch in range(args.epochs):
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            loss, acc, epoch_samples, batch_time_sum, batch_count = train_one_epoch(
                model, loader, sampler, optimizer, loss_fn,
                device, epoch, args.rank, args.log_interval, args.metrics_file
            )
            scheduler.step()

            global_loss = float(loss)
            global_acc = float(acc)
            global_samples = float(epoch_samples)
            global_batch_time = float(batch_time_sum)
            global_batch_count = float(batch_count)
            if dist.is_initialized():
                metrics = torch.tensor(
                    [loss, acc, epoch_samples, batch_time_sum, batch_count],
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                global_loss = metrics[0].item() / args.world_size
                global_acc = metrics[1].item() / args.world_size
                global_samples = metrics[2].item()
                global_batch_time = metrics[3].item()
                global_batch_count = metrics[4].item()

            global_throughput = (
                global_samples / max(global_batch_time, 1e-9)
                if global_samples > 0 else 0.0
            )
            avg_batch_time = (
                global_batch_time / max(global_batch_count, 1.0)
                if global_batch_count > 0 else 0.0
            )

            # Barrier: all nodes wait here before moving to next epoch
            try:
                dist.barrier()
            except RuntimeError as e:
                print(f"[Rank {args.rank}] ERROR at epoch {epoch+1} barrier: {e}")
                print(f"[Rank {args.rank}] One or more workers may have crashed. Aborting.")
                raise

            if args.rank == 0:
                elapsed = time.time() - t0
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] Loss: {global_loss:.4f} | "
                    f"Acc: {global_acc:.1f}% | Time: {elapsed:.1f}s\n"
                )
                max_gpu_mem_mb = None
                if device.type == "cuda":
                    max_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                metric = {
                    "type": "epoch",
                    "rank": 0,
                    "epoch": epoch + 1,
                    "total_epochs": args.epochs,
                    "loss": round(global_loss, 4),
                    "acc": round(global_acc, 1),
                    "elapsed": round(elapsed, 1),
                    "throughput": round(global_throughput, 1),
                    "avg_batch_time": round(avg_batch_time, 4),
                    "max_gpu_mem_mb": round(max_gpu_mem_mb, 1) if max_gpu_mem_mb is not None else None,
                }
                emit_metric(metric, args.metrics_file)
                save_checkpoint(model, optimizer, epoch + 1, global_loss, args.save_dir, args.rank)

        if args.rank == 0:
            print("Training complete!")

    except KeyboardInterrupt:
        if args.rank == 0:
            print("\n[Rank 0] Training interrupted by user")
    except Exception as e:
        print(f"[Rank {args.rank}] ERROR during training: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cleanup()
        except Exception as e:
            print(f"[Rank {args.rank}] Warning: Error during cleanup: {e}")


if __name__ == "__main__":
    main()