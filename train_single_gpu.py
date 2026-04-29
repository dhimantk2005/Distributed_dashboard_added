"""
Single-GPU CIFAR-10 training (same metrics as distributed runs)
==============================================================
Usage:
  python train_single_gpu.py --epochs 1 --batch-size 64
"""

import os
import argparse
import time
import sys
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ─── Config ─────────────────────────────────────────────────────────────────-

def parse_args():
    p = argparse.ArgumentParser(description="Single-GPU CIFAR-10 training")
    p.add_argument("--epochs",      type=int,   default=1,              help="Number of training epochs")
    p.add_argument("--batch-size",  type=int,   default=64,             help="Batch size")
    p.add_argument("--lr",          type=float, default=1e-3,           help="Learning rate")
    p.add_argument("--data-dir",    type=str,   default="./data",       help="Where to download CIFAR-10")
    p.add_argument("--save-dir",    type=str,   default="./checkpoints",help="Where to save model checkpoints")
    p.add_argument("--log-interval",type=int,   default=50,             help="Log every N batches")
    return p.parse_args()


# ─── Model ─────────────────────────────────────────────────────────────────--

class SimpleNet(nn.Module):
    """Small CNN for CIFAR-10."""
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


# ─── Checkpoint helpers ─────────────────────────────────────────────────-----

def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "loss":       loss,
    }, path)
    print(f"[Single GPU] Checkpoint saved -> {path}")


# ─── Training loop ─────────────────────────────────────────────────----------

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch_idx, log_interval):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    last_throughput = 0.0
    batch_time_sum = 0.0
    batch_count = 0

    headless = not sys.stderr.isatty()
    progress = tqdm(loader, desc=f"Epoch {epoch_idx + 1}", disable=headless)

    for batch_idx, (data, target) in enumerate(progress):
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_t0 = time.time()

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_elapsed = time.time() - batch_t0
        last_throughput = data.size(0) / max(batch_elapsed, 1e-9)
        batch_time_sum += batch_elapsed
        batch_count += 1

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(loader):
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%")

            metric = {
                "type": "batch",
                "rank": 0,
                "epoch": epoch_idx + 1,
                "batch": batch_idx + 1,
                "total_batches": len(loader),
                "loss": round(avg_loss, 4),
                "acc": round(acc, 1),
                "throughput": round(last_throughput, 1),
            }
            print(f"METRIC: {json.dumps(metric)}", flush=True)

    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total
    total_samples = total
    return avg_loss, avg_acc, total_samples, batch_time_sum, batch_count


# ─── Main ─────────────────────────────────────────────────────────────────----

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Single GPU] Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    print(f"[Single GPU] Loading CIFAR-10 dataset from {args.data_dir}...")
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    print(f"[Single GPU] Dataset loaded: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = SimpleNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(args.epochs):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        loss, acc, epoch_samples, batch_time_sum, batch_count = train_one_epoch(
            model, loader, optimizer, loss_fn,
            device, epoch, args.log_interval
        )
        scheduler.step()

        global_throughput = (
            epoch_samples / max(batch_time_sum, 1e-9)
            if epoch_samples > 0 else 0.0
        )
        avg_batch_time = (
            batch_time_sum / max(batch_count, 1.0)
            if batch_count > 0 else 0.0
        )

        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch+1}/{args.epochs}] Loss: {loss:.4f} | "
            f"Acc: {acc:.1f}% | Time: {elapsed:.1f}s\n"
        )
        max_gpu_mem_mb = None
        if device.type == "cuda":
            max_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metric = {
            "type": "epoch",
            "rank": 0,
            "epoch": epoch + 1,
            "total_epochs": args.epochs,
            "loss": round(loss, 4),
            "acc": round(acc, 1),
            "elapsed": round(elapsed, 1),
            "throughput": round(global_throughput, 1),
            "avg_batch_time": round(avg_batch_time, 4),
            "max_gpu_mem_mb": round(max_gpu_mem_mb, 1) if max_gpu_mem_mb is not None else None,
        }
        print(f"METRIC: {json.dumps(metric)}", flush=True)
        save_checkpoint(model, optimizer, epoch + 1, loss, args.save_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
