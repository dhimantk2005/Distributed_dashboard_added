# Distributed Training Control Plane

This repository contains the backend API, frontend dashboard, and core machine learning scripts required to set up, monitor, and benchmark distributed model training across multiple GPU nodes.

---

## Prerequisites

Before setting up the project, ensure your system has the following installed:

- **Conda** (Miniforge/Miniconda/Anaconda) for managing the Python environment  
- **Node.js & npm** (via NVM is recommended) for the React frontend  
- **NVIDIA Drivers & CUDA Toolkit** configured (especially if running in WSL) for GPU support  

---

## Setup & Installation

### 1. Create the Conda Environment

```bash
conda create -n dist_train python=3.10 -y
conda activate dist_train
```

---

### 2. Install Dependencies

#### Core ML & Backend Dependencies

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

#### Frontend Dashboard Dependencies

```bash
cd dashboard
npm install
cd ..
```

---

## Running the Control Plane

### Start the Backend

```bash
conda activate dist_train
python backend/main.py
```

Alternative:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Start the Frontend

```bash
cd dashboard
npm run dev
```

Open: http://localhost:5173

---

## Running Distributed Training (`train.py`)

### Example Command

```bash
python train.py \
  --rank 1 \
  --world-size 3 \
  --master-addr 192.168.1.40 \
  --master-port 29500 \
  --backend nccl \
  --ifname tailscale0 \
  --init-timeout-min 10
```

---

## Distributed Training Parameters

| Parameter | Description | Example |
|----------|------------|--------|
| `--rank` | Tot Unique node ID | 1  |
| `--world-size` | Total processes | ≥ 1 |
| `--master-addr` | Master IP | 192.168.1.40 |
| `--master-port` | Port | 29500 |
| `--backend` | Backend | nccl/ gloo |
| `--ifname` | Interface | eth0/ tailscale0 |
| `--init-timeout-min` | Timeout | > 0 |

---

## Benchmarking

### Standard Benchmark

```bash
python benchmarks/run_benchmarks.py batchsize \
  --script train.py \
  --rank 0 \
  --world-size 2 \
  --master-addr 192.168.1.40 \
  --master-port 29500 \
  --ifname eth0 \
  --backend nccl \
  --epochs 1 \
  --batch-sizes 32,64,128,256,512,1024 \
  --trials 1
```

---

### OOM Test

```bash
python benchmarks/run_benchmarks.py batchsize \
  --script train.py \
  --rank 0 \
  --world-size 2 \
  --master-addr 192.168.1.40 \
  --master-port 29500 \
  --ifname eth0 \
  --backend nccl \
  --epochs 1 \
  --batch-sizes 10000 \
  --trials 1
```

---

## usual OOM results 

Large batch sizes exceed GPU VRAM, causing an Out of Memory (OOM) crash.
