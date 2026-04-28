#!/usr/bin/env python3
"""
launch.py — Interactive launcher for distributed training
Run this instead of train.py for a friendlier setup experience.

  python launch.py
"""

import subprocess
import sys
import socket
import os
import platform
import time


def detect_cuda_gpus():
    """Return (has_gpu, message) based on torch CUDA availability."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch not installed"

    if not torch.cuda.is_available():
        return False, "CUDA not available"

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    msg = f"Detected {len(names)} CUDA GPU(s): {', '.join(names)}"
    return True, msg


def check_tailscale():
    try:
        result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True)
        ip = result.stdout.strip()
        if ip:
            return ip
    except FileNotFoundError:
        pass
    return None


def check_dns_resolution(host):
    """Verify that hostname/IP can be resolved."""
    try:
        socket.gethostbyname(host)
        return True
    except socket.gaierror as e:
        print(f"  ✗ DNS resolution failed for '{host}': {e}")
        return False


def check_port_open(host, port, timeout=3):
    """Check if a port is reachable. Provides detailed error info."""
    try:
        s = socket.create_connection((host, int(port)), timeout=timeout)
        s.close()
        return True, None
    except socket.timeout:
        return False, f"Connection timeout after {timeout}s (port may be closed or firewall blocking)"
    except socket.gaierror as e:
        return False, f"Cannot resolve hostname '{host}': {e}"
    except ConnectionRefusedError:
        return False, f"Connection refused (master process not running or listening on port {port})"
    except PermissionError:
        return False, f"Permission denied (may need elevated privileges)"
    except OSError as e:
        return False, f"Network error: {e}"


def diagnose_connection_issue(host, port):
    """Provide detailed troubleshooting steps for connection failures."""
    print("\n⚠ CONNECTION TROUBLESHOOTING:")
    print("-" * 55)
    
    # 1. DNS check
    print(f"1. Verifying hostname resolution for '{host}'...")
    if not check_dns_resolution(host):
        print("   → Try pinging the master: `ping {}`".format(host))
        print("   → If using Tailscale, verify: `tailscale status`")
        return
    print(f"   ✔ Hostname resolves to {socket.gethostbyname(host)}")
    
    # 2. Port check with detailed output
    print(f"\n2. Checking if port {port} is reachable...")
    _, error_msg = check_port_open(host, port, timeout=5)
    if error_msg:
        print(f"   ✗ {error_msg}")
    
    # 3. OS-specific guidance
    print(f"\n3. Firewall & Network rules:")
    if platform.system() == "Windows":
        print(f"   • If on Windows, verify rule exists:")
        print(f"     netsh advfirewall firewall show rule name='PyTorch DDP'")
        print(f"   • List all inbound rules on port {port}:")
        print(f"     netsh advfirewall firewall show rule name=all | findstr {port}")
        print(f"   • Try creating a more permissive rule:")
        print(f"     netsh advfirewall firewall add rule name='PyTorch DDP Alt' dir=in action=allow protocol=TCP localport={port} remoteip=any")
        print(f"   • On WSL2, check if master.exe is accessible from Windows:")
        print(f"     wsl -d <distro> -u root -- netstat -tlnp | grep {port}")
    elif platform.system() == "Darwin":
        print(f"   • macOS: check System Preferences > Security & Privacy > Firewall")
        print(f"   • Or use: sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate")
    else:
        print(f"   • Linux: check firewall status")
        print(f"     sudo ufw status")
        print(f"     sudo iptables -L -n | grep {port}")
    
    # 4. Tailscale-specific
    print(f"\n4. Tailscale diagnostics:")
    print(f"   • Check Tailscale status: `tailscale status`")
    print(f"   • Verify both nodes are connected: `tailscale ping <other-node-ip>`")
    print(f"   • Check for split DNS or MagicDNS issues")
    
    # 5. Master process check
    print(f"\n5. Master process check:")
    print(f"   • Verify master is running: `python train.py --rank 0 ...` is still active")
    print(f"   • Check master's training.log or console output for errors")
    print(f"   • On master, verify it's listening: `netstat -tlnp | grep {port}`")
    print("-" * 55 + "\n")


def main():
    print("=" * 55)
    print("  Distributed Training Launcher")
    print("=" * 55)

    # Detect Tailscale IP
    my_ip = check_tailscale()
    if my_ip:
        print(f"\n✔ Tailscale detected. Your IP: {my_ip}")
    else:
        print("\n⚠ Tailscale not found or not running.")
        print("  Run `tailscale up` first, then retry.")
        my_ip = input("  Enter your IP manually: ").strip()

    print("\nAre you the MASTER (rank 0) or a WORKER?")
    role = input("  Enter 'master' or 'worker': ").strip().lower()

    world_size  = int(input("\nTotal number of machines (including master): ").strip())
    master_port = input("Port to use [default: 29500]: ").strip() or "29500"
    epochs      = input("Number of epochs [default: 5]: ").strip() or "5"
    batch_size  = input("Batch size per node [default: 64]: ").strip() or "64"

    gpu_ok, gpu_msg = detect_cuda_gpus()
    if gpu_ok:
        backend = "nccl"
        print(f"\n✔ GPU check: {gpu_msg}")
        print("   Using NCCL backend for GPU training.")
    else:
        print(f"\n✗ GPU check failed: {gpu_msg}")
        retry_gpu = input("Run on CPU instead? (yes/no) [default: no]: ").strip().lower()
        if retry_gpu not in ["yes", "y"]:
            print("Aborted. Set up CUDA GPUs and retry (or run gpu_test.py for diagnostics).")
            sys.exit(1)
        backend = "gloo"
        print("   Falling back to CPU with Gloo backend.")

    if role == "master":
        rank        = 0
        master_addr = my_ip
        print(f"\n✔ You are MASTER. Share this IP with workers: {master_addr}")
    else:
        rank        = int(input("\nYour rank (1, 2, 3 ...): ").strip())
        master_addr = input("Master node's Tailscale IP: ").strip()

        print(f"\nPre-flight check: verifying connection to master ({master_addr}:{master_port}) ...")
        can_connect, error_msg = check_port_open(master_addr, master_port, timeout=5)
        
        if can_connect:
            print(f"✔ Successfully connected to master on port {master_port}!")
        else:
            print(f"✗ {error_msg}")
            print("\n⚠ IMPORTANT: Master must be running BEFORE workers connect.")
            print("   Make sure you've started: python train.py --rank 0 --world-size X --master-addr {}")
            print("   Wait 10-15 seconds after starting master before launching workers.\n")
            
            diagnose_connection_issue(master_addr, master_port)
            
            retry = input("Try to connect anyway? (yes/no) [default: no]: ").strip().lower()
            if retry not in ["yes", "y"]:
                print("Aborted. Please fix the master connection and try again.")
                sys.exit(1)

    cmd = [
        sys.executable, "train.py",
        "--rank",        str(rank),
        "--world-size",  str(world_size),
        "--master-addr", master_addr,
        "--master-port", master_port,
        "--backend",    backend,
        "--epochs",      epochs,
        "--batch-size",  batch_size,
    ]

    print(f"\nRunning:\n  {' '.join(cmd)}\n")
    print("=" * 55)

    try:
        os.execv(sys.executable, cmd)
    except FileNotFoundError:
        print("✗ ERROR: train.py not found in current directory")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)
    except PermissionError:
        print("✗ ERROR: Permission denied when running train.py")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: Failed to launch training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()