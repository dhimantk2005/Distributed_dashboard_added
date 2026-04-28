import subprocess
import sys

def separator(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

# ── 1. nvidia-smi ──────────────────────────────────
separator("1. NVIDIA-SMI")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("❌ nvidia-smi not found or no NVIDIA GPU detected")
except FileNotFoundError:
    print("❌ nvidia-smi not installed or not in PATH")

# ── 2. PyTorch ─────────────────────────────────────
separator("2. PyTorch GPU Check")
try:
    import torch
    print(f"PyTorch Version     : {torch.__version__}")
    print(f"CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version        : {torch.version.cuda}")
        print(f"GPU Count           : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}             : {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total VRAM          : {props.total_memory / 1024**3:.2f} GB")
        print(f"Allocated Memory    : {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Reserved Memory     : {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

        # Quick tensor test on GPU
        print("\n  🔥 Running quick GPU tensor test...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = x @ y
        print(f"  Tensor device       : {z.device} ✅")
    else:
        print("⚠️  CUDA not available — model will train on CPU")
except ImportError:
    print("❌ PyTorch not installed. Run: conda install pytorch -c pytorch")

# ── 3. TensorFlow ──────────────────────────────────
separator("3. TensorFlow GPU Check")
try:
    import tensorflow as tf
    print(f"TensorFlow Version  : {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs Available      : {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("⚠️  No GPU found by TensorFlow — will use CPU")
except ImportError:
    print("ℹ️  TensorFlow not installed (skip if using PyTorch)")
except ValueError as e:
    if "numpy.dtype size changed" in str(e):
        print("❌ TensorFlow NumPy ABI mismatch detected!")
        print(f"   Error: {e}")
        print("   Fix: Run: pip install --upgrade 'numpy<2' or reinstall TensorFlow")
    else:
        raise

# ── 4. CUDA / cuDNN info ───────────────────────────
separator("4. CUDA / cuDNN Info")
try:
    import torch
    if torch.cuda.is_available():
        print(f"cuDNN Enabled       : {torch.backends.cudnn.enabled}")
        print(f"cuDNN Version       : {torch.backends.cudnn.version()}")
    else:
        print("⚠️  CUDA not available, skipping cuDNN check")
except Exception as e:
    print(f"❌ Error: {e}")

# ── 5. WSL Check ───────────────────────────────────
separator("5. WSL Environment Check")
try:
    with open("/proc/version", "r") as f:
        version_info = f.read()
    if "microsoft" in version_info.lower():
        print("✅ Running inside WSL")
        print(f"  {version_info.strip()}")
    else:
        print("ℹ️  Not running in WSL (native Linux)")
except Exception as e:
    print(f"❌ Could not read /proc/version: {e}")

# ── Summary ────────────────────────────────────────
separator("SUMMARY")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU is ACTIVE  →  {torch.cuda.get_device_name(0)}")
        print("   Your model will train on GPU 🚀")
    else:
        print("❌ GPU NOT available — training will run on CPU")
        print("\n  Possible fixes:")
        print("  1. Install CUDA toolkit  : conda install cudatoolkit")
        print("  2. Reinstall PyTorch     : conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("  3. Check WSL2 drivers    : NVIDIA drivers must be installed on Windows host")
        print("  4. Confirm WSL2 (not 1)  : run 'wsl --version' in Windows terminal")
except ImportError:
    print("⚠️  PyTorch not found — install it to use GPU")

print()