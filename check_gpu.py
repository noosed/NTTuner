"""
GPU Detection Diagnostic Tool
Checks if PyTorch can detect your NVIDIA GPU
"""
import time 
print("=" * 60)
print("GPU Detection Diagnostic Tool")
print("=" * 60)

# Check 1: NVIDIA Driver
print("\n[1] Checking NVIDIA Driver...")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("[OK] NVIDIA driver detected")
        print("\nYour GPU(s):")
        lines = result.stdout.split('\n')
        for line in lines:
            if 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                print(f"  {line.strip()}")
    else:
        print("[ERROR] nvidia-smi failed - driver not installed?")
except FileNotFoundError:
    print("[ERROR] nvidia-smi not found - NVIDIA drivers not installed!")
    print("\nInstall NVIDIA drivers from: https://www.nvidia.com/download/index.aspx")
except Exception as e:
    print(f"[ERROR] {e}")

# Check 2: PyTorch Installation
print("\n[2] Checking PyTorch...")
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__} installed")
except ImportError:
    print("[ERROR] PyTorch not installed!")
    print("Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    exit(1)

# Check 3: CUDA Availability
print("\n[3] Checking CUDA Support...")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    print("\n" + "=" * 60)
    print("[OK] GPU DETECTED SUCCESSFULLY!")
    print("Your RTX 3080 is ready for training!")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("[ERROR] CUDA NOT AVAILABLE!")
    print("=" * 60)
    print("\nLikely cause: CPU-only PyTorch installed")
    print("\nFIX:")
    print("1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("\n2. Install CUDA-enabled PyTorch:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n3. Restart your terminal/IDE")
    print("\n4. Run this script again to verify")
    print("=" * 60)

# Check 4: Test GPU
if cuda_available:
    print("\n[4] Testing GPU...")
    try:
        x = torch.rand(5, 3).cuda()
        print("[OK] Successfully created tensor on GPU")
        print(f"Tensor device: {x.device}")
    except Exception as e:
        print(f"[ERROR] Failed to use GPU: {e}")

print("\nDiagnostics complete!")
time.sleep(1000)
