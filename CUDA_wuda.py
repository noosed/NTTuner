"""
Fix for PyTorch CUDA DLL Error (WinError 127)
This error means PyTorch's CUDA libraries can't find required Windows dependencies
"""
import subprocess
import sys

print("=" * 70)
print("PyTorch CUDA DLL Error - FIX SCRIPT")
print("=" * 70)
print()
print("Error: c10_cuda.dll cannot find required Windows dependencies")
print()

print("SOLUTION 1: Reinstall PyTorch (Recommended)")
print("-" * 70)
print()
print("Your current PyTorch installation has CUDA libraries that can't find")
print("the necessary Windows runtime dependencies.")
print()
print("Fix by reinstalling PyTorch with proper CUDA support:")
print()
print("Step 1: Uninstall current PyTorch")
print("  python -m pip uninstall torch torchvision torchaudio -y")
print()
print("Step 2: Install CPU-only PyTorch (will work immediately)")
print("  python -m pip install torch torchvision torchaudio")
print()
print("OR install CUDA 12.1 version (for GPU, requires CUDA runtime):")
print("  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print()
print()

print("SOLUTION 2: Install Missing Windows Dependencies")
print("-" * 70)
print()
print("The CUDA DLL needs Visual C++ Redistributables:")
print()
print("1. Download and install from Microsoft:")
print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
print()
print("2. Restart your computer")
print()
print("3. Try running the program again")
print()
print()

print("QUICK FIX: Use CPU-Only Version")
print("-" * 70)
print()
print("To get the program running RIGHT NOW (training will be slow):")
print()

response = input("Do you want to automatically fix this now? (y/n): ").lower()

if response == 'y':
    print()
    print("=" * 70)
    print("FIXING: Installing CPU-only PyTorch...")
    print("=" * 70)
    print()
    
    try:
        # Uninstall current PyTorch
        print("Step 1: Uninstalling current PyTorch...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", 
            "torch", "torchvision", "torchaudio", "-y"
        ], check=True)
        print("[OK] Uninstalled")
        print()
        
        # Install CPU version
        print("Step 2: Installing CPU-only PyTorch...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio"
        ], check=True)
        print("[OK] Installed")
        print()
        
        print("=" * 70)
        print("SUCCESS! PyTorch CPU version installed")
        print("=" * 70)
        print()
        print("You can now run the program with:")
        print("  python ollama_trainer_v2.py")
        print()
        print("NOTE: Training will use CPU (very slow)")
        print("For GPU support, install Visual C++ Redistributables first,")
        print("then reinstall PyTorch with CUDA support.")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print("[ERROR] Installation failed")
        print(f"Error: {e}")
        print()
        print("Try manually running:")
        print("  python -m pip uninstall torch torchvision torchaudio -y")
        print("  python -m pip install torch torchvision torchaudio")
        print()
else:
    print()
    print("No changes made.")
    print()
    print("To fix manually:")
    print("1. Install Visual C++ Redistributables (link above)")
    print("2. Restart your computer")
    print("3. Try running the program again")
    print()
    print("OR:")
    print("1. Uninstall PyTorch: pip uninstall torch torchvision torchaudio -y")
    print("2. Install CPU version: pip install torch torchvision torchaudio")
    print()

print("=" * 70)
input("Press ENTER to close...")
