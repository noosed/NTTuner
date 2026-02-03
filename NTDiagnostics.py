# -*- coding: utf-8 -*-
"""
NTDiagnostics - Dependency Installer & Repair Tool for NTTuner
==============================================================

This tool will:
1. Check all required dependencies
2. Detect and fix common issues
3. Install missing packages
4. Configure GPU support (CUDA/ROCm/MPS)
5. Install llama.cpp for GGUF export
6. Verify Ollama installation
7. Test the complete pipeline

Run this BEFORE using NTTuner to ensure everything works correctly.
"""

import subprocess
import sys
import os
import platform
import shutil
import json
import tempfile
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Status(Enum):
    OK = "✓"
    WARNING = "⚠"
    ERROR = "✗"
    INSTALLING = "⟳"
    SKIPPED = "○"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    fix_available: bool = False
    fix_function: Optional[callable] = None
    details: Optional[str] = None


# Core dependencies required for NTTuner
CORE_DEPENDENCIES = [
    ("torch", "torch", "PyTorch - Deep learning framework"),
    ("transformers", "transformers", "Hugging Face Transformers"),
    ("datasets", "datasets", "Hugging Face Datasets"),
    ("peft", "peft", "Parameter-Efficient Fine-Tuning"),
    ("trl", "trl", "Transformer Reinforcement Learning"),
    ("accelerate", "accelerate", "Hugging Face Accelerate"),
    ("bitsandbytes", "bitsandbytes", "8-bit optimizers (GPU only)"),
    ("safetensors", "safetensors", "Safe tensor serialization"),
    ("sentencepiece", "sentencepiece", "Tokenizer support"),
    ("protobuf", "protobuf", "Protocol buffers"),
]

# Optional but recommended
OPTIONAL_DEPENDENCIES = [
    ("unsloth", "unsloth[colab-new]", "Unsloth - 2x faster training"),
    ("flash_attn", "flash-attn", "Flash Attention 2"),
    ("xformers", "xformers", "Memory-efficient attention"),
    ("wandb", "wandb", "Weights & Biases logging"),
    ("tensorboard", "tensorboard", "TensorBoard logging"),
]

# GUI dependencies
GUI_DEPENDENCIES = [
    ("dearpygui", "dearpygui", "Dear PyGui - GUI framework"),
    ("psutil", "psutil", "System monitoring"),
]

# GGUF export dependencies
GGUF_DEPENDENCIES = [
    ("gguf", "gguf", "GGUF format support"),
    ("numpy", "numpy", "Numerical computing"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(text: str):
    """Print a formatted header"""
    width = 70
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def print_status(name: str, status: Status, message: str, indent: int = 0):
    """Print a status line"""
    prefix = "  " * indent
    status_colors = {
        Status.OK: "\033[92m",      # Green
        Status.WARNING: "\033[93m",  # Yellow
        Status.ERROR: "\033[91m",    # Red
        Status.INSTALLING: "\033[94m",  # Blue
        Status.SKIPPED: "\033[90m",  # Gray
    }
    reset = "\033[0m"
    
    # Check if terminal supports colors
    if sys.platform == "win32":
        # Enable ANSI colors on Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            status_colors = {s: "" for s in Status}
            reset = ""
    
    color = status_colors.get(status, "")
    print(f"{prefix}[{color}{status.value}{reset}] {name}: {message}")


def run_command(cmd: List[str], capture: bool = True, timeout: int = 300) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except Exception as e:
        return -1, "", str(e)


def pip_install(package: str, extra_args: List[str] = None) -> Tuple[bool, str]:
    """Install a package using pip"""
    cmd = [sys.executable, "-m", "pip", "install", package]
    if extra_args:
        cmd.extend(extra_args)
    
    print_status(package, Status.INSTALLING, "Installing...", indent=1)
    returncode, stdout, stderr = run_command(cmd, timeout=600)
    
    if returncode == 0:
        return True, "Installed successfully"
    else:
        return False, stderr or stdout or "Installation failed"


def check_import(module_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a module can be imported and get its version"""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if version is None:
            version = getattr(module, "version", None)
        if version is None and hasattr(module, "VERSION"):
            version = module.VERSION
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_python_version() -> CheckResult:
    """Check Python version compatibility"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3:
        return CheckResult(
            "Python Version",
            Status.ERROR,
            f"Python 3.8+ required, found {version_str}",
            fix_available=False
        )
    elif version.minor < 8:
        return CheckResult(
            "Python Version",
            Status.ERROR,
            f"Python 3.8+ required, found {version_str}",
            fix_available=False
        )
    elif version.minor < 10:
        return CheckResult(
            "Python Version",
            Status.WARNING,
            f"Python {version_str} (3.10+ recommended)",
            fix_available=False
        )
    else:
        return CheckResult(
            "Python Version",
            Status.OK,
            f"Python {version_str}"
        )


def check_platform() -> CheckResult:
    """Check operating system"""
    system = platform.system()
    machine = platform.machine()
    
    details = f"{system} {platform.release()} ({machine})"
    
    if system == "Windows":
        return CheckResult("Platform", Status.OK, details)
    elif system == "Linux":
        return CheckResult("Platform", Status.OK, details)
    elif system == "Darwin":
        # Check for Apple Silicon
        if machine == "arm64":
            return CheckResult("Platform", Status.OK, f"macOS Apple Silicon ({details})")
        else:
            return CheckResult("Platform", Status.OK, f"macOS Intel ({details})")
    else:
        return CheckResult("Platform", Status.WARNING, f"Unknown platform: {details}")


def check_gpu() -> CheckResult:
    """Comprehensive GPU detection"""
    gpu_info = {
        "type": "None",
        "name": "None",
        "memory": 0,
        "cuda_version": None,
    }
    
    # Check CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["type"] = "CUDA"
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info["cuda_version"] = torch.version.cuda
            
            return CheckResult(
                "GPU",
                Status.OK,
                f"NVIDIA {gpu_info['name']} ({gpu_info['memory']:.1f} GB, CUDA {gpu_info['cuda_version']})"
            )
    except:
        pass
    
    # Check ROCm (AMD)
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            gpu_info["type"] = "ROCm"
            gpu_info["name"] = torch.hip.get_device_name(0)
            return CheckResult(
                "GPU",
                Status.OK,
                f"AMD {gpu_info['name']} (ROCm)"
            )
    except:
        pass
    
    # Check MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info["type"] = "MPS"
            return CheckResult(
                "GPU",
                Status.OK,
                "Apple Silicon (Metal Performance Shaders)"
            )
    except:
        pass
    
    # Check for NVIDIA GPU without CUDA
    try:
        returncode, stdout, stderr = run_command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if returncode == 0 and stdout.strip():
            gpu_name = stdout.strip().split(",")[0]
            return CheckResult(
                "GPU",
                Status.WARNING,
                f"NVIDIA {gpu_name} detected but CUDA not available in PyTorch",
                fix_available=True,
                details="Install PyTorch with CUDA support"
            )
    except:
        pass
    
    return CheckResult(
        "GPU",
        Status.WARNING,
        "No GPU detected - training will be VERY slow on CPU",
        fix_available=True,
        details="Consider using a machine with GPU or cloud GPU"
    )


def check_memory() -> CheckResult:
    """Check available system memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        if total_gb < 8:
            return CheckResult(
                "System Memory",
                Status.WARNING,
                f"{total_gb:.1f} GB total ({available_gb:.1f} GB available) - 16GB+ recommended"
            )
        elif total_gb < 16:
            return CheckResult(
                "System Memory",
                Status.OK,
                f"{total_gb:.1f} GB total ({available_gb:.1f} GB available)"
            )
        else:
            return CheckResult(
                "System Memory",
                Status.OK,
                f"{total_gb:.1f} GB total ({available_gb:.1f} GB available)"
            )
    except ImportError:
        return CheckResult(
            "System Memory",
            Status.WARNING,
            "Could not check (psutil not installed)",
            fix_available=True
        )


def check_disk_space() -> CheckResult:
    """Check available disk space"""
    try:
        if sys.platform == "win32":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(str(Path.home())),
                None, None,
                ctypes.pointer(free_bytes)
            )
            free_gb = free_bytes.value / (1024**3)
        else:
            stat = os.statvfs(Path.home())
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        if free_gb < 10:
            return CheckResult(
                "Disk Space",
                Status.ERROR,
                f"Only {free_gb:.1f} GB free - need at least 10GB"
            )
        elif free_gb < 50:
            return CheckResult(
                "Disk Space",
                Status.WARNING,
                f"{free_gb:.1f} GB free - 50GB+ recommended for models"
            )
        else:
            return CheckResult(
                "Disk Space",
                Status.OK,
                f"{free_gb:.1f} GB free"
            )
    except Exception as e:
        return CheckResult(
            "Disk Space",
            Status.WARNING,
            f"Could not check: {e}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_dependency(module_name: str, pip_name: str, description: str) -> CheckResult:
    """Check a single dependency"""
    can_import, version_or_error = check_import(module_name)
    
    if can_import:
        version_str = f"v{version_or_error}" if version_or_error else ""
        return CheckResult(
            description,
            Status.OK,
            f"Installed {version_str}".strip()
        )
    else:
        return CheckResult(
            description,
            Status.ERROR,
            "Not installed",
            fix_available=True,
            details=pip_name
        )


def check_pytorch_cuda() -> CheckResult:
    """Check PyTorch CUDA configuration"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            # Check if NVIDIA GPU exists
            try:
                returncode, stdout, _ = run_command(["nvidia-smi", "-L"])
                if returncode == 0 and "GPU" in stdout:
                    return CheckResult(
                        "PyTorch CUDA",
                        Status.ERROR,
                        "NVIDIA GPU found but PyTorch CUDA not working",
                        fix_available=True,
                        details="Reinstall PyTorch with CUDA"
                    )
            except:
                pass
            
            return CheckResult(
                "PyTorch CUDA",
                Status.SKIPPED,
                "No NVIDIA GPU (skipped)"
            )
        
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        
        msg = f"CUDA {cuda_version}"
        if cudnn_version:
            msg += f", cuDNN {cudnn_version}"
        
        return CheckResult(
            "PyTorch CUDA",
            Status.OK,
            msg
        )
    except ImportError:
        return CheckResult(
            "PyTorch CUDA",
            Status.ERROR,
            "PyTorch not installed",
            fix_available=True
        )


def check_bitsandbytes() -> CheckResult:
    """Check bitsandbytes installation (tricky on Windows)"""
    try:
        import bitsandbytes as bnb
        
        # Try to actually use it
        try:
            import torch
            if torch.cuda.is_available():
                # Quick test
                linear = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False)
                return CheckResult(
                    "bitsandbytes",
                    Status.OK,
                    f"Working (v{bnb.__version__})"
                )
        except Exception as e:
            return CheckResult(
                "bitsandbytes",
                Status.WARNING,
                f"Installed but may have issues: {str(e)[:50]}",
                fix_available=True
            )
        
        return CheckResult(
            "bitsandbytes",
            Status.OK,
            f"Installed (v{bnb.__version__})"
        )
    except ImportError:
        if sys.platform == "win32":
            return CheckResult(
                "bitsandbytes",
                Status.WARNING,
                "Not installed (optional on Windows)",
                fix_available=True,
                details="bitsandbytes-windows"
            )
        return CheckResult(
            "bitsandbytes",
            Status.ERROR,
            "Not installed",
            fix_available=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LLAMA.CPP CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def find_llama_cpp() -> Optional[Path]:
    """Find llama.cpp installation"""
    search_paths = [
        Path.home() / "llama.cpp",
        Path.cwd() / "llama.cpp",
        Path.cwd().parent / "llama.cpp",
    ]
    
    if sys.platform == "win32":
        username = os.environ.get("USERNAME", "")
        search_paths.extend([
            Path(f"C:/llama.cpp"),
            Path(f"C:/Users/{username}/llama.cpp"),
            Path(f"C:/Users/{username}/Documents/llama.cpp"),
            Path(f"C:/Users/{username}/Desktop/llama.cpp"),
            Path(f"D:/llama.cpp"),
        ])
    else:
        search_paths.extend([
            Path("/usr/local/share/llama.cpp"),
            Path("/opt/llama.cpp"),
        ])
    
    for path in search_paths:
        if path.exists():
            # Check for convert script
            convert_scripts = [
                "convert_hf_to_gguf.py",
                "convert-hf-to-gguf.py",
                "convert.py"
            ]
            for script in convert_scripts:
                if (path / script).exists():
                    return path
    
    return None


def find_llama_quantize() -> Optional[Path]:
    """Find llama-quantize binary"""
    if sys.platform == "win32":
        exe_names = ["llama-quantize.exe", "quantize.exe"]
    else:
        exe_names = ["llama-quantize", "quantize"]
    
    # Check PATH
    for exe in exe_names:
        path = shutil.which(exe)
        if path:
            return Path(path)
    
    # Check llama.cpp directory
    llama_cpp = find_llama_cpp()
    if llama_cpp:
        for exe in exe_names:
            for subdir in ["", "build/bin", "build", "bin"]:
                exe_path = llama_cpp / subdir / exe if subdir else llama_cpp / exe
                if exe_path.exists():
                    return exe_path
    
    return None


def check_llama_cpp() -> CheckResult:
    """Check llama.cpp installation"""
    llama_path = find_llama_cpp()
    
    if llama_path:
        # Check for convert script
        convert_script = None
        for script in ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]:
            if (llama_path / script).exists():
                convert_script = llama_path / script
                break
        
        if convert_script:
            return CheckResult(
                "llama.cpp",
                Status.OK,
                f"Found at {llama_path}",
                details=str(convert_script)
            )
        else:
            return CheckResult(
                "llama.cpp",
                Status.WARNING,
                f"Found at {llama_path} but convert script missing",
                fix_available=True
            )
    
    return CheckResult(
        "llama.cpp",
        Status.ERROR,
        "Not found - required for GGUF export",
        fix_available=True,
        details="Will clone from GitHub"
    )


def check_llama_quantize() -> CheckResult:
    """Check llama-quantize binary"""
    quantize_path = find_llama_quantize()
    
    if quantize_path:
        return CheckResult(
            "llama-quantize",
            Status.OK,
            f"Found at {quantize_path}"
        )
    
    return CheckResult(
        "llama-quantize",
        Status.WARNING,
        "Not found - quantization will use Python fallback",
        fix_available=True,
        details="Build llama.cpp or download prebuilt"
    )


def check_gguf_package() -> CheckResult:
    """Check gguf Python package"""
    can_import, version = check_import("gguf")
    
    if can_import:
        return CheckResult(
            "gguf package",
            Status.OK,
            f"Installed {f'v{version}' if version else ''}"
        )
    
    return CheckResult(
        "gguf package",
        Status.ERROR,
        "Not installed - required for GGUF export",
        fix_available=True,
        details="gguf"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_ollama() -> CheckResult:
    """Check Ollama installation"""
    ollama_path = shutil.which("ollama")
    
    if not ollama_path:
        # Check common installation paths
        if sys.platform == "win32":
            possible_paths = [
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                Path("C:/Program Files/Ollama/ollama.exe"),
            ]
        elif sys.platform == "darwin":
            possible_paths = [
                Path("/usr/local/bin/ollama"),
                Path.home() / ".ollama" / "ollama",
            ]
        else:
            possible_paths = [
                Path("/usr/local/bin/ollama"),
                Path("/usr/bin/ollama"),
            ]
        
        for path in possible_paths:
            if path.exists():
                ollama_path = str(path)
                break
    
    if not ollama_path:
        return CheckResult(
            "Ollama",
            Status.ERROR,
            "Not installed",
            fix_available=True,
            details="https://ollama.ai"
        )
    
    # Check if running
    returncode, stdout, stderr = run_command(["ollama", "list"], timeout=10)
    
    if returncode == 0:
        # Count models
        lines = stdout.strip().split('\n')
        model_count = len(lines) - 1 if len(lines) > 1 else 0
        return CheckResult(
            "Ollama",
            Status.OK,
            f"Running ({model_count} models installed)"
        )
    else:
        return CheckResult(
            "Ollama",
            Status.WARNING,
            "Installed but not running",
            fix_available=True,
            details="Start Ollama service"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FIX FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fix_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print_header("Installing PyTorch with CUDA")
    
    # Detect best CUDA version
    cuda_version = "cu121"  # Default to CUDA 12.1
    
    try:
        returncode, stdout, _ = run_command(["nvidia-smi"])
        if returncode == 0:
            # Parse CUDA version from nvidia-smi
            import re
            match = re.search(r"CUDA Version: (\d+)\.(\d+)", stdout)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                if major >= 12:
                    cuda_version = "cu121"
                elif major == 11 and minor >= 8:
                    cuda_version = "cu118"
                else:
                    cuda_version = "cu117"
    except:
        pass
    
    print(f"  Detected CUDA compatibility: {cuda_version}")
    
    # Uninstall existing torch
    print("  Removing existing PyTorch...")
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    
    # Install PyTorch with CUDA
    print(f"  Installing PyTorch with {cuda_version}...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_version}"
    ]
    
    returncode, stdout, stderr = run_command(cmd, timeout=900)
    
    if returncode == 0:
        print_status("PyTorch CUDA", Status.OK, "Installed successfully", indent=1)
        return True
    else:
        print_status("PyTorch CUDA", Status.ERROR, f"Failed: {stderr[:100]}", indent=1)
        return False


def fix_bitsandbytes():
    """Install bitsandbytes (with Windows support)"""
    print_header("Installing bitsandbytes")
    
    if sys.platform == "win32":
        # Use Windows-specific package
        print("  Using Windows-compatible bitsandbytes...")
        success, msg = pip_install("bitsandbytes-windows")
        if not success:
            # Try alternative
            success, msg = pip_install("bitsandbytes", ["--prefer-binary"])
    else:
        success, msg = pip_install("bitsandbytes")
    
    if success:
        print_status("bitsandbytes", Status.OK, "Installed", indent=1)
    else:
        print_status("bitsandbytes", Status.WARNING, f"Optional: {msg[:50]}", indent=1)
    
    return success


def fix_llama_cpp():
    """Install llama.cpp"""
    print_header("Installing llama.cpp")
    
    install_path = Path.home() / "llama.cpp"
    
    if install_path.exists():
        print(f"  Directory exists at {install_path}")
        print("  Updating...")
        returncode, _, stderr = run_command(
            ["git", "pull"],
            timeout=120
        )
        if returncode != 0:
            print_status("llama.cpp", Status.WARNING, "Could not update", indent=1)
    else:
        print(f"  Cloning to {install_path}...")
        returncode, _, stderr = run_command([
            "git", "clone",
            "https://github.com/ggerganov/llama.cpp.git",
            str(install_path)
        ], timeout=300)
        
        if returncode != 0:
            print_status("llama.cpp", Status.ERROR, f"Clone failed: {stderr[:50]}", indent=1)
            return False
    
    # Install Python dependencies
    print("  Installing Python dependencies...")
    pip_install("gguf")
    pip_install("sentencepiece")
    
    # Verify
    convert_script = install_path / "convert_hf_to_gguf.py"
    if convert_script.exists():
        print_status("llama.cpp", Status.OK, f"Installed at {install_path}", indent=1)
        return True
    else:
        print_status("llama.cpp", Status.ERROR, "Convert script not found", indent=1)
        return False


def fix_dependencies(deps: List[Tuple[str, str, str]], optional: bool = False):
    """Install a list of dependencies"""
    for module_name, pip_name, description in deps:
        can_import, _ = check_import(module_name)
        if not can_import:
            success, msg = pip_install(pip_name, ["--break-system-packages"] if sys.platform == "linux" else [])
            if not success and not optional:
                print_status(description, Status.ERROR, f"Failed: {msg[:50]}", indent=1)
            elif not success and optional:
                print_status(description, Status.SKIPPED, "Optional, skipped", indent=1)


def fix_all():
    """Run all fixes"""
    print_header("Running All Fixes")
    
    # Core dependencies
    print("\n  Installing core dependencies...")
    fix_dependencies(CORE_DEPENDENCIES)
    
    # GUI dependencies
    print("\n  Installing GUI dependencies...")
    fix_dependencies(GUI_DEPENDENCIES)
    
    # GGUF dependencies
    print("\n  Installing GGUF dependencies...")
    fix_dependencies(GGUF_DEPENDENCIES)
    
    # llama.cpp
    print("\n  Setting up llama.cpp...")
    fix_llama_cpp()
    
    # Check if CUDA fix needed
    try:
        import torch
        if not torch.cuda.is_available():
            # Check for NVIDIA GPU
            returncode, _, _ = run_command(["nvidia-smi", "-L"])
            if returncode == 0:
                print("\n  Fixing PyTorch CUDA...")
                fix_pytorch_cuda()
    except:
        pass
    
    print("\n" + "─" * 70)
    print("  Fix process complete. Re-run diagnostics to verify.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(auto_fix: bool = False) -> Dict[str, List[CheckResult]]:
    """Run all diagnostic checks"""
    results = {
        "system": [],
        "core": [],
        "gpu": [],
        "gguf": [],
        "ollama": [],
    }
    
    # System checks
    print_header("System Checks")
    
    results["system"].append(check_python_version())
    print_status(*[results["system"][-1].name, results["system"][-1].status, results["system"][-1].message])
    
    results["system"].append(check_platform())
    print_status(*[results["system"][-1].name, results["system"][-1].status, results["system"][-1].message])
    
    results["system"].append(check_memory())
    print_status(*[results["system"][-1].name, results["system"][-1].status, results["system"][-1].message])
    
    results["system"].append(check_disk_space())
    print_status(*[results["system"][-1].name, results["system"][-1].status, results["system"][-1].message])
    
    # GPU checks
    print_header("GPU Detection")
    
    results["gpu"].append(check_gpu())
    print_status(*[results["gpu"][-1].name, results["gpu"][-1].status, results["gpu"][-1].message])
    
    results["gpu"].append(check_pytorch_cuda())
    print_status(*[results["gpu"][-1].name, results["gpu"][-1].status, results["gpu"][-1].message])
    
    # Core dependencies
    print_header("Core Dependencies")
    
    for module_name, pip_name, description in CORE_DEPENDENCIES:
        result = check_dependency(module_name, pip_name, description)
        results["core"].append(result)
        print_status(result.name, result.status, result.message)
        
        if auto_fix and result.status == Status.ERROR and result.fix_available:
            pip_install(pip_name)
    
    # GUI dependencies
    print_header("GUI Dependencies")
    
    for module_name, pip_name, description in GUI_DEPENDENCIES:
        result = check_dependency(module_name, pip_name, description)
        results["core"].append(result)
        print_status(result.name, result.status, result.message)
        
        if auto_fix and result.status == Status.ERROR and result.fix_available:
            pip_install(pip_name)
    
    # GGUF export checks
    print_header("GGUF Export")
    
    results["gguf"].append(check_llama_cpp())
    print_status(*[results["gguf"][-1].name, results["gguf"][-1].status, results["gguf"][-1].message])
    
    results["gguf"].append(check_llama_quantize())
    print_status(*[results["gguf"][-1].name, results["gguf"][-1].status, results["gguf"][-1].message])
    
    results["gguf"].append(check_gguf_package())
    print_status(*[results["gguf"][-1].name, results["gguf"][-1].status, results["gguf"][-1].message])
    
    if auto_fix:
        for result in results["gguf"]:
            if result.status == Status.ERROR and "llama.cpp" in result.name:
                fix_llama_cpp()
                break
    
    # Ollama checks
    print_header("Ollama")
    
    results["ollama"].append(check_ollama())
    print_status(*[results["ollama"][-1].name, results["ollama"][-1].status, results["ollama"][-1].message])
    
    return results


def print_summary(results: Dict[str, List[CheckResult]]):
    """Print a summary of all checks"""
    print_header("Summary")
    
    total_ok = 0
    total_warning = 0
    total_error = 0
    fixable = []
    
    for category, checks in results.items():
        for check in checks:
            if check.status == Status.OK:
                total_ok += 1
            elif check.status == Status.WARNING:
                total_warning += 1
            elif check.status == Status.ERROR:
                total_error += 1
                if check.fix_available:
                    fixable.append(check)
    
    print(f"\n  ✓ Passed: {total_ok}")
    print(f"  ⚠ Warnings: {total_warning}")
    print(f"  ✗ Errors: {total_error}")
    
    if fixable:
        print(f"\n  {len(fixable)} issues can be auto-fixed:")
        for check in fixable:
            print(f"    - {check.name}")
    
    if total_error == 0 and total_warning == 0:
        print("\n  ╔════════════════════════════════════════════╗")
        print("  ║  All checks passed! NTTuner is ready.     ║")
        print("  ╚════════════════════════════════════════════╝")
    elif total_error == 0:
        print("\n  System is functional with minor warnings.")
        print("  You can proceed with NTTuner.")
    else:
        print("\n  ╔════════════════════════════════════════════╗")
        print("  ║  Some issues need to be fixed.            ║")
        print("  ║  Run with --fix to auto-repair.           ║")
        print("  ╚════════════════════════════════════════════╝")


def interactive_menu():
    """Show interactive menu"""
    while True:
        print("\n" + "═" * 70)
        print("  NTDiagnostics - Main Menu")
        print("═" * 70)
        print("\n  1. Run Full Diagnostics")
        print("  2. Run Diagnostics + Auto-Fix")
        print("  3. Install/Update llama.cpp")
        print("  4. Fix PyTorch CUDA")
        print("  5. Install All Dependencies")
        print("  6. Check GPU Only")
        print("  7. Check Ollama Only")
        print("  0. Exit")
        print()
        
        try:
            choice = input("  Select option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break
        
        if choice == "1":
            results = run_diagnostics(auto_fix=False)
            print_summary(results)
        elif choice == "2":
            results = run_diagnostics(auto_fix=True)
            print_summary(results)
        elif choice == "3":
            fix_llama_cpp()
        elif choice == "4":
            fix_pytorch_cuda()
        elif choice == "5":
            fix_all()
        elif choice == "6":
            print_header("GPU Check")
            result = check_gpu()
            print_status(result.name, result.status, result.message)
            result = check_pytorch_cuda()
            print_status(result.name, result.status, result.message)
        elif choice == "7":
            print_header("Ollama Check")
            result = check_ollama()
            print_status(result.name, result.status, result.message)
        elif choice == "0":
            print("\n  Goodbye!")
            break
        else:
            print("  Invalid option")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ██╗████████╗██████╗ ██╗ █████╗  ██████╗ ███╗   ██╗ ██████╗ ███████╗ ║
║   ████╗  ██║╚══██╔══╝██╔══██╗██║██╔══██╗██╔════╝ ████╗  ██║██╔═══██╗██╔════╝ ║
║   ██╔██╗ ██║   ██║   ██║  ██║██║███████║██║  ███╗██╔██╗ ██║██║   ██║███████╗ ║
║   ██║╚██╗██║   ██║   ██║  ██║██║██╔══██║██║   ██║██║╚██╗██║██║   ██║╚════██║ ║
║   ██║ ╚████║   ██║   ██████╔╝██║██║  ██║╚██████╔╝██║ ╚████║╚██████╔╝███████║ ║
║   ╚═╝  ╚═══╝   ╚═╝   ╚═════╝ ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝ ║
║                                                                              ║
║              Dependency Installer & Repair Tool for NTTuner                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ["--fix", "-f", "fix"]:
            results = run_diagnostics(auto_fix=True)
            print_summary(results)
        elif arg in ["--check", "-c", "check"]:
            results = run_diagnostics(auto_fix=False)
            print_summary(results)
        elif arg in ["--install-all", "-i", "install"]:
            fix_all()
        elif arg in ["--llama", "llama"]:
            fix_llama_cpp()
        elif arg in ["--cuda", "cuda"]:
            fix_pytorch_cuda()
        elif arg in ["--help", "-h", "help"]:
            print("""
Usage: python NTDiagnostics.py [option]

Options:
  --check, -c      Run diagnostics only
  --fix, -f        Run diagnostics and auto-fix issues
  --install-all    Install all dependencies
  --llama          Install/update llama.cpp
  --cuda           Fix PyTorch CUDA installation
  --help, -h       Show this help

No arguments: Show interactive menu
            """)
        else:
            print(f"Unknown option: {arg}")
            print("Use --help for usage information")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
