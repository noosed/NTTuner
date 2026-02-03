# -*- coding: utf-8 -*-
try:
    import chronicals
    from chronicals import ChronicalsTrainer, ChronicalsConfig, SequencePacker
    from chronicals.optim import LoRAPlusOptimizer

    CHRONICALS_AVAILABLE = True
except ImportError:
    CHRONICALS_AVAILABLE = False

# DearPyGui is lazy loaded to prevent early crashes
dpg = None


def _init_dpg():
    """Initialize DearPyGui only when needed"""
    global dpg
    if dpg is None:
        try:
            import dearpygui.dearpygui as _dpg
            dpg = _dpg

            # Windows DPI fix
            if sys.platform == "win32":
                try:
                    import ctypes
                    try:
                        ctypes.windll.shcore.SetProcessDpiAwareness(2)
                    except:
                        ctypes.windll.user32.SetProcessDPIAware()
                except:
                    pass

            return True
        except Exception as e:
            print(f"ERROR: Could not load DearPyGui: {e}")
            print("Install with: pip install dearpygui")
            return False
    return True


# import dearpygui.dearpygui as dpg  # Lazy loaded
import subprocess
import os
import json
import threading
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import traceback
import sys
import shutil

# ═══════════════════════════════════════════════════════════════════════
# CRASH PROTECTION & ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════

import ctypes
import signal
import atexit


def safe_exit():
    """Safe cleanup on exit"""
    try:
        # import dearpygui.dearpygui as dpg  # Lazy loaded
        if dpg.is_dearpygui_running():
            dpg.stop_dearpygui()
            dpg.destroy_context()
    except:
        pass


atexit.register(safe_exit)


def signal_handler(signum, frame):
    """Handle crash signals"""
    print(f"\n[!] Received signal {signum}, exiting safely...")
    safe_exit()
    sys.exit(1)


# Register signal handlers
if sys.platform == "win32":
    # Windows-specific crash handling
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetErrorMode(0x0001 | 0x0002)  # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
    except:
        pass
else:
    signal.signal(signal.SIGSEGV, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)


# ═══════════════════════════════════════════════════════════════════════

def safe_subprocess_run(cmd, **kwargs):
    """Safely run subprocess with crash protection"""
    try:
        # Set default timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 3600

        # Ensure we capture output
        if 'capture_output' not in kwargs:
            kwargs['capture_output'] = True
        if 'text' not in kwargs:
            kwargs['text'] = True

        # On Windows, add CREATE_NO_WINDOW flag to avoid popup windows
        if sys.platform == "win32":
            import subprocess
            if 'creationflags' not in kwargs:
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

        return safe_subprocess_run(cmd, **kwargs)
    except subprocess.TimeoutExpired as e:
        # Return a fake result object
        class TimeoutResult:
            returncode = -1
            stdout = ""
            stderr = f"Process timed out after {e.timeout} seconds"

        return TimeoutResult()
    except FileNotFoundError as e:
        class NotFoundResult:
            returncode = -1
            stdout = ""
            stderr = f"Command not found: {cmd[0]}"

        return NotFoundResult()
    except MemoryError:
        class MemoryResult:
            returncode = -1
            stdout = ""
            stderr = "Out of memory while running process"

        return MemoryResult()
    except Exception as e:
        class ErrorResult:
            returncode = -1
            stdout = ""
            stderr = f"Process error: {str(e)}"

        return ErrorResult()


# ENHANCED GPU DETECTION
# ═══════════════════════════════════════════════════════════════════════


# Set memory limits to prevent crashes
def set_memory_limits():
    """Set reasonable memory limits to prevent crashes"""
    if sys.platform == "win32":
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            # Use at most 80% of available memory
            max_memory = int(available_memory * 0.8)

            # Set process memory limit (Windows)
            import ctypes
            kernel32 = ctypes.windll.kernel32
            job = kernel32.CreateJobObjectA(None, None)
            if job:
                # JOBOBJECT_EXTENDED_LIMIT_INFORMATION structure
                class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                    _fields_ = [
                        ("BasicLimitInformation", ctypes.c_byte * 72),
                        ("IoInfo", ctypes.c_byte * 24),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                    ]

                limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                limits.ProcessMemoryLimit = max_memory

                # Set the job limits
                kernel32.SetInformationJobObject(
                    job, 9,  # JobObjectExtendedLimitInformation
                    ctypes.byref(limits),
                    ctypes.sizeof(limits)
                )
        except:
            pass  # Silent fail if psutil not available or limits can't be set


try:
    set_memory_limits()
except:
    pass


def detect_gpu_comprehensive():
    """
    Comprehensive GPU detection with robust error handling
    Supports: NVIDIA CUDA, AMD ROCm, Apple Metal (MPS)
    """
    gpu_info = {
        "has_gpu": False,
        "gpu_type": "CPU",
        "gpu_name": "None",
        "gpu_memory": 0.0,
        "gpu_count": 0,
        "backend": "cpu",
        "cuda_version": None,
        "details": [],
        "warnings": []
    }

    # Try to import torch
    try:
        import torch
        gpu_info["details"].append(f"PyTorch {torch.__version__}")
    except ImportError:
        gpu_info["details"].append("PyTorch not installed")
        gpu_info["warnings"].append("Run: pip install torch")
        return gpu_info
    except Exception as e:
        gpu_info["details"].append(f"PyTorch error: {e}")
        return gpu_info

    # CUDA Detection (NVIDIA)
    try:
        if torch.cuda.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "CUDA"
            gpu_info["backend"] = "cuda"
            gpu_info["gpu_count"] = torch.cuda.device_count()

            try:
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_info["gpu_memory"] = props.total_memory / (1024 ** 3)
            except Exception:
                gpu_info["gpu_name"] = "CUDA GPU"
                gpu_info["gpu_memory"] = 8.0

            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["details"].append(f"CUDA {torch.version.cuda}")

            gpu_info["details"].append(f"Device: {gpu_info['gpu_name']}")
            gpu_info["details"].append(f"VRAM: {gpu_info['gpu_memory']:.1f} GB")

            if gpu_info["gpu_count"] > 1:
                gpu_info["details"].append(f"GPUs: {gpu_info['gpu_count']}")

            # Check bitsandbytes
            try:
                import bitsandbytes
                gpu_info["details"].append("Quantization: Available")
            except Exception:
                gpu_info["warnings"].append("bitsandbytes not available - no 4-bit/8-bit")

            return gpu_info
    except Exception:
        pass

    # MPS Detection (Apple Silicon)
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "MPS"
            gpu_info["backend"] = "mps"
            gpu_info["gpu_count"] = 1

            try:
                import platform
                if 'arm' in platform.machine().lower():
                    gpu_info["gpu_name"] = "Apple Silicon GPU"
                    try:
                        import psutil
                        total_ram = psutil.virtual_memory().total / (1024 ** 3)
                        gpu_info["gpu_memory"] = total_ram * 0.7
                    except Exception:
                        gpu_info["gpu_memory"] = 16.0
                else:
                    gpu_info["gpu_name"] = "Metal GPU"
                    gpu_info["gpu_memory"] = 8.0
            except Exception:
                gpu_info["gpu_name"] = "Apple Metal GPU"
                gpu_info["gpu_memory"] = 8.0

            gpu_info["details"].append("Apple Metal Performance Shaders")
            gpu_info["details"].append(f"Device: {gpu_info['gpu_name']}")
            gpu_info["details"].append(f"Unified Memory: ~{gpu_info['gpu_memory']:.1f} GB")
            gpu_info["warnings"].append("MPS is experimental")

            return gpu_info
    except Exception:
        pass

    # ROCm Detection (AMD)
    try:
        if hasattr(torch, 'hip') and torch.hip.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "ROCm"
            gpu_info["backend"] = "hip"
            gpu_info["gpu_count"] = torch.hip.device_count()
            gpu_info["gpu_name"] = torch.hip.get_device_name(0)

            try:
                props = torch.hip.get_device_properties(0)
                gpu_info["gpu_memory"] = props.total_memory / (1024 ** 3)
            except Exception:
                gpu_info["gpu_memory"] = 8.0

            gpu_info["details"].append("AMD ROCm")
            gpu_info["details"].append(f"Device: {gpu_info['gpu_name']}")
            gpu_info["details"].append(f"VRAM: {gpu_info['gpu_memory']:.1f} GB")

            return gpu_info
    except Exception:
        pass

    # CPU Fallback
    gpu_info["details"].append("No GPU detected - using CPU")
    gpu_info["details"].append("Training will be very slow")
    gpu_info["warnings"].append("Consider using cloud GPU (Google Colab, etc.)")

    return gpu_info


# Initialize GPU detection
GPU_INFO = detect_gpu_comprehensive()
HAS_GPU = GPU_INFO["has_gpu"]
GPU_TYPE = GPU_INFO["gpu_type"]
GPU_NAME = GPU_INFO["gpu_name"]
GPU_MEMORY = GPU_INFO["gpu_memory"]
GPU_COUNT = GPU_INFO["gpu_count"]
GPU_BACKEND = GPU_INFO["backend"]


# ═══════════════════════════════════════════════════════════════════════
# DEPENDENCY CHECKS
# ═══════════════════════════════════════════════════════════════════════

def get_ollama_models():
    """Get list of installed Ollama models"""
    try:
        result = safe_subprocess_run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                models = []
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
                return models
    except:
        pass
    return []


def get_popular_models():
    """Get comprehensive list of models for fine-tuning"""
    models = {
        "Ollama Models (Installed)\u200b": get_ollama_models(),
        # ── Ollama-pull names (colon format).  These are ONLY for the
        #     "Download" button.  They will be blocked from training automatically.
        "Ollama Pull Only (Download first)": [
            "llama3.2:3b", "llama3.1:8b",
            "mistral:7b", "mixtral:8x7b",
            "phi3:mini", "phi4:14b",
            "gemma:7b", "gemma2:9b",
            "qwen2.5:0.5b", "qwen2.5:7b", "qwen2.5:14b",
        ],
        # ── HuggingFace IDs  ─── safe to load directly ───
        "Small Models (CPU-friendly)": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "stabilityai/stablelm-2-1_6b",
        ],
        "Medium Models (GPU recommended)": [
            "meta-llama/Llama-3.2-3B-Instruct",
            "unsloth/llama-3-8b-bnb-4bit",
            "unsloth/mistral-7b-v0.3-bnb-4bit",
            "unsloth/Phi-3-mini-4k-instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        "Large Models (Good GPU required)": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "unsloth/llama-3-70b-bnb-4bit",
            "meta-llama/Llama-3.1-70B-Instruct",
        ]
    }

    all_models = []
    for category, model_list in models.items():
        if model_list:
            all_models.append(f"--- {category} ---")
            all_models.extend(model_list)
    return all_models


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from unsloth import FastLanguageModel

    HAS_UNSLOTH = True
except:
    HAS_UNSLOTH = False

try:
    from trl import SFTTrainer
    from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from datasets import load_dataset, Dataset
    from peft import LoraConfig, get_peft_model

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

DEPS_AVAILABLE = HAS_TORCH and HAS_TRANSFORMERS


# ═══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file"""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except:
        return "unknown"


def get_library_versions() -> Dict[str, str]:
    """Get versions of key libraries"""
    versions = {}
    try:
        if HAS_TORCH:
            versions["torch"] = torch.__version__
        if HAS_TRANSFORMERS:
            versions["transformers"] = __import__("transformers").__version__
            versions["datasets"] = __import__("datasets").__version__
            versions["trl"] = __import__("trl").__version__
            versions["peft"] = __import__("peft").__version__
        if HAS_UNSLOTH:
            versions["unsloth"] = "available"
    except:
        pass
    return versions


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def get_device_map() -> str:
    """Get appropriate device map based on backend"""
    if GPU_BACKEND == "cuda":
        return "auto"
    elif GPU_BACKEND == "hip":
        return "auto"
    elif GPU_BACKEND == "mps":
        return "mps"
    else:
        return "cpu"


def get_torch_dtype():
    """Get appropriate torch dtype based on backend"""
    if not HAS_TORCH:
        return None
    if GPU_BACKEND in ["cuda", "hip"]:
        return torch.float16
    elif GPU_BACKEND == "mps":
        return torch.float16
    else:
        return torch.float32


# ────────────────────────────────────────────────────────────────────────
# Added: Full GGUF export options (2026)
# ────────────────────────────────────────────────────────────────────────

# Comprehensive list of llama.cpp quantization types
GGUF_QUANT_TYPES = [
    # Standard K-quants (most common)
    "Q2_K", "Q2_K_S",
    "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K",
    "Q8_0",
    # I-quants (importance matrix based)
    "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_XS", "IQ4_NL",
    # Full precision
    "F16", "F32", "BF16",
    # Copy (no quantization)
    "COPY",
]

# Preset quantization groups for batch export
GGUF_PRESETS = {
    "Standard Quality (Q4_K_M)": ["Q4_K_M"],
    "High Quality (Q5_K_M)": ["Q5_K_M"],
    "Best Quality (Q6_K)": ["Q6_K"],
    "Maximum Quality (Q8_0)": ["Q8_0"],
    "Full Precision (F16)": ["F16"],
    "Small Size (Q3_K_M)": ["Q3_K_M"],
    "Tiny Size (Q2_K)": ["Q2_K"],
    "IQ Optimized (IQ4_XS)": ["IQ4_XS"],
    "All K-Quants": ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    "All Common": ["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
    "Size Ladder (Q2→Q8)": ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    "IQ Series": ["IQ2_M", "IQ3_M", "IQ4_XS", "IQ4_NL"],
}


# ────────────────────────────────────────────────────────────────────────
# Chat-template registry & auto-detection
# ────────────────────────────────────────────────────────────────────────

# Each entry: (list-of-name-fragments-to-match, marker-tokens-present-in-formatted-text, wrap-function)
# wrap_function(system_prompt, user_msg, assistant_msg) -> formatted text string
# If the dataset text already contains the marker tokens we skip wrapping.

def _wrap_chatml(system, user, assistant):
    """Qwen / Yi / Internlm ChatML style"""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>")
    return "\n".join(parts)


def _wrap_llama3(system, user, assistant):
    """Meta Llama-3 / Llama-3.1 / Llama-3.2 style"""
    parts = []
    parts.append("<|begin_of_text|>")
    if system:
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")
    parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>")
    parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>")
    return "".join(parts)


def _wrap_mistral(system, user, assistant):
    """Mistral / Mixtral style"""
    # Mistral doesn't have an official system token; prepend to user turn
    user_text = f"{system}\n\n{user}" if system else user
    return f"[INST] {user_text} [/INST]{assistant}</s>"


def _wrap_phi(system, user, assistant):
    """Microsoft Phi-3 / Phi-4 style"""
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}<|end|>")
    parts.append(f"<|user|>\n{user}<|end|>")
    parts.append(f"<|assistant|>\n{assistant}<|end|>")
    return "\n".join(parts)


def _wrap_gemma(system, user, assistant):
    """Google Gemma style"""
    user_text = f"{system}\n\n{user}" if system else user
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>"


def _wrap_tinyllama(system, user, assistant):
    """TinyLlama / vicuna chat style"""
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}</s>")
    parts.append(f"<|user|>\n{user}</s>")
    parts.append(f"<|assistant|>\n{assistant}</s>")
    return "\n".join(parts)


def _wrap_stablelm(system, user, assistant):
    """StableLM-2 style (same as TinyLlama chat)"""
    return _wrap_tinyllama(system, user, assistant)


# (match_fragments, display_name, marker_tokens, wrap_fn)
MODEL_TEMPLATES: List[Tuple[List[str], str, List[str], Any]] = [
    # Qwen family
    (["qwen"], "ChatML (Qwen)", ["<|im_start|>", "<|im_end|>"], _wrap_chatml),
    # Yi / Internlm also use ChatML
    (["yi-", "internlm"], "ChatML", ["<|im_start|>", "<|im_end|>"], _wrap_chatml),
    # Llama 3.x
    (["llama-3", "llama3", "llama-3.1", "llama-3.2"],
     "Llama-3", ["<|start_header_id|>", "<|eot_id|>"], _wrap_llama3),
    # Mistral / Mixtral
    (["mistral", "mixtral"],
     "Mistral", ["[INST]", "[/INST]"], _wrap_mistral),
    # Phi
    (["phi-3", "phi3", "phi-4", "phi4"],
     "Phi", ["<|system|>", "<|user|>", "<|assistant|>"], _wrap_phi),
    # Gemma
    (["gemma"], "Gemma", ["<start_of_turn>", "<end_of_turn>"], _wrap_gemma),
    # TinyLlama
    (["tinyllama"], "TinyLlama", ["<|system|>", "<|user|>", "<|assistant|>"], _wrap_tinyllama),
    # StableLM
    (["stablelm"], "StableLM", ["<|system|>", "<|user|>", "<|assistant|>"], _wrap_stablelm),
]


def detect_template_for_model(model_name: str) -> Optional[Tuple[str, List[str], Any]]:
    """
    Given a HuggingFace model name, return (display_name, marker_tokens, wrap_fn)
    or None if no template is recognised.
    """
    lower = model_name.lower().replace("/", " ").replace("-", " ").replace("_", " ")
    for fragments, display, markers, wrap_fn in MODEL_TEMPLATES:
        if any(frag.lower() in lower for frag in fragments):
            return display, markers, wrap_fn
    return None


def dataset_has_markers(text: str, markers: List[str]) -> bool:
    """Check whether a text string already contains the expected template marker tokens."""
    return all(m in text for m in markers)


# ────────────────────────────────────────────────────────────────────────
# End: Chat-template registry
# ────────────────────────────────────────────────────────────────────────


def find_llama_quantize(llama_cpp_root: Optional[Path] = None) -> Optional[str]:
    """
    Find the llama-quantize (or quantize) binary.

    If *llama_cpp_root* is supplied (e.g. derived from a known convert-script
    path) we search its build trees first — this is the most reliable way to
    locate the binary on custom / non-standard installs.

    Also checks for llama-cpp-python package and returns "python-fallback" if
    the package is available but no binary is found.
    """
    # ── binary names to try (order matters: most common first) ────────────
    binary_names = [
        "llama-quantize",
        "llama-quantize.exe",  # explicit Windows extension
        "quantize",
        "quantize.exe",
        "llama.cpp-quantize",
        "main-quantize",
    ]

    # ── 1) check PATH first ───────────────────────────────────────────────
    for name in binary_names:
        path = shutil.which(name)
        if path:
            return path

    # ── 1.5) Check for llama-cpp-python package ───────────────────────────
    # Many users install via pip and don't have the compiled binaries
    try:
        import llama_cpp
        # Check if the package has the binary
        package_dir = Path(llama_cpp.__file__).parent
        for name in binary_names:
            candidate = package_dir / name
            if candidate.is_file():
                return str(candidate)
        # Also check in bin subdirectory
        bin_dir = package_dir / "bin"
        if bin_dir.is_dir():
            for name in binary_names:
                candidate = bin_dir / name
                if candidate.is_file():
                    return str(candidate)
    except ImportError:
        pass

    # ── helper: given a base directory, try every known sub-layout ────────
    def _search_root(root: Path) -> Optional[str]:
        # Sub-directories where compiled binaries land on various platforms /
        # build systems (CMake default, Visual Studio, Xcode, plain make, …)
        sub_dirs = [
            root,  # root itself
            root / "build",  # single-config CMake (Windows default)
            root / "build" / "bin",  # Linux/Mac cmake default
            root / "build" / "Release",  # Windows cmake / VS default
            root / "build" / "Debug",  # Windows debug builds
            root / "build" / "x64" / "Release",  # some VS project layouts
            root / "build" / "x86" / "Release",
            root / "build" / "bin" / "Release",  # bin/Release variant
            root / "build" / "Release" / "x64",  # Release/x64 variant
            root / "Release",  # flat VS output
            root / "bin",
        ]
        for sub in sub_dirs:
            if not sub.is_dir():
                continue
            for name in binary_names:
                candidate = sub / name
                if candidate.is_file():
                    return str(candidate)

        # Hardcoded layouts missed – walk root/build (tiny tree, < 1 ms).
        # This catches every CMake variant without needing to guess.
        build_dir = root / "build"
        if build_dir.is_dir():
            binary_set = set(binary_names)
            for dirpath, _dirs, files in os.walk(str(build_dir)):
                for fname in files:
                    if fname in binary_set:
                        return os.path.join(dirpath, fname)
        return None

    # ── 2) if caller handed us a known root, try it first ────────────────
    if llama_cpp_root and llama_cpp_root.is_dir():
        hit = _search_root(llama_cpp_root)
        if hit:
            return hit

    # ── 3) well-known install roots ───────────────────────────────────────
    well_known_roots = [
        Path.home() / "llama.cpp",
        Path.cwd() / "llama.cpp",
        Path("/usr/local/share/llama.cpp"),
        Path("/opt/llama.cpp"),
    ]
    for root in well_known_roots:
        hit = _search_root(root)
        if hit:
            return hit

    # ── 4) legacy flat paths (keep for back-compat) ──────────────────────
    legacy_paths = [
        Path("/usr/local/bin/llama-quantize"),
        Path("/usr/bin/llama-quantize"),
    ]
    for p in legacy_paths:
        if p.is_file():
            return str(p)

    # ── 5) walk the known llama.cpp root fully (if we have one) ──────
    # _search_root checks a fixed list of sub-dirs; if the binary is in
    # an unexpected nested folder this catches it instantly.
    if llama_cpp_root and llama_cpp_root.is_dir():
        binary_set = set(binary_names)
        try:
            for dirpath, _dirs, files in os.walk(str(llama_cpp_root)):
                for fname in files:
                    if fname in binary_set:
                        return os.path.join(dirpath, fname)
        except (PermissionError, OSError):
            pass

    # ── 6) brute-force recursive scan (Windows drive roots) ──────────
    # Last resort – only if we have no root at all.  Uses os.walk so the
    # timeout check fires after every directory.
    if sys.platform == "win32":
        import string
        import time as _time
        deadline = _time.monotonic() + 10  # 10 s total cap
        binary_set = set(binary_names)  # O(1) membership test
        drive_roots = [f"{d}:\\" for d in string.ascii_uppercase]
        for drive in drive_roots:
            if not os.path.isdir(drive):
                continue
            try:
                for dirpath, _dirs, files in os.walk(drive):
                    if _time.monotonic() > deadline:
                        return None  # hard stop
                    for fname in files:
                        if fname in binary_set:
                            return os.path.join(dirpath, fname)
            except (PermissionError, OSError):
                continue

    # ── 7) Return "python-fallback" marker if llama-cpp-python is installed ───
    # This signals that we can use the Python API instead of the binary
    try:
        import llama_cpp
        return "python-fallback"
    except ImportError:
        pass

    return None


def _derive_llama_cpp_root() -> Optional[Path]:
    """
    Locate the llama.cpp source / install root by searching for the
    convert_hf_to_gguf.py script — the same logic GGUFExportManager uses.
    Returns the directory that contains the script, or None.
    """
    script_names = ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py", "convert.py"]
    # Include the directory that contains this script itself so that a
    # llama.cpp clone sitting beside NTTuner.py is found immediately.
    _script_dir = Path(sys.argv[0]).resolve().parent if sys.argv else Path.cwd()
    search_paths = [
        _script_dir / "llama.cpp",
        _script_dir,  # convert script right next to NTTuner
        Path.home() / "llama.cpp",
        Path("/usr/local/share/llama.cpp"),
        Path("/opt/llama.cpp"),
        Path.cwd() / "llama.cpp",
    ]
    for base in search_paths:
        for name in script_names:
            if (base / name).exists():
                return base
    # pip-installed llama_cpp package
    try:
        import llama_cpp
        _file = getattr(llama_cpp, "__file__", None)
        if _file:
            pkg = Path(_file).parent
            for name in script_names:
                if (pkg / name).exists():
                    return pkg
    except (ImportError, AttributeError, TypeError):
        pass
    return None


def find_llama_quantize_full() -> Optional[str]:
    """
    Public accessor – returns the cached quantize path.  The actual
    scan runs once on a background thread at import time; this just
    waits for it (instant if already done).
    """
    return _get_cached_quantize()


def find_llama_gguf_split() -> Optional[str]:
    """Find llama-gguf-split binary for splitting large models"""
    binary_names = ["llama-gguf-split", "gguf-split"]

    for name in binary_names:
        path = shutil.which(name)
        if path:
            return path

    return None


# ── Background-thread quantize-path cache ────────────────────────────────
# find_llama_quantize can take up to 10 s on Windows (full-drive walk).
# We run it exactly once, on a daemon thread, at import time.  Everything
# that needs the path calls _get_cached_quantize() which blocks only if
# the scan hasn't finished yet (usually already done by the time the user
# clicks anything).
import threading as _threading

_QUANTIZE_CACHE: Optional[str] = None
_QUANTIZE_CACHE_READY = _threading.Event()


def _run_quantize_scan():
    global _QUANTIZE_CACHE
    root = _derive_llama_cpp_root()
    _QUANTIZE_CACHE = find_llama_quantize(root)
    if not _QUANTIZE_CACHE:
        _QUANTIZE_CACHE = find_llama_quantize(None)  # full drive scan
    _QUANTIZE_CACHE_READY.set()


# Fire the scan immediately at import time (daemon = won't block exit)
_threading.Thread(target=_run_quantize_scan, daemon=True).start()


def _get_cached_quantize() -> Optional[str]:
    """Return the cached llama-quantize path.  Blocks at most ~10 s on
    first call (only if the background scan is still running)."""
    _QUANTIZE_CACHE_READY.wait(timeout=12)
    return _QUANTIZE_CACHE


# ────────────────────────────────────────────────────────────────────────
# End: GGUF export utilities
# ────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────
# Added: GGUF Export Configuration dataclass (2026)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class GGUFExportConfig:
    """Configuration for advanced GGUF export options"""
    use_advanced_export: bool = False
    quant_type: str = "Q4_K_M"
    preset: str = "Standard Quality (Q4_K_M)"
    export_all_quants: bool = False
    selected_quants: List[str] = field(default_factory=lambda: ["Q4_K_M"])
    imatrix_path: str = ""
    custom_flags: str = ""
    output_filename_pattern: str = "{model_name}-{quant_type}"
    skip_merge_lora_only: bool = False
    llama_quantize_path: str = ""
    auto_import_ollama: bool = True
    keep_f16_base: bool = False

    def get_effective_quants(self) -> List[str]:
        """Get the list of quantization types to export"""
        if self.export_all_quants:
            return ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
        elif self.preset in GGUF_PRESETS:
            return GGUF_PRESETS[self.preset]
        else:
            return self.selected_quants if self.selected_quants else [self.quant_type]


# ────────────────────────────────────────────────────────────────────────
# End: GGUF Export Configuration
# ────────────────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """Training configuration with validation"""
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_path: str = ""
    lora_rank: int = 32
    use_chronicals: bool = False  # New 2026 "Insane Fast Mode" flag
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    epochs: int = 1
    batch_size: int = 1
    grad_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 512
    output_name: str = "my-fine-tuned-model"
    output_dir: str = "./fine_tuned_output"
    quant_method: str = "q5_k_m"
    save_steps: int = 100
    logging_steps: int = 10
    # ────────────────────────────────────────────────────────────────────
    # Added: GGUF export config field (2026)
    # ────────────────────────────────────────────────────────────────────
    gguf_export: GGUFExportConfig = field(default_factory=GGUFExportConfig)

    # ────────────────────────────────────────────────────────────────────

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration"""
        if not self.base_model.strip():
            return False, "Base model cannot be empty"
        # Block Ollama colon-style names (e.g. "qwen2.5:0.5b") — they crash HF loader
        if ":" in self.base_model and "/" not in self.base_model:
            return False, (f"'{self.base_model}' is an Ollama pull name and cannot be loaded directly. "
                           f"Use the Download button first, then select the HuggingFace equivalent "
                           f"(e.g. Qwen/Qwen2.5-0.5B-Instruct).")
        if not self.dataset_path.strip():
            return False, "Dataset path cannot be empty"
        if not os.path.exists(self.dataset_path):
            return False, f"Dataset file not found: {self.dataset_path}"
        if self.lora_rank < 1 or self.lora_rank > 512:
            return False, "LoRA rank must be between 1 and 512"
        if self.epochs < 1:
            return False, "Epochs must be at least 1"
        if self.batch_size < 1:
            return False, "Batch size must be at least 1"
        if self.learning_rate <= 0:
            return False, "Learning rate must be positive"
        if self.max_seq_length < 128:
            return False, "Max sequence length must be at least 128"
        return True, "Configuration valid"

    def get_warnings(self) -> List[str]:
        """Get configuration warnings"""
        warnings = []
        if self.lora_rank > 128:
            warnings.append(f"High LoRA rank ({self.lora_rank}) may increase training time")
        if self.epochs > 3:
            warnings.append(f"Many epochs ({self.epochs}) may lead to overfitting")
        if self.batch_size * self.grad_accumulation > 32:
            warnings.append(f"Large effective batch size ({self.batch_size * self.grad_accumulation})")
        if self.learning_rate > 5e-4:
            warnings.append(f"High learning rate ({self.learning_rate:.2e}) may cause instability")
        if self.max_seq_length > 2048 and not HAS_GPU:
            warnings.append(f"Long sequences ({self.max_seq_length}) will be very slow on CPU")
        if HAS_GPU:
            estimated_vram = self.estimate_vram_usage()
            if estimated_vram > GPU_MEMORY:
                warnings.append(f"Estimated VRAM ({estimated_vram:.1f}GB) exceeds available ({GPU_MEMORY:.1f}GB)")
        return warnings

    def estimate_vram_usage(self) -> float:
        """Estimate VRAM usage in GB"""
        if not HAS_GPU:
            return 0.0
        base = 2.0
        model = 4.0
        batch = (self.batch_size * self.max_seq_length * 4) / (1024 ** 3)
        lora = (self.lora_rank * 2 * 0.001)
        grad = batch * self.grad_accumulation * 2
        return min(base + model + batch + lora + grad, GPU_MEMORY * 0.95)


# ═══════════════════════════════════════════════════════════════════════
# DATASET HANDLER - OPTIMIZED FOR NTCOMPANION OUTPUT
# ═══════════════════════════════════════════════════════════════════════

class DatasetHandler:
    """Handle NTCompanion datasets and other formats"""

    @staticmethod
    def detect_format(filepath: str) -> str:
        """Detect dataset format"""
        ext = Path(filepath).suffix.lower()
        if ext == '.jsonl':
            return 'jsonl'
        elif ext == '.json':
            return 'json'
        elif ext == '.csv':
            return 'csv'
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            if first_line.startswith('{'):
                try:
                    json.loads(first_line)
                    return 'jsonl'
                except:
                    return 'json'
            elif first_line.startswith('['):
                return 'json'
            else:
                return 'csv'
        except:
            return 'unknown'

    @staticmethod
    def load_dataset(filepath: str) -> Tuple[Optional[List[Dict]], str]:
        """
        Load dataset - optimized for NTCompanion JSONL format
        Expected format: {"text": "<formatted conversation>"}
        """
        try:
            format_type = DatasetHandler.detect_format(filepath)
            data = []

            if format_type == 'jsonl':
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))

            elif format_type == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        data = loaded
                    else:
                        data = [loaded]

            elif format_type == 'csv':
                import csv
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)

            else:
                return None, f"Unknown format: {format_type}"

            if not data:
                return None, "Dataset is empty"

            return data, "OK"

        except Exception as e:
            return None, f"Failed to load dataset: {str(e)}"

    @staticmethod
    def validate_dataset(data: List[Dict]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate dataset structure - works with NTCompanion format"""
        if not data:
            return False, "Dataset is empty", {}

        stats = {
            "total_entries": len(data),
            "avg_length": 0,
            "min_length": float('inf'),
            "max_length": 0,
            "has_text_field": False,
            "format_issues": []
        }

        text_lengths = []

        for i, entry in enumerate(data[:100]):  # Sample first 100
            if not isinstance(entry, dict):
                stats["format_issues"].append(f"Entry {i} is not a dictionary")
                continue

            # NTCompanion format: {"text": "..."}
            text = entry.get('text', '') or entry.get('content', '') or entry.get('prompt', '')

            if not text:
                stats["format_issues"].append(f"Entry {i} has no text content")
                continue

            stats["has_text_field"] = True
            text_lengths.append(len(text))

        if text_lengths:
            stats["avg_length"] = sum(text_lengths) / len(text_lengths)
            stats["min_length"] = min(text_lengths)
            stats["max_length"] = max(text_lengths)

        if not stats["has_text_field"]:
            return False, "No valid text fields found in dataset", stats

        if len(stats["format_issues"]) > len(data) * 0.5:
            return False, "More than 50% of entries have format issues", stats

        return True, "Dataset valid", stats

    @staticmethod
    def preview_entries(data: List[Dict], count: int = 3) -> List[str]:
        """Generate preview of dataset entries"""
        previews = []
        for i, entry in enumerate(data[:count]):
            text = entry.get('text', '') or entry.get('content', '') or entry.get('prompt', '')
            preview = f"Entry {i + 1}:\n"
            preview += f"  Length: {len(text)} chars\n"
            snippet = text[:300] + "..." if len(text) > 300 else text
            preview += f"  Preview: {snippet}\n"
            previews.append(preview)
        return previews


# ════════════════════════════════════════════════════════════════════════════
# REWORKED GGUF EXPORT SYSTEM - More Reliable and User-Friendly
# ════════════════════════════════════════════════════════════════════════════

import subprocess
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


# ────────────────────────────────────────────────────────────────────────────
# GGUF CONVERSION BACKENDS - Multiple fallback options
# ────────────────────────────────────────────────────────────────────────────

class GGUFConversionBackend:
    """Abstract base for GGUF conversion methods"""

    def __init__(self, log_callback):
        self.log = log_callback

    def is_available(self) -> bool:
        """Check if this backend is available"""
        raise NotImplementedError

    def convert_to_f16(self, model_path: Path, output_file: Path) -> Tuple[bool, str]:
        """Convert HF model to F16 GGUF"""
        raise NotImplementedError

    def quantize(self, input_gguf: Path, output_file: Path, quant_type: str,
                 imatrix_path: Optional[str] = None) -> Tuple[bool, str]:
        """Quantize F16 GGUF to specified type"""
        raise NotImplementedError


class LlamaCppBackend(GGUFConversionBackend):
    """llama.cpp python scripts backend"""

    def __init__(self, log_callback):
        super().__init__(log_callback)
        self.convert_script = None
        self.quantize_binary = None
        self._scan_tools()

    def _scan_tools(self):
        """Scan for llama.cpp tools"""
        # Find convert script
        script_names = [
            "convert_hf_to_gguf.py",
            "convert-hf-to-gguf.py",
            "convert.py"
        ]

        search_paths = [
            Path.cwd() / "llama.cpp",
            Path.home() / "llama.cpp",
            Path("/usr/local/share/llama.cpp"),
            Path("/opt/llama.cpp"),
            Path(sys.executable).parent.parent / "llama.cpp",  # Python venv
        ]

        # Check if llama-cpp-python is installed
        try:
            import llama_cpp
            pkg_path = Path(llama_cpp.__file__).parent
            search_paths.insert(0, pkg_path)
            search_paths.insert(0, pkg_path.parent / "llama.cpp")
        except ImportError:
            pass

        for base in search_paths:
            if not base.exists():
                continue
            for name in script_names:
                script = base / name
                if script.exists():
                    self.convert_script = script
                    self.log(f"[OK] Found convert script: {script}\n")
                    break
            if self.convert_script:
                break

        # Find quantize binary
        binary_names = ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]

        # Check PATH first
        for name in binary_names:
            path = shutil.which(name)
            if path:
                self.quantize_binary = Path(path)
                self.log(f"[OK] Found quantize binary in PATH: {path}\n")
                return

        # Check build directories
        for base in search_paths:
            if not base.exists():
                continue

            # Common build locations
            build_dirs = [
                base / "build" / "bin",
                base / "build",
                base / "build" / "Release",
                base / "build" / "Debug",
                base / "bin",
            ]

            for build_dir in build_dirs:
                if not build_dir.exists():
                    continue
                for name in binary_names:
                    binary = build_dir / name
                    if binary.exists() and binary.is_file():
                        self.quantize_binary = binary
                        self.log(f"[OK] Found quantize binary: {binary}\n")
                        return

    def is_available(self) -> bool:
        """Check if llama.cpp tools are available"""
        return self.convert_script is not None or self.quantize_binary is not None

    def convert_to_f16(self, model_path: Path, output_file: Path) -> Tuple[bool, str]:
        """Convert using llama.cpp python script"""
        if not self.convert_script:
            return False, "convert script not found"

        self.log(f"Converting with llama.cpp script: {self.convert_script.name}\n")

        try:
            cmd = [
                sys.executable,
                str(self.convert_script),
                str(model_path),
                "--outfile", str(output_file),
                "--outtype", "f16",
            ]

            self.log(f"Running: {' '.join(cmd)}\n")

            result = safe_subprocess_run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=self.convert_script.parent  # Run in llama.cpp directory
            )

            if result.returncode == 0 and output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                self.log(f"✓ Conversion successful: {size_mb:.1f} MB\n")
                return True, "OK"
            else:
                error = result.stderr if result.stderr else result.stdout
                self.log(f"✗ Conversion failed:\n{error}\n")
                return False, error or "Unknown error"

        except subprocess.TimeoutExpired:
            return False, "Conversion timed out (30 min limit)"
        except Exception as e:
            return False, f"Exception: {str(e)}"

    def quantize(self, input_gguf: Path, output_file: Path, quant_type: str,
                 imatrix_path: Optional[str] = None) -> Tuple[bool, str]:
        """Quantize using llama-quantize binary"""
        if not self.quantize_binary:
            return False, "quantize binary not found"

        self.log(f"Quantizing to {quant_type}...\n")

        try:
            cmd = [str(self.quantize_binary), str(input_gguf), str(output_file), quant_type]

            if imatrix_path and Path(imatrix_path).exists():
                cmd.extend(["--imatrix", imatrix_path])
                self.log(f"  Using importance matrix: {imatrix_path}\n")

            self.log(f"Running: {' '.join(cmd)}\n")

            result = safe_subprocess_run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0 and output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                self.log(f"✓ Quantization successful: {size_mb:.1f} MB\n")
                return True, "OK"
            else:
                error = result.stderr if result.stderr else result.stdout
                self.log(f"✗ Quantization failed:\n{error}\n")
                return False, error or "Unknown error"

        except subprocess.TimeoutExpired:
            return False, "Quantization timed out (1 hour limit)"
        except Exception as e:
            return False, f"Exception: {str(e)}"


class UnslothBackend(GGUFConversionBackend):
    """Unsloth library backend"""

    def is_available(self) -> bool:
        try:
            from unsloth import FastLanguageModel
            return True
        except ImportError:
            return False

    def convert_to_f16(self, model_path: Path, output_file: Path) -> Tuple[bool, str]:
        """Convert using Unsloth's built-in GGUF export"""
        self.log("Converting with Unsloth...\n")

        try:
            from unsloth import FastLanguageModel

            # Load model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_path),
                max_seq_length=2048,
                load_in_4bit=False,
            )

            # Export to GGUF
            output_dir = output_file.parent
            model.save_pretrained_gguf(
                str(output_dir),
                tokenizer,
                quantization_method="f16",
            )

            # Find the generated file
            gguf_files = list(output_dir.glob("*.gguf"))
            if gguf_files:
                # Rename to our desired output name
                gguf_files[0].rename(output_file)
                size_mb = output_file.stat().st_size / (1024 * 1024)
                self.log(f"✓ Unsloth conversion successful: {size_mb:.1f} MB\n")
                return True, "OK"
            else:
                return False, "No GGUF file generated"

        except Exception as e:
            self.log(f"✗ Unsloth conversion failed: {str(e)}\n")
            return False, str(e)

    def quantize(self, input_gguf: Path, output_file: Path, quant_type: str,
                 imatrix_path: Optional[str] = None) -> Tuple[bool, str]:
        """Unsloth can also quantize directly"""
        self.log("Quantizing with Unsloth...\n")

        try:
            from unsloth import FastLanguageModel

            # Unsloth can save directly to quantized GGUF
            # Load the model from the original HF path (we need the model_path)
            # This is a limitation - need the original model
            return False, "Unsloth quantization requires original model (not supported in this flow)"

        except Exception as e:
            return False, str(e)


class LlamaCppPythonBackend(GGUFConversionBackend):
    """llama-cpp-python package backend"""

    def is_available(self) -> bool:
        try:
            import llama_cpp
            return True
        except ImportError:
            return False

    def convert_to_f16(self, model_path: Path, output_file: Path) -> Tuple[bool, str]:
        """Convert using llama-cpp-python if it has conversion support"""
        try:
            import llama_cpp

            # Check if llama-cpp-python has conversion utilities
            if hasattr(llama_cpp, 'convert_hf_to_gguf'):
                self.log("Converting with llama-cpp-python...\n")
                llama_cpp.convert_hf_to_gguf(
                    str(model_path),
                    str(output_file),
                    ftype="f16"
                )

                if output_file.exists():
                    size_mb = output_file.stat().st_size / (1024 * 1024)
                    self.log(f"✓ Conversion successful: {size_mb:.1f} MB\n")
                    return True, "OK"
                else:
                    return False, "No output file generated"
            else:
                return False, "llama-cpp-python doesn't support conversion"

        except Exception as e:
            return False, str(e)

    def quantize(self, input_gguf: Path, output_file: Path, quant_type: str,
                 imatrix_path: Optional[str] = None) -> Tuple[bool, str]:
        """Quantize using llama-cpp-python if supported"""
        try:
            import llama_cpp

            # Load and quantize
            if hasattr(llama_cpp, 'quantize_gguf'):
                self.log(f"Quantizing to {quant_type} with llama-cpp-python...\n")
                llama_cpp.quantize_gguf(
                    str(input_gguf),
                    str(output_file),
                    qtype=quant_type,
                    imatrix=imatrix_path if imatrix_path else None
                )

                if output_file.exists():
                    size_mb = output_file.stat().st_size / (1024 * 1024)
                    self.log(f"✓ Quantization successful: {size_mb:.1f} MB\n")
                    return True, "OK"
                else:
                    return False, "No output file generated"
            else:
                return False, "llama-cpp-python doesn't support quantization"

        except Exception as e:
            return False, str(e)


# ────────────────────────────────────────────────────────────────────────────
# MAIN GGUF EXPORT MANAGER - Orchestrates backends
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class GGUFExportConfig:
    """Configuration for GGUF export"""
    use_advanced_export: bool = False
    quant_type: str = "Q4_K_M"
    preset: str = "Standard Quality (Q4_K_M)"
    export_all_quants: bool = False
    selected_quants: List[str] = field(default_factory=lambda: ["Q4_K_M"])
    imatrix_path: str = ""
    custom_flags: str = ""
    output_filename_pattern: str = "{model_name}-{quant_type}"
    skip_merge_lora_only: bool = False
    llama_quantize_path: str = ""
    auto_import_ollama: bool = True
    keep_f16_base: bool = False

    def get_effective_quants(self) -> List[str]:
        """Get the list of quantization types to export"""
        # Predefined presets
        GGUF_PRESETS = {
            "Standard Quality (Q4_K_M)": ["Q4_K_M"],
            "High Quality (Q5_K_M)": ["Q5_K_M"],
            "Best Quality (Q6_K)": ["Q6_K"],
            "Maximum Quality (Q8_0)": ["Q8_0"],
            "Full Precision (F16)": ["F16"],
            "Small Size (Q3_K_M)": ["Q3_K_M"],
            "Tiny Size (Q2_K)": ["Q2_K"],
            "IQ Optimized (IQ4_XS)": ["IQ4_XS"],
            "All K-Quants": ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
            "All Common": ["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
            "Size Ladder (Q2→Q8)": ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
            "IQ Series": ["IQ2_M", "IQ3_M", "IQ4_XS", "IQ4_NL"],
        }

        if self.export_all_quants:
            return ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
        elif self.preset in GGUF_PRESETS:
            return GGUF_PRESETS[self.preset]
        else:
            return self.selected_quants if self.selected_quants else [self.quant_type]


class ImprovedGGUFExportManager:
    """
    Improved GGUF Export Manager with multiple backend support
    Automatically falls back through available methods
    """

    def __init__(self, log_callback):
        self.log = log_callback
        self.backends = []
        self._init_backends()

    def _init_backends(self):
        """Initialize all available backends in priority order"""
        self.log("Initializing GGUF conversion backends...\n")

        # Priority order: llama.cpp > Unsloth > llama-cpp-python
        backends_to_try = [
            ("llama.cpp", LlamaCppBackend),
            ("Unsloth", UnslothBackend),
            ("llama-cpp-python", LlamaCppPythonBackend),
        ]

        for name, backend_class in backends_to_try:
            try:
                backend = backend_class(self.log)
                if backend.is_available():
                    self.backends.append(backend)
                    self.log(f"  ✓ {name} backend available\n")
                else:
                    self.log(f"  ✗ {name} backend not available\n")
            except MemoryError:
                self.log(f"  ✗ {name} backend: Out of memory\n")
            except OSError as e:
                self.log(f"  ✗ {name} backend: OS error - {e}\n")
            except Exception as e:
                self.log(f"  ✗ {name} backend error: {e}\n")
                import traceback
                self.log(f"     {traceback.format_exc()[:200]}\n")

        if not self.backends:
            self.log("\n[WARNING] No GGUF backends available!\n")
            self.log("Install one of:\n")
            self.log("  - llama.cpp: git clone + build\n")
            self.log("  - pip install unsloth\n")
            self.log("  - pip install llama-cpp-python\n")
        else:
            self.log(f"\n{len(self.backends)} backend(s) ready\n")

    def convert_to_f16(self, model_path: Path, output_file: Path) -> Tuple[bool, str, Optional[Path]]:
        """
        Convert HF model to F16 GGUF, trying all available backends
        """
        self.log(f"\nConverting {model_path.name} to F16 GGUF...\n")

        if not self.backends:
            return False, "No conversion backends available", None

        # Try each backend until one succeeds
        for backend in self.backends:
            self.log(f"Trying {backend.__class__.__name__}...\n")
            try:
                success, error = backend.convert_to_f16(model_path, output_file)
                if success:
                    return True, "OK", output_file
                else:
                    self.log(f"  Failed: {error}\n")
            except Exception as e:
                self.log(f"  Exception: {str(e)}\n")

        return False, "All conversion methods failed", None

    def quantize(self, input_gguf: Path, output_file: Path, quant_type: str,
                 imatrix_path: Optional[str] = None) -> Tuple[bool, str, Optional[Path]]:
        """
        Quantize F16 GGUF, trying available backends
        """
        self.log(f"\nQuantizing to {quant_type}...\n")

        if not self.backends:
            return False, "No quantization backends available", None

        # Try each backend
        for backend in self.backends:
            try:
                success, error = backend.quantize(input_gguf, output_file, quant_type, imatrix_path)
                if success:
                    return True, "OK", output_file
                else:
                    self.log(f"  {backend.__class__.__name__} failed: {error}\n")
            except Exception as e:
                self.log(f"  {backend.__class__.__name__} exception: {str(e)}\n")

        return False, "All quantization methods failed", None

    def run_full_export(self, model_path: Path, output_path: Path,
                        gguf_config: GGUFExportConfig, model_name: str) -> Dict[str, Any]:
        """
        Run complete GGUF export with all configured options
        """
        results = {
            "success": False,
            "f16_path": None,
            "quantized_files": {},
            "errors": [],
        }

        output_path.mkdir(parents=True, exist_ok=True)

        self.log("=" * 70 + "\n")
        self.log("GGUF EXPORT STARTING\n")
        self.log("=" * 70 + "\n")
        self.log(f"Model: {model_path}\n")
        self.log(f"Output: {output_path}\n\n")

        # Step 1: Convert to F16
        f16_file = output_path / f"{model_name}-f16.gguf"
        success, error, f16_path = self.convert_to_f16(model_path, f16_file)

        if not success:
            results["errors"].append(f"F16 conversion: {error}")
            self.log("\n" + "=" * 70 + "\n")
            self.log("EXPORT FAILED\n")
            self.log("=" * 70 + "\n")
            return results

        results["f16_path"] = f16_path
        self.log("\n✓ F16 base file ready\n\n")

        # Step 2: Quantize to requested types
        quant_list = gguf_config.get_effective_quants()
        self.log(f"Quantization targets: {', '.join(quant_list)}\n\n")

        for quant_type in quant_list:
            if quant_type.upper() in ["F16", "FP16"]:
                # F16 already exists
                results["quantized_files"]["F16"] = f16_path
                continue

            # Generate output filename
            output_filename = gguf_config.output_filename_pattern.format(
                model_name=model_name,
                quant_type=quant_type.lower()
            )
            if not output_filename.endswith(".gguf"):
                output_filename += ".gguf"

            quant_file = output_path / output_filename

            success, error, quant_path = self.quantize(
                f16_path, quant_file, quant_type.upper(),
                gguf_config.imatrix_path if gguf_config.imatrix_path else None
            )

            if success:
                results["quantized_files"][quant_type] = quant_path

                # Auto-import to Ollama if requested
                if gguf_config.auto_import_ollama:
                    ollama_name = f"{model_name}-{quant_type.lower()}"
                    self.import_to_ollama(quant_path, ollama_name)
            else:
                results["errors"].append(f"{quant_type}: {error}")

        # Clean up F16 if not keeping it
        if not gguf_config.keep_f16_base and f16_path and f16_path.exists():
            if "F16" not in quant_list:
                try:
                    f16_path.unlink()
                    self.log("\nCleaned up intermediate F16 file\n")
                except Exception as e:
                    self.log(f"Warning: Could not delete F16 file: {e}\n")

        results["success"] = len(results["quantized_files"]) > 0

        # Summary
        self.log("\n" + "=" * 70 + "\n")
        self.log("EXPORT SUMMARY\n")
        self.log("=" * 70 + "\n")
        self.log(f"Successful: {len(results['quantized_files'])}/{len(quant_list)}\n\n")

        for qtype, path in results["quantized_files"].items():
            if path and path.exists():
                size_gb = path.stat().st_size / (1024 ** 3)
                self.log(f"  ✓ {qtype:8s} {path.name:40s} ({size_gb:.2f} GB)\n")

        if results["errors"]:
            self.log(f"\nErrors: {len(results['errors'])}\n")
            for error in results["errors"]:
                self.log(f"  ✗ {error}\n")

        self.log("=" * 70 + "\n")

        return results

    def import_to_ollama(self, gguf_path: Path, model_name: str) -> Tuple[bool, str]:
        """Import GGUF file to Ollama"""
        self.log(f"  Importing to Ollama as '{model_name}'...\n")

        try:
            # Create Modelfile
            modelfile_content = f"""FROM {gguf_path.absolute()}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
            modelfile_path = gguf_path.parent / f"Modelfile.{model_name}"
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)

            # Run ollama create
            cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            result = safe_subprocess_run(cmd, capture_output=True, text=True, timeout=300)

            # Clean up Modelfile
            try:
                modelfile_path.unlink()
            except:
                pass

            if result.returncode == 0:
                self.log(f"    ✓ Imported to Ollama\n")
                return True, "OK"
            else:
                error = result.stderr or result.stdout or "Unknown error"
                self.log(f"    ✗ Ollama import failed: {error}\n")
                return False, error

        except Exception as e:
            return False, str(e)


# ════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# TRAINING MANAGER
# ═══════════════════════════════════════════════════════════════════════

class TrainingManager:
    """Manage training with comprehensive monitoring"""

    def __init__(self, log_callback):
        self.log = log_callback
        self.is_training = False
        self.should_stop = False
        self.trainer = None
        self.model = None
        self.tokenizer = None
        self.start_time = None
        self.current_step = 0
        self.total_steps = 0
        self.last_loss = None
        # ────────────────────────────────────────────────────────────────
        # Added: GGUF export manager instance (2026) - lazy initialized
        # ────────────────────────────────────────────────────────────────
        self._gguf_manager = None  # Initialized on first use via property
        # ────────────────────────────────────────────────────────────────

    @property
    def gguf_manager(self):
        """Lazy-loaded GGUF manager"""
        if self._gguf_manager is None:
            self._gguf_manager = ImprovedGGUFExportManager(self.log)
        return self._gguf_manager

    @gguf_manager.setter
    def gguf_manager(self, value):
        self._gguf_manager = value

    def prepare_dataset(self, config: TrainingConfig) -> Tuple[Optional[Any], str]:
        """Prepare dataset for training with chat-template auto-detection and auto-wrapping."""
        self.log("Loading dataset...\n")

        data, error = DatasetHandler.load_dataset(config.dataset_path)
        if not data:
            return None, error

        self.log(f"Loaded {len(data)} entries\n")

        valid, message, stats = DatasetHandler.validate_dataset(data)
        if not valid:
            return None, message

        self.log(f"Dataset validated:\n")
        self.log(f"  Total entries: {stats['total_entries']}\n")
        self.log(f"  Avg length: {stats['avg_length']:.0f} chars\n")
        self.log(f"  Range: {stats['min_length']}-{stats['max_length']} chars\n")

        if stats['format_issues']:
            self.log(f"  Warning: {len(stats['format_issues'])} entries with issues\n")

        # ── Chat-template detection & auto-wrap ──────────────────────────
        template_info = detect_template_for_model(config.base_model)

        if template_info:
            tmpl_name, markers, wrap_fn = template_info
            self.log(f"\n[TEMPLATE] Detected: {tmpl_name}\n")
            self.log(f"  Expected markers: {markers}\n")

            # Sample the first entry to check whether markers are already present
            sample_text = (data[0].get('text', '') or
                           data[0].get('content', '') or
                           data[0].get('prompt', ''))

            if dataset_has_markers(sample_text, markers):
                self.log(f"  ✓ Dataset already contains {tmpl_name} markers — using as-is\n")
            else:
                # Check whether the entries have separate system/user/assistant fields
                # so we can auto-wrap them
                has_structured = any(
                    k in data[0] for k in ('user', 'question', 'input', 'prompt')
                ) and any(
                    k in data[0] for k in ('assistant', 'answer', 'output', 'response', 'completion')
                )

                if has_structured:
                    self.log(f"  ⟳ Auto-wrapping {len(data)} entries with {tmpl_name} template...\n")
                    wrapped = []
                    for entry in data:
                        sys_msg = (entry.get('system', '') or
                                   entry.get('system_prompt', '') or
                                   'You are a helpful assistant.')
                        user_msg = (entry.get('user', '') or
                                    entry.get('question', '') or
                                    entry.get('input', '') or
                                    entry.get('prompt', ''))
                        asst_msg = (entry.get('assistant', '') or
                                    entry.get('answer', '') or
                                    entry.get('output', '') or
                                    entry.get('response', '') or
                                    entry.get('completion', ''))
                        if user_msg and asst_msg:
                            wrapped.append({'text': wrap_fn(sys_msg, user_msg, asst_msg)})
                        else:
                            # Fallback: keep raw text
                            raw = entry.get('text', '') or entry.get('content', '')
                            if raw:
                                wrapped.append({'text': raw})
                    data = wrapped
                    self.log(f"  ✓ Auto-wrap complete — {len(data)} entries ready\n")
                else:
                    # Single 'text' field but no markers — loud warning
                    self.log(f"  ⚠ WARNING: Dataset 'text' field does NOT contain {tmpl_name} markers\n")
                    self.log(f"    and no structured fields (user/assistant) found for auto-wrapping.\n")
                    self.log(f"    Expected markers: {markers}\n")
                    self.log(f"    Training will proceed but quality may be poor.\n")
                    self.log(f"    → Format your JSONL with the correct chat template before training.\n\n")
        else:
            self.log(f"\n[TEMPLATE] No known template for '{config.base_model}' — using dataset as-is\n")
        # ── end template logic ────────────────────────────────────────────

        try:
            processed_data = []
            for entry in data:
                text = entry.get('text', '') or entry.get('content', '') or entry.get('prompt', '')
                if text:
                    processed_data.append({'text': text})

            dataset = Dataset.from_list(processed_data)
            self.log(f"Dataset prepared: {len(dataset)} training examples\n")
            return dataset, "OK"

        except Exception as e:
            return None, f"Dataset preparation failed: {str(e)}"

    def load_model(self, config: TrainingConfig) -> Tuple[bool, str]:
        """Load model with enhanced GPU support"""
        self.log(f"Loading model: {config.base_model}\n")
        self.log(f"Using backend: {GPU_TYPE} ({GPU_BACKEND})\n")

        try:
            if HAS_UNSLOTH and GPU_BACKEND == "cuda":
                self.log("Using Unsloth (optimized for NVIDIA GPUs)...\n")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.base_model,
                    max_seq_length=config.max_seq_length,
                    dtype=None,
                    load_in_4bit=True,
                )
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    use_gradient_checkpointing="unsloth",
                )
            else:
                backend_name = GPU_TYPE if HAS_GPU else "CPU"
                self.log(f"Using standard Transformers ({backend_name})...\n")

                self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                device_map = get_device_map()
                torch_dtype = get_torch_dtype()

                self.log(f"Device: {device_map}, dtype: {torch_dtype}\n")

                self.model = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                )

                lora_config = LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, lora_config)

            self.log("Model loaded successfully\n")

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self.log(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n")

            return True, "OK"

        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            self.log(f"ERROR: {error_msg}\n")
            self.log(f"{traceback.format_exc()}\n")
            return False, error_msg

    def train(self, config: TrainingConfig) -> bool:
        """Execute training with monitoring"""
        if not DEPS_AVAILABLE:
            self.log("ERROR: Missing dependencies\n")
            return False

        self.is_training = True
        self.should_stop = False
        self.start_time = time.time()

        # Initialize progress bar
        if dpg.does_item_exist("training_progress_bar"):
            dpg.set_value("training_progress_bar", 0.0)
        if dpg.does_item_exist("progress_text"):
            dpg.set_value("progress_text", "Initializing training...")

        try:
            dataset, error = self.prepare_dataset(config)
            if not dataset:
                self.log(f"ERROR: {error}\n")
                return False

            success, error = self.load_model(config)
            if not success:
                return False

            self.total_steps = (len(dataset) // (config.batch_size * config.grad_accumulation)) * config.epochs
            self.log(f"Total training steps: {self.total_steps}\n")

            optim = "adamw_8bit" if GPU_BACKEND == "cuda" else "adamw_torch"

            training_args = TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=config.epochs,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.grad_accumulation,
                learning_rate=config.learning_rate,
                warmup_steps=config.warmup_steps,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                save_total_limit=3,
                fp16=(GPU_BACKEND in ["cuda", "hip"]),
                optim=optim,
                report_to="none",
            )

            class ProgressCallback(TrainerCallback):
                def __init__(self, manager):
                    self.manager = manager

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        self.manager.current_step = state.global_step
                        if 'loss' in logs:
                            self.manager.last_loss = logs['loss']
                        elapsed = time.time() - self.manager.start_time
                        if state.global_step > 0:
                            # Calculate progress
                            progress = state.global_step / self.manager.total_steps
                            time_per_step = elapsed / state.global_step
                            remaining = self.manager.total_steps - state.global_step
                            eta = remaining * time_per_step

                            # Update progress bar and text in GUI
                            if dpg.does_item_exist("training_progress_bar"):
                                dpg.set_value("training_progress_bar", progress)
                            if dpg.does_item_exist("progress_text"):
                                progress_info = (
                                    f"Step {state.global_step}/{self.manager.total_steps} "
                                    f"({progress * 100:.1f}%) | Loss: {logs.get('loss', 0):.4f} | "
                                    f"ETA: {format_time(eta)}"
                                )
                                dpg.set_value("progress_text", progress_info)

                            # Also log to text window
                            self.manager.log(
                                f"Step {state.global_step}/{self.manager.total_steps} | "
                                f"Loss: {logs.get('loss', 0):.4f} | ETA: {format_time(eta)}\n"
                            )

                def on_epoch_end(self, args, state, control, **kwargs):
                    self.manager.log(f"Epoch {int(state.epoch)} complete\n")

            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))

            # Prepare dataset in standard format for SFTTrainer
            # Convert to simple text list format that works across TRL versions
            formatted_dataset = dataset.map(
                lambda x: {"text": x["text"]},
                remove_columns=dataset.column_names
            )

            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=formatted_dataset,
                args=training_args,
                callbacks=[ProgressCallback(self)],
            )

            self.log("=" * 60 + "\n")
            self.log("Training started\n")
            self.log("=" * 60 + "\n")

            self.trainer.train()

            if not self.should_stop:
                # Mark progress as complete
                if dpg.does_item_exist("training_progress_bar"):
                    dpg.set_value("training_progress_bar", 1.0)
                if dpg.does_item_exist("progress_text"):
                    dpg.set_value("progress_text", "Training Complete! ✓")

                self.log("=" * 60 + "\n")
                self.log("Training complete!\n")
                self.log("=" * 60 + "\n")
                self.save_model(config)

            return True

        except Exception as e:
            # Reset progress bar on error
            if dpg.does_item_exist("training_progress_bar"):
                dpg.set_value("training_progress_bar", 0.0)
            if dpg.does_item_exist("progress_text"):
                dpg.set_value("progress_text", "Training Failed ✗")

            self.log(f"ERROR: Training failed\n{str(e)}\n{traceback.format_exc()}\n")
            return False

        finally:
            self.is_training = False
            self.cleanup()

    def merge_lora_to_base(self, config: TrainingConfig, adapter_path: Path) -> Optional[Path]:
        """
        Merge LoRA adapter weights into the base model and save as a full
        model directory (config.json + tokenizer + all weight shards).
        convert_hf_to_gguf.py REQUIRES this — an adapter-only directory
        does not contain config.json and will crash the converter.

        Returns the path to the merged model directory, or None on failure.
        """
        merged_path = adapter_path / "merged"
        self.log(f"Merging LoRA adapter into base model → {merged_path}\n")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # Load base model at full precision on CPU to avoid OOM during merge
            self.log(f"  Loading base model: {config.base_model}\n")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=get_torch_dtype(),
                device_map="cpu",
            )

            # Load the tokenizer that was actually used during training.
            # If a pad_token (or any other token) was added, the adapter's
            # embed_tokens / lm_head will be larger than the vanilla base model.
            # We MUST resize the base model BEFORE overlaying the adapter —
            # otherwise PeftModel.from_pretrained raises a size-mismatch error.
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
            if len(tokenizer) != base_model.config.vocab_size:
                self.log(
                    f"  Resizing embeddings: base={base_model.config.vocab_size} "
                    f"→ adapter tokenizer={len(tokenizer)}\n"
                )
                base_model.resize_token_embeddings(len(tokenizer))

            # Overlay the LoRA adapter
            self.log("  Applying LoRA adapter...\n")
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

            # Merge weights and drop the adapter wrapper
            self.log("  Merging weights (this may take a moment)...\n")
            merged_model = peft_model.merge_and_unload()

            # Persist the full merged model
            merged_path.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(merged_path))
            tokenizer.save_pretrained(str(merged_path))

            self.log(f"  Merged model saved to: {merged_path}\n")

            # Sanity-check: config.json must now exist
            if not (merged_path / "config.json").exists():
                self.log("  [!] WARNING: config.json missing after merge — export may fail\n")

            # Free memory
            del merged_model, peft_model, base_model
            if HAS_TORCH and HAS_GPU:
                if GPU_BACKEND == "cuda":
                    torch.cuda.empty_cache()
                elif GPU_BACKEND == "mps":
                    torch.mps.empty_cache()

            return merged_path

        except Exception as e:
            self.log(f"  ERROR during merge: {str(e)}\n")
            self.log(f"  {traceback.format_exc()}\n")
            return None

    def save_model(self, config: TrainingConfig):
        """Save LoRA adapter, merge into base model, then export to GGUF"""
        try:
            self.log("Saving model...\n")
            output_path = Path(config.output_dir) / config.output_name
            output_path.mkdir(parents=True, exist_ok=True)

            # Save the LoRA adapter (adapter_config.json + weights only)
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
            self.log(f"Adapter saved to: {output_path}\n")

            self.create_manifest(config, output_path)

            # ────────────────────────────────────────────────────────────────
            # Merge LoRA → full model, then hand the MERGED path to the
            # GGUF exporter so convert_hf_to_gguf.py gets a valid directory
            # with config.json and all weight shards.
            # ────────────────────────────────────────────────────────────────
            merged_path = self.merge_lora_to_base(config, output_path)
            if merged_path:
                self._run_gguf_export(config, merged_path)
            else:
                self.log("[!] Merge failed — skipping GGUF export.\n")
                self.log("    You can still use the adapter in output_path.\n")
            # ────────────────────────────────────────────────────────────────

        except Exception as e:
            self.log(f"Save error: {str(e)}\n")

    # ────────────────────────────────────────────────────────────────────────
    # Added: GGUF export method (2026)
    # ────────────────────────────────────────────────────────────────────────
    def _run_gguf_export(self, config: TrainingConfig, model_path: Path):
        """Run GGUF export based on configuration"""
        gguf_config = config.gguf_export

        if gguf_config.use_advanced_export:
            # Use advanced GGUF export
            self.log("\nStarting Advanced GGUF Export...\n")
            gguf_output_path = model_path / "gguf"

            results = self.gguf_manager.run_full_export(
                model_path=model_path,
                output_path=gguf_output_path,
                gguf_config=gguf_config,
                model_name=config.output_name
            )

            # Auto-import to Ollama if enabled and successful
            if results["success"] and gguf_config.auto_import_ollama:
                # Pick the best quantized file for Ollama import
                quant_priority = ["Q4_K_M", "Q5_K_M", "Q4_0", "Q6_K", "Q8_0", "F16"]
                import_path = None

                for qtype in quant_priority:
                    if qtype in results["quantized_files"]:
                        import_path = results["quantized_files"][qtype]
                        break

                if not import_path and results["quantized_files"]:
                    import_path = list(results["quantized_files"].values())[0]

                if import_path:
                    self.gguf_manager.import_to_ollama(import_path, config.output_name)
        else:
            # Use default/legacy GGUF export behavior
            self._run_default_gguf_export(config, model_path)

    def _run_default_gguf_export(self, config: TrainingConfig, model_path: Path):
        """Default GGUF export (preserves original behavior)"""
        self.log("\nRunning default GGUF export...\n")

        try:
            if HAS_UNSLOTH:
                # Unsloth built-in export
                from unsloth import FastLanguageModel

                gguf_path = model_path / "gguf"
                gguf_path.mkdir(exist_ok=True)

                self.log(f"Exporting to GGUF ({config.quant_method})...\n")

                # Reload model for export
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=str(model_path),
                    max_seq_length=config.max_seq_length,
                    load_in_4bit=False,
                )

                model.save_pretrained_gguf(
                    str(gguf_path),
                    tokenizer,
                    quantization_method=config.quant_method.lower(),
                )

                self.log(f"GGUF exported to: {gguf_path}\n")

                # Auto-import to Ollama
                gguf_files = list(gguf_path.glob("*.gguf"))
                if gguf_files:
                    self.gguf_manager.import_to_ollama(gguf_files[0], config.output_name)

            else:
                self.log("Unsloth not available - skipping default GGUF export\n")
                self.log("Enable 'Advanced GGUF Export' for llama.cpp-based export\n")

        except Exception as e:
            self.log(f"Default GGUF export failed: {str(e)}\n")
            self.log("Try enabling 'Advanced GGUF Export' with llama.cpp path\n")

    # ────────────────────────────────────────────────────────────────────────
    # End: GGUF export methods
    # ────────────────────────────────────────────────────────────────────────

    def create_manifest(self, config: TrainingConfig, output_path: Path):
        """Create training manifest"""
        try:
            manifest = {
                "training_date": datetime.now().isoformat(),
                "config": asdict(config),
                "system_info": {
                    "gpu_type": GPU_TYPE,
                    "gpu_name": GPU_NAME,
                    "gpu_memory": f"{GPU_MEMORY:.1f}GB",
                    "backend": GPU_BACKEND,
                },
                "library_versions": get_library_versions(),
                "dataset_hash": compute_file_hash(config.dataset_path),
                "training_stats": {
                    "total_steps": self.total_steps,
                    "final_loss": self.last_loss,
                    "duration": format_time(time.time() - self.start_time) if self.start_time else "N/A",
                }
            }

            manifest_path = output_path / "training_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)

            self.log(f"Manifest saved: {manifest_path}\n")

        except Exception as e:
            self.log(f"Manifest error: {str(e)}\n")

    def stop(self):
        """Request training stop"""
        self.should_stop = True
        self.log("Stop requested - finishing current step...\n")

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.trainer:
                del self.trainer
            if HAS_TORCH and HAS_GPU:
                if GPU_BACKEND == "cuda":
                    torch.cuda.empty_cache()
                elif GPU_BACKEND == "mps":
                    torch.mps.empty_cache()
            self.log("Cleanup complete\n")
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════
# GUI APPLICATION
# ═══════════════════════════════════════════════════════════════════════

class NTTunerGUI:
    """Main GUI application - optimized for NTCompanion datasets"""

    def __init__(self):
        self.config = TrainingConfig()
        self.trainer = None
        self.available_models = []

    def append_log(self, message: str):
        """Append message to log"""
        if dpg.does_item_exist("log"):
            current = dpg.get_value("log")
            dpg.set_value("log", current + message)
            if dpg.does_item_exist("log_window"):
                dpg.set_y_scroll("log_window", -1.0)

    def read_config_from_gui(self):
        """Read configuration from GUI"""
        custom_model = dpg.get_value("base_model").strip()
        if custom_model and not custom_model.startswith("---"):
            self.config.base_model = custom_model
        else:
            combo_value = dpg.get_value("base_model_combo")
            if combo_value and not combo_value.startswith("---"):
                self.config.base_model = combo_value

        self.config.dataset_path = dpg.get_value("dataset_path")
        self.config.lora_rank = dpg.get_value("lora_rank")
        self.config.lora_alpha = dpg.get_value("lora_alpha")
        self.config.lora_dropout = dpg.get_value("lora_dropout")
        self.config.epochs = dpg.get_value("epochs")
        self.config.batch_size = dpg.get_value("batch_size")
        self.config.grad_accumulation = dpg.get_value("grad_accumulation")
        self.config.learning_rate = dpg.get_value("learning_rate")
        self.config.warmup_steps = dpg.get_value("warmup_steps")
        self.config.max_seq_length = dpg.get_value("max_seq_length")
        self.config.output_name = dpg.get_value("output_name")
        self.config.output_dir = dpg.get_value("output_dir")
        self.config.save_steps = dpg.get_value("save_steps")
        self.config.logging_steps = dpg.get_value("logging_steps")

        # ────────────────────────────────────────────────────────────────
        # Added: Read GGUF export config from GUI (2026)
        # ────────────────────────────────────────────────────────────────
        self._read_gguf_config_from_gui()
        # ────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # Added: GGUF config reader (2026)
    # ────────────────────────────────────────────────────────────────────────
    def _read_gguf_config_from_gui(self):
        """Read GGUF export configuration from GUI"""
        if dpg.does_item_exist("gguf_use_advanced"):
            self.config.gguf_export.use_advanced_export = dpg.get_value("gguf_use_advanced")
        if dpg.does_item_exist("gguf_quant_type"):
            self.config.gguf_export.quant_type = dpg.get_value("gguf_quant_type")
        if dpg.does_item_exist("gguf_preset"):
            self.config.gguf_export.preset = dpg.get_value("gguf_preset")
        if dpg.does_item_exist("gguf_export_all"):
            self.config.gguf_export.export_all_quants = dpg.get_value("gguf_export_all")
        if dpg.does_item_exist("gguf_imatrix_path"):
            self.config.gguf_export.imatrix_path = dpg.get_value("gguf_imatrix_path")
        if dpg.does_item_exist("gguf_custom_flags"):
            self.config.gguf_export.custom_flags = dpg.get_value("gguf_custom_flags")
        if dpg.does_item_exist("gguf_filename_pattern"):
            self.config.gguf_export.output_filename_pattern = dpg.get_value("gguf_filename_pattern")
        if dpg.does_item_exist("gguf_skip_merge"):
            self.config.gguf_export.skip_merge_lora_only = dpg.get_value("gguf_skip_merge")
        if dpg.does_item_exist("gguf_quantize_path"):
            self.config.gguf_export.llama_quantize_path = dpg.get_value("gguf_quantize_path")
        if dpg.does_item_exist("gguf_auto_ollama"):
            self.config.gguf_export.auto_import_ollama = dpg.get_value("gguf_auto_ollama")
        if dpg.does_item_exist("gguf_keep_f16"):
            self.config.gguf_export.keep_f16_base = dpg.get_value("gguf_keep_f16")

    # ────────────────────────────────────────────────────────────────────────

    def validate_and_warn(self) -> bool:
        """Validate configuration"""
        self.read_config_from_gui()
        valid, message = self.config.validate()
        if not valid:
            self.append_log(f"[ERROR] {message}\n")
            return False
        self.append_log("[OK] Configuration valid\n")
        warnings = self.config.get_warnings()
        if warnings:
            self.append_log("\n[!] WARNINGS:\n")
            for warning in warnings:
                self.append_log(f"  - {warning}\n")
            self.append_log("\n")
        return True

    def train_with_chronicals(self, config, progress_callback):
        """Dedicated Chronicals path to keep main logic clean."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.log("Initializing Chronicals 'Insane Fast Mode' (v0.1.0)...")

        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # Chronicals specific setup
        c_config = ChronicalsConfig(use_fused_kernels=True, use_packing="BFD")
        model = chronicals.prepare_model_for_lora(
            model, r=config.lora_rank, alpha=config.lora_alpha
        )

        # Packing logic using your existing handler
        raw_data = self.dataset_handler.load(config.dataset_path)
        packer = SequencePacker(tokenizer, max_seq_length=config.max_seq_length)
        packed_ds = packer.pack(raw_data)

        optimizer = LoRAPlusOptimizer(model.parameters(), lr=config.learning_rate)

        trainer = ChronicalsTrainer(
            model=model,
            config=c_config,
            train_dataset=packed_ds,
            optimizer=optimizer,
            args=trl.SFTConfig(
                output_dir=config.output_dir,
                per_device_train_batch_size=config.batch_size,
                num_train_epochs=config.epochs,
                bf16=torch.cuda.is_bf16_supported(),
            )
        )
        trainer.train()

        # Store model + tokenizer on self so that the shared save_model()
        # path (adapter save → merge → GGUF export) works correctly.
        self.model = model
        self.tokenizer = tokenizer
        self.trainer.save_model(config)  # handles adapter save + merge + GGUF

    def start_training_callback(self):
        """Start training"""
        if not self.validate_and_warn():
            return
        if not DEPS_AVAILABLE:
            self.append_log("[ERROR] Missing dependencies\n")
            return
        self.set_training_state(True)

        def training_thread():
            success = self.trainer.train(self.config)
            dpg.configure_item("btn_start", enabled=True)
            dpg.configure_item("btn_stop", enabled=False)
            self.set_training_state(False)

        threading.Thread(target=training_thread, daemon=True).start()

    def stop_training_callback(self):
        """Stop training"""
        if self.trainer:
            self.trainer.stop()
            # Reset progress indicators
            if dpg.does_item_exist("training_progress_bar"):
                dpg.set_value("training_progress_bar", 0.0)
            if dpg.does_item_exist("progress_text"):
                dpg.set_value("progress_text", "Training Stopped by User")

    def set_training_state(self, is_training: bool):
        """Lock/unlock UI"""
        dpg.configure_item("btn_start", enabled=not is_training)
        dpg.configure_item("btn_stop", enabled=is_training)

    def auto_configure(self):
        """Auto-configure based on hardware and dataset"""
        self.append_log("Auto-configuring...\n")

        dataset_path = dpg.get_value("dataset_path")
        if not dataset_path or not os.path.exists(dataset_path):
            self.append_log("[!] Load a dataset first\n")
            return

        data, error = DatasetHandler.load_dataset(dataset_path)
        if not data:
            self.append_log(f"[ERROR] {error}\n")
            return

        valid, message, stats = DatasetHandler.validate_dataset(data)
        if not valid:
            self.append_log(f"[ERROR] {message}\n")
            return

        # Configure based on GPU
        if HAS_GPU:
            if GPU_MEMORY >= 16:
                dpg.set_value("batch_size", 2)
                dpg.set_value("grad_accumulation", 8)
                dpg.set_value("max_seq_length", 1024)
                dpg.set_value("lora_rank", 64)
            elif GPU_MEMORY >= 8:
                dpg.set_value("batch_size", 1)
                dpg.set_value("grad_accumulation", 8)
                dpg.set_value("max_seq_length", 512)
                dpg.set_value("lora_rank", 32)
            else:
                dpg.set_value("batch_size", 1)
                dpg.set_value("grad_accumulation", 4)
                dpg.set_value("max_seq_length", 256)
                dpg.set_value("lora_rank", 16)
        else:
            dpg.set_value("batch_size", 1)
            dpg.set_value("grad_accumulation", 2)
            dpg.set_value("max_seq_length", 256)
            dpg.set_value("lora_rank", 8)

        # Configure epochs based on dataset size
        if stats["total_entries"] < 100:
            dpg.set_value("epochs", 3)
        elif stats["total_entries"] < 1000:
            dpg.set_value("epochs", 2)
        else:
            dpg.set_value("epochs", 1)

        self.append_log("[OK] Auto-configuration complete\n")

    def show_dataset_preview(self):
        """Show dataset preview"""
        dataset_path = dpg.get_value("dataset_path")
        if not dataset_path or not os.path.exists(dataset_path):
            self.append_log("[!] No dataset loaded\n")
            return

        data, error = DatasetHandler.load_dataset(dataset_path)
        if not data:
            self.append_log(f"[ERROR] {error}\n")
            return

        previews = DatasetHandler.preview_entries(data, count=3)

        if dpg.does_item_exist("preview_window"):
            dpg.delete_item("preview_window")

        with dpg.window(label="Dataset Preview", tag="preview_window", width=700, height=500,
                        pos=[200, 150], no_resize=True):
            dpg.add_text(f"Dataset: {dataset_path}", color=[100, 200, 255])
            dpg.add_text(f"Showing first 3 of {len(data)} entries")
            dpg.add_separator()

            with dpg.child_window(height=400, border=True):
                for preview in previews:
                    dpg.add_text(preview, wrap=0)
                    dpg.add_separator()

    def show_file_dialog(self):
        dpg.show_item("file_dialog")

    def show_output_dir_dialog(self):
        dpg.show_item("output_dir_dialog")

    def select_file_callback(self, sender, app_data):
        selections = app_data["selections"]
        if selections:
            filepath = list(selections.values())[0]
            dpg.set_value("dataset_path", filepath)
            self.append_log(f"Dataset selected: {filepath}\n")

    def select_output_dir_callback(self, sender, app_data):
        selections = app_data["selections"]
        if selections:
            dirpath = list(selections.values())[0]
            dpg.set_value("output_dir", dirpath)

    def model_selected_callback(self, sender, app_data):
        """Called when user picks a model from the combo OR edits the custom text field.
        Detects the chat template, updates the indicator label, and blocks
        Ollama colon-style names from being used for training."""
        if app_data.startswith("---"):
            return
        dpg.set_value("base_model", app_data)
        self._update_template_indicator(app_data)

    def _update_template_indicator(self, model_name: str):
        """Detect template for *model_name* and refresh the GUI indicator text."""
        if not dpg.does_item_exist("template_indicator"):
            return

        # Ollama colon names cannot be loaded by transformers
        if ":" in model_name and "/" not in model_name:
            dpg.set_value("template_indicator",
                          f"⚠ '{model_name}' is an Ollama name — use Download first, then pick the HuggingFace equivalent")
            dpg.configure_item("template_indicator", color=[255, 180, 80])
            return

        info = detect_template_for_model(model_name)
        if info:
            display, markers, _ = info
            dpg.set_value("template_indicator",
                          f"✓ Detected template: {display}   (markers: {', '.join(markers)})")
            dpg.configure_item("template_indicator", color=[100, 220, 100])
        else:
            dpg.set_value("template_indicator",
                          "? No known template — dataset must already be formatted correctly")
            dpg.configure_item("template_indicator", color=[255, 200, 80])

    def refresh_models(self):
        self.append_log("Refreshing models...\n")
        self.available_models = get_popular_models()
        if dpg.does_item_exist("base_model_combo"):
            dpg.configure_item("base_model_combo", items=self.available_models)

    def download_model(self):
        model = dpg.get_value("base_model_combo")
        if model.startswith("---") or "/" in model:
            self.append_log("[!] Select an Ollama model\n")
            return

        self.append_log(f"Downloading {model}...\n")

        def download():
            try:
                result = safe_subprocess_run(["ollama", "pull", model], capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    self.append_log(f"[OK] {model} downloaded\n")
                else:
                    self.append_log(f"[ERROR] {result.stderr}\n")
            except Exception as e:
                self.append_log(f"[ERROR] {str(e)}\n")

        threading.Thread(target=download, daemon=True).start()

    def clear_log_callback(self):
        dpg.set_value("log", "")

    def save_config_callback(self):
        dpg.show_item("save_config_dialog")

    def load_config_callback(self):
        dpg.show_item("load_config_dialog")

    def save_config_file(self, sender, app_data):
        selections = app_data["selections"]
        if selections:
            filepath = list(selections.values())[0]
            try:
                self.read_config_from_gui()
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(asdict(self.config), f, indent=2)
                self.append_log(f"[OK] Config saved\n")
            except Exception as e:
                self.append_log(f"[ERROR] {str(e)}\n")

    def load_config_file(self, sender, app_data):
        selections = app_data["selections"]
        if selections:
            filepath = list(selections.values())[0]
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, value in data.items():
                    if dpg.does_item_exist(key):
                        dpg.set_value(key, value)
                # ────────────────────────────────────────────────────────────
                # Added: Load GGUF export config (2026)
                # ────────────────────────────────────────────────────────────
                if "gguf_export" in data:
                    gguf_data = data["gguf_export"]
                    if dpg.does_item_exist("gguf_use_advanced"):
                        dpg.set_value("gguf_use_advanced", gguf_data.get("use_advanced_export", False))
                    if dpg.does_item_exist("gguf_quant_type"):
                        dpg.set_value("gguf_quant_type", gguf_data.get("quant_type", "Q4_K_M"))
                    if dpg.does_item_exist("gguf_preset"):
                        dpg.set_value("gguf_preset", gguf_data.get("preset", "Standard Quality (Q4_K_M)"))
                    if dpg.does_item_exist("gguf_export_all"):
                        dpg.set_value("gguf_export_all", gguf_data.get("export_all_quants", False))
                    if dpg.does_item_exist("gguf_imatrix_path"):
                        dpg.set_value("gguf_imatrix_path", gguf_data.get("imatrix_path", ""))
                    if dpg.does_item_exist("gguf_custom_flags"):
                        dpg.set_value("gguf_custom_flags", gguf_data.get("custom_flags", ""))
                    if dpg.does_item_exist("gguf_filename_pattern"):
                        dpg.set_value("gguf_filename_pattern",
                                      gguf_data.get("output_filename_pattern", "{model_name}-{quant_type}"))
                    if dpg.does_item_exist("gguf_skip_merge"):
                        dpg.set_value("gguf_skip_merge", gguf_data.get("skip_merge_lora_only", False))
                    if dpg.does_item_exist("gguf_quantize_path"):
                        dpg.set_value("gguf_quantize_path", gguf_data.get("llama_quantize_path", ""))
                    if dpg.does_item_exist("gguf_auto_ollama"):
                        dpg.set_value("gguf_auto_ollama", gguf_data.get("auto_import_ollama", True))
                    if dpg.does_item_exist("gguf_keep_f16"):
                        dpg.set_value("gguf_keep_f16", gguf_data.get("keep_f16_base", False))
                # ────────────────────────────────────────────────────────────
                self.append_log(f"[OK] Config loaded\n")
            except Exception as e:
                self.append_log(f"[ERROR] {str(e)}\n")

    # ────────────────────────────────────────────────────────────────────────
    # Added: GGUF export GUI callbacks (2026)
    # ────────────────────────────────────────────────────────────────────────
    def _on_gguf_preset_changed(self, sender, app_data):
        """Update quant type when preset changes"""
        if app_data in GGUF_PRESETS:
            quants = GGUF_PRESETS[app_data]
            if quants and dpg.does_item_exist("gguf_quant_type"):
                dpg.set_value("gguf_quant_type", quants[0])

    def _on_gguf_advanced_toggled(self, sender, app_data):
        """Enable/disable advanced options based on checkbox"""
        enabled = app_data
        controls = [
            "gguf_quant_type", "gguf_preset", "gguf_export_all",
            "gguf_imatrix_path", "gguf_custom_flags", "gguf_filename_pattern",
            "gguf_skip_merge", "gguf_quantize_path", "gguf_auto_ollama",
            "gguf_keep_f16", "gguf_browse_imatrix", "gguf_browse_quantize",
            "btn_export_only"
        ]
        for ctrl in controls:
            if dpg.does_item_exist(ctrl):
                dpg.configure_item(ctrl, enabled=enabled)

    def _show_imatrix_dialog(self):
        """Show file dialog for importance matrix"""
        dpg.show_item("imatrix_dialog")

    def _select_imatrix_callback(self, sender, app_data):
        """Handle imatrix file selection"""
        selections = app_data["selections"]
        if selections:
            filepath = list(selections.values())[0]
            dpg.set_value("gguf_imatrix_path", filepath)

    def _show_quantize_dialog(self):
        """Show file dialog for llama-quantize binary"""
        dpg.show_item("quantize_dialog")

    def _select_quantize_callback(self, sender, app_data):
        """Handle quantize binary selection"""
        selections = app_data["selections"]
        if selections:
            filepath = list(selections.values())[0]
            # Reject script files – the user may have browsed to the
            # convert script instead of the quantize binary by mistake.
            _script_exts = {".py", ".sh", ".bat", ".cmd", ".ps1"}
            if Path(filepath).suffix.lower() in _script_exts:
                self.append_log(
                    f"[!] '{Path(filepath).name}' is a script, not llama-quantize.\n"
                    "    Please browse to llama-quantize.exe instead.\n"
                )
                dpg.set_value("gguf_quantize_path", "")
            else:
                dpg.set_value("gguf_quantize_path", filepath)

    def _run_export_only(self):
        """Run GGUF export on an existing model without training"""
        self.read_config_from_gui()

        model_path = Path(self.config.output_dir) / self.config.output_name
        if not model_path.exists():
            self.append_log(f"[ERROR] Model not found: {model_path}\n")
            self.append_log("Train a model first or check output path\n")
            return

        # ── detect adapter-only directory and merge if needed ───────────────
        export_path = model_path  # default: use as-is
        if not (model_path / "config.json").exists():
            # Looks like an adapter-only directory — try to merge
            self.append_log(
                f"[!] {model_path} appears to be an adapter-only directory\n"
                "    (no config.json). Attempting merge with base model...\n"
            )
            # Re-use TrainingManager's merge helper (it only needs config + path)
            tmp_mgr = TrainingManager(self.append_log)
            merged = tmp_mgr.merge_lora_to_base(self.config, model_path)
            if merged is None:
                self.append_log(
                    "[ERROR] Merge failed. Please train with NTTuner so the\n"
                    "        merged model is saved automatically, or place a\n"
                    "        full (merged) model in the output directory.\n"
                )
                return
            export_path = merged
        # ─────────────────────────────────────────────────────────────────────

        self.append_log(f"Running GGUF export on: {export_path}\n")

        def export_thread():
            # Reuse the existing manager so any Browse-set path is kept
            gguf_manager = self.trainer.gguf_manager
            gguf_output = export_path / "gguf"

            results = gguf_manager.run_full_export(
                model_path=export_path,
                output_path=gguf_output,
                gguf_config=self.config.gguf_export,
                model_name=self.config.output_name
            )

            if results["success"] and self.config.gguf_export.auto_import_ollama:
                # Import first successful quant
                if results["quantized_files"]:
                    first_quant = list(results["quantized_files"].values())[0]
                    gguf_manager.import_to_ollama(first_quant, self.config.output_name)

        threading.Thread(target=export_thread, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────────
    # End: GGUF export GUI callbacks
    # ────────────────────────────────────────────────────────────────────────

    def create_gui(self):
        """Create GUI"""
        if not _init_dpg():
            print("Cannot create GUI - DearPyGui failed to load")
            return False
        """Create GUI"""
        dpg.create_context()

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [18, 18, 22])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [25, 25, 30])
                dpg.add_theme_color(dpg.mvThemeCol_Border, [50, 50, 60])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [30, 30, 38])
                dpg.add_theme_color(dpg.mvThemeCol_Button, [40, 40, 50])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [55, 55, 70])
                dpg.add_theme_color(dpg.mvThemeCol_Text, [220, 220, 230])
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 15, 15)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)

        dpg.bind_theme(global_theme)

        with dpg.file_dialog(directory_selector=False, show=False, callback=self.select_file_callback,
                             tag="file_dialog", width=700, height=400):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".json", color=(150, 255, 150, 255))
            dpg.add_file_extension(".jsonl", color=(150, 255, 150, 255))

        with dpg.file_dialog(directory_selector=True, show=False, callback=self.select_output_dir_callback,
                             tag="output_dir_dialog", width=700, height=400):
            pass

        with dpg.file_dialog(directory_selector=False, show=False, callback=self.save_config_file,
                             tag="save_config_dialog", width=700, height=400, default_filename="config.json"):
            dpg.add_file_extension(".json", color=(150, 255, 150, 255))

        with dpg.file_dialog(directory_selector=False, show=False, callback=self.load_config_file,
                             tag="load_config_dialog", width=700, height=400):
            dpg.add_file_extension(".json", color=(150, 255, 150, 255))

        # ────────────────────────────────────────────────────────────────────
        # Added: File dialogs for GGUF export (2026)
        # ────────────────────────────────────────────────────────────────────
        with dpg.file_dialog(directory_selector=False, show=False,
                             callback=self._select_imatrix_callback,
                             tag="imatrix_dialog", width=700, height=400):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".dat", color=(255, 200, 100, 255))
            dpg.add_file_extension(".imatrix", color=(255, 200, 100, 255))

        with dpg.file_dialog(directory_selector=False, show=False,
                             callback=self._select_quantize_callback,
                             tag="quantize_dialog", width=700, height=400):
            dpg.add_file_extension(".*")
        # ────────────────────────────────────────────────────────────────────

        with dpg.window(tag="main", label="NTTuner - Fine-Tuning for NTCompanion Datasets"):
            with dpg.group(horizontal=True):
                dpg.add_text("NTTuner Professional", color=[0, 180, 255])
                dpg.add_text("| Optimized for NTCompanion", color=[120, 220, 140])

            dpg.add_spacer(height=5)

            status = f"GPU: {GPU_TYPE} - {GPU_NAME}"
            if HAS_GPU:
                status += f" ({GPU_MEMORY:.1f}GB)"
            status += f" | Backend: {GPU_BACKEND}"
            dpg.add_text(status, color=[0, 255, 100] if HAS_GPU else [255, 200, 100])
            dpg.add_separator()

            with dpg.collapsing_header(label="Model & Dataset", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_combo(label="Base Model", items=self.available_models,
                                  default_value=self.config.base_model,
                                  tag="base_model_combo", width=320, callback=self.model_selected_callback)
                    dpg.add_button(label="Refresh", callback=self.refresh_models, width=80)
                    dpg.add_button(label="Download", callback=self.download_model, width=80)

                dpg.add_input_text(label="Custom Model", default_value=self.config.base_model, tag="base_model",
                                   width=500,
                                   callback=lambda s, a: self._update_template_indicator(a),
                                   on_enter=True)

                # Live template indicator — updated whenever model selection changes
                dpg.add_text("? Enter or select a model to detect its chat template",
                             tag="template_indicator", color=[255, 200, 80])

                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="Dataset Path", default_value=self.config.dataset_path,
                                       tag="dataset_path", width=380,
                                       hint="NTCompanion JSONL format")
                    dpg.add_button(label="Browse", callback=self.show_file_dialog, width=80)
                    dpg.add_button(label="Preview", callback=self.show_dataset_preview, width=80)

                dpg.add_text("Note: NTCompanion datasets are pre-formatted with chat templates",
                             color=[150, 200, 150])

            with dpg.collapsing_header(label="LoRA Configuration"):
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(label="Rank", default_value=self.config.lora_rank, min_value=8,
                                       max_value=256, tag="lora_rank", width=200)
                    dpg.add_slider_int(label="Alpha", default_value=self.config.lora_alpha, min_value=8,
                                       max_value=512, tag="lora_alpha", width=200)
                dpg.add_slider_float(label="Dropout", default_value=self.config.lora_dropout, min_value=0.0,
                                     max_value=0.5, tag="lora_dropout", width=300, format="%.3f")

            with dpg.collapsing_header(label="Training Parameters"):
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(label="Epochs", default_value=self.config.epochs, min_value=1,
                                       max_value=10, tag="epochs", width=150)
                    dpg.add_slider_int(label="Batch Size", default_value=self.config.batch_size, min_value=1,
                                       max_value=16, tag="batch_size", width=150)
                    dpg.add_slider_int(label="Grad Accum", default_value=self.config.grad_accumulation,
                                       min_value=1, max_value=32, tag="grad_accumulation", width=150)

                with dpg.group(horizontal=True):
                    dpg.add_input_float(label="Learning Rate", default_value=self.config.learning_rate,
                                        tag="learning_rate", width=150, format="%.2e")
                    dpg.add_slider_int(label="Warmup", default_value=self.config.warmup_steps, min_value=0,
                                       max_value=500, tag="warmup_steps", width=150)
                    dpg.add_slider_int(label="Max Seq Len", default_value=self.config.max_seq_length,
                                       min_value=128, max_value=8192, tag="max_seq_length", width=150)

                with dpg.group(horizontal=True):
                    dpg.add_slider_int(label="Save Steps", default_value=self.config.save_steps, min_value=10,
                                       max_value=1000, tag="save_steps", width=150)
                    dpg.add_slider_int(label="Log Steps", default_value=self.config.logging_steps, min_value=1,
                                       max_value=100, tag="logging_steps", width=150)

            with dpg.collapsing_header(label="Output Configuration"):
                dpg.add_input_text(label="Model Name", default_value=self.config.output_name, tag="output_name",
                                   width=400)
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="Output Dir", default_value=self.config.output_dir, tag="output_dir",
                                       width=380)
                    dpg.add_button(label="Browse", callback=self.show_output_dir_dialog, width=80)

            # ────────────────────────────────────────────────────────────────────
            # Added: Advanced GGUF Export Section (2026)
            # ────────────────────────────────────────────────────────────────────
            with dpg.collapsing_header(label="Advanced GGUF Export", default_open=False):
                dpg.add_text("Configure GGUF quantization and export options", color=[180, 180, 200])
                dpg.add_separator()

                dpg.add_checkbox(label="Use advanced GGUF export instead of default",
                                 tag="gguf_use_advanced", default_value=False,
                                 callback=self._on_gguf_advanced_toggled)

                dpg.add_spacer(height=5)

                with dpg.group(horizontal=True):
                    dpg.add_combo(label="Preset", items=list(GGUF_PRESETS.keys()),
                                  default_value="Standard Quality (Q4_K_M)",
                                  tag="gguf_preset", width=250,
                                  callback=self._on_gguf_preset_changed, enabled=False)
                    dpg.add_combo(label="Quant Type", items=GGUF_QUANT_TYPES,
                                  default_value="Q4_K_M",
                                  tag="gguf_quant_type", width=150, enabled=False)

                dpg.add_checkbox(label="Export all intermediate quants (Q2_K → Q8_0)",
                                 tag="gguf_export_all", default_value=False, enabled=False)
                dpg.add_checkbox(label="Keep F16 base file after quantization",
                                 tag="gguf_keep_f16", default_value=False, enabled=False)

                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="Importance Matrix", tag="gguf_imatrix_path",
                                       width=350, hint="Optional .dat file", enabled=False)
                    dpg.add_button(label="Browse", tag="gguf_browse_imatrix",
                                   callback=self._show_imatrix_dialog, width=80, enabled=False)

                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="llama-quantize Path", tag="gguf_quantize_path",
                                       width=350, hint="Auto-detected if empty", enabled=False)
                    dpg.add_button(label="Browse", tag="gguf_browse_quantize",
                                   callback=self._show_quantize_dialog, width=80, enabled=False)

                dpg.add_input_text(label="Custom Flags", tag="gguf_custom_flags", width=450,
                                   hint="Extra args for llama-quantize", enabled=False)
                dpg.add_input_text(label="Filename Pattern", tag="gguf_filename_pattern",
                                   default_value="{model_name}-{quant_type}", width=450, enabled=False)

                dpg.add_separator()

                dpg.add_checkbox(label="Skip merge / Export LoRA adapter only",
                                 tag="gguf_skip_merge", default_value=False, enabled=False)
                dpg.add_checkbox(label="Auto-import to Ollama after export",
                                 tag="gguf_auto_ollama", default_value=True, enabled=False)

                dpg.add_spacer(height=5)

                dpg.add_button(label="Export GGUF Now (existing model)", tag="btn_export_only",
                               callback=self._run_export_only, width=250, enabled=False)

                # Show detected llama-quantize path (uses full root-based search)
                detected_path = find_llama_quantize_full()
                if detected_path == "python-fallback":
                    dpg.add_text("Using llama-cpp-python package (no binary needed)",
                                 color=[100, 200, 100], tag="gguf_quantize_status")
                elif detected_path:
                    dpg.add_text(f"Detected llama-quantize: {detected_path}",
                                 color=[100, 200, 100], tag="gguf_quantize_status")
                else:
                    dpg.add_text("llama-quantize not found - install llama-cpp-python or build llama.cpp",
                                 color=[255, 200, 100], tag="gguf_quantize_status")
            # ────────────────────────────────────────────────────────────────────
            # End: Advanced GGUF Export Section
            # ────────────────────────────────────────────────────────────────────

            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_button(label="Start Training", callback=self.start_training_callback, tag="btn_start",
                               width=130, height=40)
                dpg.add_button(label="Stop", callback=self.stop_training_callback, tag="btn_stop", width=130,
                               height=40, enabled=False)
                dpg.add_button(label="Auto-Config", callback=self.auto_configure, width=130, height=40)
                dpg.add_button(label="Clear Log", callback=self.clear_log_callback, width=100, height=40)
                dpg.add_button(label="Save Config", callback=self.save_config_callback, width=100, height=40)
                dpg.add_button(label="Load Config", callback=self.load_config_callback, width=100, height=40)
            # Look for a line like this in your code:
            with dpg.group(horizontal=True):
                # ... existing buttons or inputs ...

                # PASTE HERE (Ensure it is indented at the same level as the other inputs)
                dpg.add_checkbox(
                    label="Use Chronicals (3.5x Speedup)",
                    tag="use_chronicals_chk",
                    default_value=False,
                    callback=lambda s, a: setattr(self.config, 'use_chronicals', a)
                )
            dpg.add_separator()

            dpg.add_text("Training Progress:", color=[150, 150, 160])
            dpg.add_progress_bar(tag="training_progress_bar", default_value=0.0, width=-1, height=25)
            dpg.add_text("", tag="progress_text", color=[100, 200, 100])

            dpg.add_spacer(height=5)
            dpg.add_text("Training Log:", color=[150, 150, 160])
            with dpg.child_window(tag="log_window", height=280, border=True):
                dpg.add_text("", tag="log", wrap=0)

            dpg.add_separator()
            dpg.add_text("NTTuner Professional | Optimized for NTCompanion Datasets | Advanced GGUF Export",
                         color=[80, 80, 90])

        dpg.create_viewport(title='NTTuner - Fine-Tuning Studio', width=1050, height=1000)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

    def run(self):
        """Run application"""
        self.trainer = TrainingManager(self.append_log)
        self.available_models = get_popular_models()

        self.create_gui()

        self.append_log("=" * 70 + "\n")
        self.append_log("NTTuner Professional - Optimized for NTCompanion\n")
        self.append_log("=" * 70 + "\n")
        self.append_log("GPU DETECTION:\n")

        for detail in GPU_INFO["details"]:
            self.append_log(f"  {detail}\n")

        self.append_log("\nFEATURES:\n")
        self.append_log("  ✓ NTCompanion JSONL format support\n")
        self.append_log("  ✓ Enhanced GPU detection (CUDA/ROCm/MPS)\n")
        self.append_log("  ✓ Dataset validation and statistics\n")
        self.append_log("  ✓ VRAM usage estimation\n")
        self.append_log("  ✓ Progress tracking with ETA\n")
        self.append_log("  ✓ Auto-configuration\n")
        self.append_log("  ✓ Advanced GGUF export options\n")
        self.append_log("=" * 70 + "\n\n")

        if not DEPS_AVAILABLE:
            self.append_log("[!] Missing dependencies\n")
            self.append_log("Install: pip install torch transformers datasets trl peft\n\n")

        ollama_models = get_ollama_models()
        if ollama_models:
            self.append_log(f"[OK] {len(ollama_models)} Ollama models found\n")

        # Check for llama-quantize (full root-based search)
        quantize_path = find_llama_quantize_full()
        if quantize_path == "python-fallback":
            self.append_log("[OK] llama-cpp-python package found (using Python API)\n")
        elif quantize_path:
            self.append_log(f"[OK] llama-quantize found: {quantize_path}\n")
        else:
            self.append_log("[!] llama-quantize not found - install llama-cpp-python or build llama.cpp\n")

        self.append_log("\nReady! Load NTCompanion dataset and start training\n")
        self.append_log("=" * 70 + "\n")

        dpg.start_dearpygui()
        dpg.destroy_context()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = NTTunerGUI()
    app.run()
