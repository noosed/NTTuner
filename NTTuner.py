# -*- coding: utf-8 -*-
"""
NTTuner - Professional LLM Fine-Tuning Studio
Optimized for NTCompanion datasets with enhanced GPU detection

FEATURES:
✓ Perfect compatibility with NTCompanion JSONL output
✓ Enhanced GPU detection (CUDA/ROCm/MPS)
✓ Dataset validation and statistics
✓ VRAM usage estimation
✓ Progress tracking with ETA
✓ Auto-configuration based on hardware
✓ Checkpoint resume and recovery
✓ Multi-format dataset support (JSONL/JSON/CSV)
✓ Blank system context support
✓ Advanced GGUF export options (2026)
"""

import dearpygui.dearpygui as dpg
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
# ENHANCED GPU DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_gpu_comprehensive():
    """
    Comprehensive GPU detection supporting:
    - NVIDIA CUDA
    - AMD ROCm
    - Apple Metal (MPS)
    """
    gpu_info = {
        "has_gpu": False,
        "gpu_type": "CPU",
        "gpu_name": "None",
        "gpu_memory": 0.0,
        "gpu_count": 0,
        "backend": "none",
        "cuda_version": None,
        "details": []
    }

    try:
        import torch
        gpu_info["details"].append(f"PyTorch {torch.__version__} detected")

        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "CUDA"
            gpu_info["backend"] = "cuda"
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["details"].append(f"CUDA {torch.version.cuda}")

            gpu_info["details"].append(f"Device: {gpu_info['gpu_name']}")
            gpu_info["details"].append(f"VRAM: {gpu_info['gpu_memory']:.1f} GB")
            gpu_info["details"].append(f"GPU Count: {gpu_info['gpu_count']}")
            return gpu_info

        # Check ROCm (AMD)
        if hasattr(torch, 'hip') and torch.hip.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "ROCm"
            gpu_info["backend"] = "hip"
            gpu_info["gpu_count"] = torch.hip.device_count()
            gpu_info["gpu_name"] = torch.hip.get_device_name(0)
            try:
                gpu_info["gpu_memory"] = torch.hip.get_device_properties(0).total_memory / (1024 ** 3)
            except:
                gpu_info["gpu_memory"] = 8.0
            gpu_info["details"].append(f"ROCm Device: {gpu_info['gpu_name']}")
            gpu_info["details"].append(f"VRAM: {gpu_info['gpu_memory']:.1f} GB")
            return gpu_info

        # Check Metal (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info["has_gpu"] = True
            gpu_info["gpu_type"] = "MPS"
            gpu_info["backend"] = "mps"
            gpu_info["gpu_count"] = 1
            try:
                import platform
                machine = platform.machine()
                if 'arm' in machine.lower():
                    gpu_info["gpu_name"] = "Apple Silicon (M-series)"
                    import psutil
                    total_ram = psutil.virtual_memory().total / (1024 ** 3)
                    gpu_info["gpu_memory"] = total_ram * 0.7
                else:
                    gpu_info["gpu_name"] = "Metal-capable GPU"
                    gpu_info["gpu_memory"] = 8.0
            except:
                gpu_info["gpu_name"] = "Apple Metal GPU"
                gpu_info["gpu_memory"] = 8.0
            gpu_info["details"].append(f"Metal Performance Shaders enabled")
            gpu_info["details"].append(f"Device: {gpu_info['gpu_name']}")
            gpu_info["details"].append(f"Unified Memory: ~{gpu_info['gpu_memory']:.1f} GB")
            return gpu_info

        gpu_info["details"].append("No GPU acceleration available")
        gpu_info["details"].append("Training will use CPU (very slow)")

    except ImportError:
        gpu_info["details"].append("PyTorch not installed")
    except Exception as e:
        gpu_info["details"].append(f"GPU detection error: {str(e)}")

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
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
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
        "Ollama Models (Installed)": get_ollama_models(),
        "Popular Ollama Models": [
            "llama3.2:3b", "llama3.1:8b", "llama3:70b",
            "mistral:7b", "mixtral:8x7b",
            "phi3:mini", "phi4:14b",
            "gemma:7b", "gemma2:9b",
            "qwen2.5:7b", "qwen2.5:14b",
        ],
        "Small Models (CPU-friendly)": [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "stabilityai/stablelm-2-1_6b",
            "Qwen/Qwen2-1.5B-Instruct",
        ],
        "Medium Models (GPU recommended)": [
            "unsloth/llama-3-8b-bnb-4bit",
            "unsloth/mistral-7b-v0.3-bnb-4bit",
            "unsloth/Phi-3-mini-4k-instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        "Large Models (Good GPU required)": [
            "unsloth/llama-3-70b-bnb-4bit",
            "meta-llama/Llama-3.1-8B-Instruct",
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


def find_llama_quantize() -> Optional[str]:
    """Find llama-quantize or llama.cpp quantize binary"""
    # Common names for the quantize binary
    binary_names = [
        "llama-quantize",
        "quantize",
        "llama.cpp-quantize",
        "main-quantize",
    ]

    # Check PATH first
    for name in binary_names:
        path = shutil.which(name)
        if path:
            return path

    # Check common installation locations
    common_paths = [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
        Path.home() / "llama.cpp" / "llama-quantize",
        Path.home() / "llama.cpp" / "quantize",
        Path("/usr/local/bin/llama-quantize"),
        Path("/usr/bin/llama-quantize"),
        Path("/opt/llama.cpp/llama-quantize"),
        Path.cwd() / "llama-quantize",
        Path.cwd() / "llama.cpp" / "llama-quantize",
    ]

    for path in common_paths:
        if path.exists() and path.is_file():
            return str(path)

    return None


def find_llama_gguf_split() -> Optional[str]:
    """Find llama-gguf-split binary for splitting large models"""
    binary_names = ["llama-gguf-split", "gguf-split"]

    for name in binary_names:
        path = shutil.which(name)
        if path:
            return path

    return None


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
# Added: GGUF EXPORT MANAGER (2026)
# ════════════════════════════════════════════════════════════════════════════

class GGUFExportManager:
    """
    Handles advanced GGUF export with full llama.cpp quantization options.
    Compatible with both Unsloth-trained and native PEFT/transformers LoRA adapters.
    """

    def __init__(self, log_callback):
        self.log = log_callback
        self.quantize_path = find_llama_quantize()

    def detect_adapter_type(self, model_path: Path) -> str:
        """Detect if the adapter is Unsloth or standard PEFT"""
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config, 'r') as f:
                    config = json.load(f)
                if "unsloth" in str(config).lower():
                    return "unsloth"
                return "peft"
            except:
                pass

        # Check for Unsloth-specific files
        if (model_path / "unsloth_config.json").exists():
            return "unsloth"

        return "unknown"

    def convert_to_gguf_f16(self, model_path: Path, output_path: Path,
                            gguf_config: GGUFExportConfig) -> Tuple[bool, str, Optional[Path]]:
        """
        Convert model to F16 GGUF format (base for further quantization).
        Handles both merged models and LoRA-only exports.
        """
        self.log("Converting to GGUF F16 base format...\n")

        f16_output = output_path / "model-f16.gguf"

        try:
            # Try Unsloth's built-in GGUF export first (if available and applicable)
            if HAS_UNSLOTH and not gguf_config.skip_merge_lora_only:
                try:
                    self.log("Attempting Unsloth GGUF export...\n")
                    from unsloth import FastLanguageModel

                    # Load the model
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=str(model_path),
                        max_seq_length=2048,
                        load_in_4bit=False,
                    )

                    # Save to GGUF
                    model.save_pretrained_gguf(
                        str(output_path),
                        tokenizer,
                        quantization_method="f16",
                    )

                    # Find the output file
                    gguf_files = list(output_path.glob("*.gguf"))
                    if gguf_files:
                        # Rename to standard name
                        gguf_files[0].rename(f16_output)
                        self.log(f"Unsloth GGUF export successful: {f16_output}\n")
                        return True, "OK", f16_output

                except Exception as e:
                    self.log(f"Unsloth export failed, falling back to llama.cpp: {str(e)}\n")

            # Fallback: Use llama.cpp convert script
            convert_script = self._find_convert_script()
            if convert_script:
                self.log(f"Using convert script: {convert_script}\n")

                cmd = [
                    sys.executable, str(convert_script),
                    str(model_path),
                    "--outfile", str(f16_output),
                    "--outtype", "f16",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                if result.returncode == 0 and f16_output.exists():
                    self.log(f"F16 GGUF created: {f16_output}\n")
                    return True, "OK", f16_output
                else:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    self.log(f"Convert failed: {error_msg}\n")
                    return False, error_msg, None

            # Try transformers built-in if available
            try:
                self.log("Attempting transformers GGUF export...\n")
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model = AutoModelForCausalLM.from_pretrained(str(model_path))
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))

                # Some newer transformers versions support GGUF export
                if hasattr(model, 'save_pretrained_gguf'):
                    model.save_pretrained_gguf(str(f16_output), tokenizer)
                    if f16_output.exists():
                        return True, "OK", f16_output
            except Exception as e:
                self.log(f"Transformers GGUF export not available: {str(e)}\n")

            return False, "No suitable GGUF conversion method found", None

        except Exception as e:
            error_msg = f"F16 conversion failed: {str(e)}"
            self.log(f"ERROR: {error_msg}\n")
            return False, error_msg, None

    def _find_convert_script(self) -> Optional[Path]:
        """Find llama.cpp convert script"""
        script_names = [
            "convert_hf_to_gguf.py",
            "convert-hf-to-gguf.py",
            "convert.py",
        ]

        search_paths = [
            Path.home() / "llama.cpp",
            Path("/usr/local/share/llama.cpp"),
            Path("/opt/llama.cpp"),
            Path.cwd() / "llama.cpp",
        ]

        for base_path in search_paths:
            for script_name in script_names:
                script_path = base_path / script_name
                if script_path.exists():
                    return script_path

        # Check if installed via pip
        try:
            import llama_cpp
            pkg_path = Path(llama_cpp.__file__).parent
            for script_name in script_names:
                script_path = pkg_path / script_name
                if script_path.exists():
                    return script_path
        except ImportError:
            pass

        return None

    def quantize_gguf(self, input_gguf: Path, output_path: Path,
                      quant_type: str, gguf_config: GGUFExportConfig,
                      model_name: str) -> Tuple[bool, str, Optional[Path]]:
        """
        Quantize a GGUF file using llama-quantize.
        """
        # Generate output filename from pattern
        output_filename = gguf_config.output_filename_pattern.format(
            model_name=model_name,
            quant_type=quant_type.lower(),
        )
        if not output_filename.endswith(".gguf"):
            output_filename += ".gguf"

        output_file = output_path / output_filename

        self.log(f"Quantizing to {quant_type}: {output_file.name}...\n")

        quantize_bin = gguf_config.llama_quantize_path or self.quantize_path

        if not quantize_bin or not Path(quantize_bin).exists():
            # Try to find it
            quantize_bin = find_llama_quantize()
            if not quantize_bin:
                return False, "llama-quantize binary not found. Please install llama.cpp or specify path.", None

        try:
            cmd = [quantize_bin, str(input_gguf), str(output_file), quant_type]

            # Add importance matrix if specified
            if gguf_config.imatrix_path and Path(gguf_config.imatrix_path).exists():
                cmd.extend(["--imatrix", gguf_config.imatrix_path])
                self.log(f"  Using importance matrix: {gguf_config.imatrix_path}\n")

            # Add custom flags
            if gguf_config.custom_flags:
                custom_args = gguf_config.custom_flags.split()
                cmd.extend(custom_args)
                self.log(f"  Custom flags: {gguf_config.custom_flags}\n")

            self.log(f"  Running: {' '.join(cmd)}\n")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0 and output_file.exists():
                file_size = output_file.stat().st_size / (1024 ** 3)
                self.log(f"  ✓ {quant_type} complete: {output_file.name} ({file_size:.2f} GB)\n")
                return True, "OK", output_file
            else:
                error_msg = result.stderr or result.stdout or "Unknown quantization error"
                self.log(f"  ✗ {quant_type} failed: {error_msg}\n")
                return False, error_msg, None

        except subprocess.TimeoutExpired:
            return False, "Quantization timed out (>1 hour)", None
        except Exception as e:
            return False, str(e), None

    def export_lora_only_gguf(self, adapter_path: Path, output_path: Path,
                              gguf_config: GGUFExportConfig) -> Tuple[bool, str, Optional[Path]]:
        """
        Export LoRA adapter only as GGUF (without merging with base model).
        This creates a smaller file that can be applied at runtime.
        """
        self.log("Exporting LoRA adapter as standalone GGUF...\n")

        lora_output = output_path / "lora-adapter.gguf"

        try:
            # Look for export-lora-to-gguf.py or similar
            script_names = [
                "export-lora-to-gguf.py",
                "convert-lora-to-gguf.py",
                "lora_to_gguf.py",
            ]

            script_path = None
            for base in [Path.home() / "llama.cpp", Path.cwd() / "llama.cpp"]:
                for name in script_names:
                    candidate = base / name
                    if candidate.exists():
                        script_path = candidate
                        break
                if script_path:
                    break

            if script_path:
                cmd = [
                    sys.executable, str(script_path),
                    str(adapter_path),
                    "--outfile", str(lora_output),
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode == 0 and lora_output.exists():
                    self.log(f"LoRA GGUF exported: {lora_output}\n")
                    return True, "OK", lora_output

            # Fallback: try using llama-export-lora if available
            export_bin = shutil.which("llama-export-lora")
            if export_bin:
                cmd = [export_bin, "-m", str(adapter_path), "-o", str(lora_output)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode == 0 and lora_output.exists():
                    return True, "OK", lora_output

            return False, "LoRA-only GGUF export not available. Install llama.cpp with export-lora support.", None

        except Exception as e:
            return False, str(e), None

    def run_full_export(self, model_path: Path, output_path: Path,
                        gguf_config: GGUFExportConfig, model_name: str) -> Dict[str, Any]:
        """
        Run the complete GGUF export process with all configured options.
        Returns a dict with results for each quantization type.
        """
        results = {
            "success": False,
            "f16_path": None,
            "quantized_files": {},
            "errors": [],
            "lora_only_path": None,
        }

        output_path.mkdir(parents=True, exist_ok=True)

        self.log("=" * 60 + "\n")
        self.log("ADVANCED GGUF EXPORT\n")
        self.log("=" * 60 + "\n")

        # Handle LoRA-only export if requested
        if gguf_config.skip_merge_lora_only:
            self.log("LoRA-only export mode selected\n")
            success, error, lora_path = self.export_lora_only_gguf(
                model_path, output_path, gguf_config
            )
            if success:
                results["lora_only_path"] = lora_path
                results["success"] = True
            else:
                results["errors"].append(f"LoRA export: {error}")
            return results

        # Convert to F16 GGUF first
        success, error, f16_path = self.convert_to_gguf_f16(
            model_path, output_path, gguf_config
        )

        if not success:
            results["errors"].append(f"F16 conversion: {error}")
            return results

        results["f16_path"] = f16_path

        # Get list of quants to export
        quant_list = gguf_config.get_effective_quants()
        self.log(f"\nQuantization targets: {', '.join(quant_list)}\n\n")

        # Run quantization for each type
        for quant_type in quant_list:
            if quant_type.upper() in ["F16", "FP16"]:
                # F16 already done
                results["quantized_files"]["F16"] = f16_path
                continue

            success, error, quant_path = self.quantize_gguf(
                f16_path, output_path, quant_type.upper(),
                gguf_config, model_name
            )

            if success:
                results["quantized_files"][quant_type] = quant_path
            else:
                results["errors"].append(f"{quant_type}: {error}")

        # Clean up F16 base if not keeping it
        if not gguf_config.keep_f16_base and f16_path and f16_path.exists():
            if "F16" not in quant_list and "f16" not in [q.lower() for q in quant_list]:
                try:
                    f16_path.unlink()
                    self.log("Cleaned up intermediate F16 file\n")
                except:
                    pass

        results["success"] = len(results["quantized_files"]) > 0

        # Summary
        self.log("\n" + "=" * 60 + "\n")
        self.log("EXPORT SUMMARY\n")
        self.log("=" * 60 + "\n")
        self.log(f"Successful exports: {len(results['quantized_files'])}\n")
        for qtype, path in results["quantized_files"].items():
            if path and path.exists():
                size_gb = path.stat().st_size / (1024 ** 3)
                self.log(f"  ✓ {qtype}: {path.name} ({size_gb:.2f} GB)\n")

        if results["errors"]:
            self.log(f"Errors: {len(results['errors'])}\n")
            for error in results["errors"]:
                self.log(f"  ✗ {error}\n")

        return results

    def import_to_ollama(self, gguf_path: Path, model_name: str) -> Tuple[bool, str]:
        """Import a GGUF file into Ollama"""
        self.log(f"Importing to Ollama as '{model_name}'...\n")

        try:
            # Create Modelfile
            modelfile_content = f"""FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
            modelfile_path = gguf_path.parent / "Modelfile"
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)

            # Run ollama create
            cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.log(f"✓ Imported to Ollama as '{model_name}'\n")
                return True, "OK"
            else:
                error = result.stderr or result.stdout or "Unknown error"
                self.log(f"Ollama import failed: {error}\n")
                return False, error

        except Exception as e:
            return False, str(e)


# ════════════════════════════════════════════════════════════════════════════
# End: GGUF EXPORT MANAGER
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
        # Added: GGUF export manager instance (2026)
        # ────────────────────────────────────────────────────────────────
        self.gguf_manager = GGUFExportManager(log_callback)
        # ────────────────────────────────────────────────────────────────

    def prepare_dataset(self, config: TrainingConfig) -> Tuple[Optional[Any], str]:
        """Prepare NTCompanion dataset for training"""
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

        try:
            # NTCompanion format: Each entry already has formatted "text" field
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
                            time_per_step = elapsed / state.global_step
                            remaining = self.manager.total_steps - state.global_step
                            eta = remaining * time_per_step
                            self.manager.log(
                                f"Step {state.global_step}/{self.manager.total_steps} | "
                                f"Loss: {logs.get('loss', 0):.4f} | ETA: {format_time(eta)}\n"
                            )

                def on_epoch_end(self, args, state, control, **kwargs):
                    self.manager.log(f"Epoch {int(state.epoch)} complete\n")

            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=training_args,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=config.max_seq_length,
                callbacks=[ProgressCallback(self)],
            )

            self.log("=" * 60 + "\n")
            self.log("Training started\n")
            self.log("=" * 60 + "\n")

            self.trainer.train()

            if not self.should_stop:
                self.log("=" * 60 + "\n")
                self.log("Training complete!\n")
                self.log("=" * 60 + "\n")
                self.save_model(config)

            return True

        except Exception as e:
            self.log(f"ERROR: Training failed\n{str(e)}\n{traceback.format_exc()}\n")
            return False

        finally:
            self.is_training = False
            self.cleanup()

    def save_model(self, config: TrainingConfig):
        """Save trained model"""
        try:
            self.log("Saving model...\n")
            output_path = Path(config.output_dir) / config.output_name
            output_path.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))

            self.log(f"Model saved to: {output_path}\n")
            self.create_manifest(config, output_path)

            # ────────────────────────────────────────────────────────────────
            # Added: Advanced GGUF export after model save (2026)
            # ────────────────────────────────────────────────────────────────
            self._run_gguf_export(config, output_path)
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
        if not app_data.startswith("---"):
            dpg.set_value("base_model", app_data)

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
                result = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=300)
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
            dpg.set_value("gguf_quantize_path", filepath)

    def _run_export_only(self):
        """Run GGUF export on an existing model without training"""
        self.read_config_from_gui()

        model_path = Path(self.config.output_dir) / self.config.output_name
        if not model_path.exists():
            self.append_log(f"[ERROR] Model not found: {model_path}\n")
            self.append_log("Train a model first or check output path\n")
            return

        self.append_log(f"Running GGUF export on: {model_path}\n")

        def export_thread():
            gguf_manager = GGUFExportManager(self.append_log)
            gguf_output = model_path / "gguf"

            results = gguf_manager.run_full_export(
                model_path=model_path,
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
                                   width=500)

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

                # Show detected llama-quantize path
                detected_path = find_llama_quantize()
                if detected_path:
                    dpg.add_text(f"Detected llama-quantize: {detected_path}", color=[100, 200, 100])
                else:
                    dpg.add_text("llama-quantize not found - specify path or install llama.cpp",
                                 color=[255, 200, 100])
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

            dpg.add_separator()

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

        # Check for llama-quantize
        quantize_path = find_llama_quantize()
        if quantize_path:
            self.append_log(f"[OK] llama-quantize found: {quantize_path}\n")
        else:
            self.append_log("[!] llama-quantize not found - install llama.cpp for advanced GGUF export\n")

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
