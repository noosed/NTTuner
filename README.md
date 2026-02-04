<img width="1000" height="720" alt="image" src="https://github.com/user-attachments/assets/da945fa6-0bdb-4d7c-b209-4e73ca07824e" />

# NTTuner Enhanced

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/noosed/NTTuner)](https://github.com/noosed/NTTuner/stargazers)
[![Multi-Backend](https://img.shields.io/badge/GPU-Multi--Backend-orange.svg)](README.md)

A desktop GUI application for fine-tuning large language models and deploying them directly to Ollama. Built for ease of use, with support for **NVIDIA, AMD, Intel, Apple Silicon, and more**.

## 🎯 Enhanced Multi-Backend Support

This enhanced version extends NTTuner with comprehensive GPU support for non-NVIDIA users:

* **Vulkan Detection** - Support for Intel, AMD, and other GPUs via Vulkan API
* **OpenCL Detection** - Universal GPU compatibility layer for diverse hardware
* **Enhanced ROCm** - Better AMD GPU detection and configuration
* **Automatic Backend Selection** - Detects and uses the best available GPU backend
* **100% Backward Compatible** - All original NVIDIA/CUDA functionality preserved

## Overview

NTTuner simplifies the process of fine-tuning language models by providing an intuitive interface that handles the complexity of LoRA training, GGUF conversion, and Ollama integration. Whether you're customizing a model for a specific task or experimenting with different training configurations, this tool streamlines the entire workflow.

Now with **multi-backend GPU support**, you're no longer limited to NVIDIA hardware!

## Features

### Training Capabilities

* **LoRA Fine-tuning**: Efficient parameter-efficient training with configurable rank and alpha
* **Multi-Backend GPU Support**: NVIDIA CUDA, AMD ROCm, Apple MPS, Intel/other Vulkan, OpenCL
* **CPU Fallback**: Full support for CPU-only training (though significantly slower)
* **Unsloth Integration**: Optional 2-5x speedup with Unsloth library on supported NVIDIA GPUs
* **Background Training**: Non-blocking UI that remains responsive during training

### GPU Backend Support

| Backend | Hardware | Status | Performance |
|---------|----------|--------|-------------|
| **CUDA** | NVIDIA GPUs | ✅ Fully Supported | Excellent (with Unsloth) |
| **ROCm** | AMD GPUs | ✅ Enhanced Support | Good |
| **MPS** | Apple Silicon | ✅ Supported | Good |
| **Vulkan** | Intel/AMD/Others | ✅ Detection Added | CPU Fallback* |
| **OpenCL** | Various GPUs | ✅ Detection Added | CPU Fallback* |
| **CPU** | Any Processor | ✅ Always Available | Slow |

*Vulkan and OpenCL are detected and guidance is provided for acceleration options.

### Model Management

* **Ollama Integration**: Automatically imports models into your local Ollama instance
* **Model Discovery**: Detects and lists all installed Ollama models
* **Download Support**: Built-in downloader for popular Ollama models
* **HuggingFace Support**: Direct integration with HuggingFace model hub

### User Interface

* **Drag and Drop**: Drop dataset files directly onto the interface
* **Configuration Management**: Save and load training configurations as JSON
* **Real-time Logging**: Live training progress and detailed diagnostics
* **Model Browser**: Categorized dropdown of popular models by size and purpose

### Output Options

* **Multiple Quantization Levels**: Choose from q4\_k\_m to f16 based on your needs
* **Custom Output Directories**: Specify where to save your trained models
* **Automatic GGUF Export**: Converts models to GGUF format for Ollama compatibility

## Installation

### Prerequisites

* Python 3.10 or higher
* GPU (NVIDIA, AMD, Intel, or Apple Silicon recommended)
* Ollama installed ([download here](https://ollama.ai))

### Core Installation (All Users)

```bash
# Clone the repository
git clone https://github.com/noosed/nttuner.git
cd nttuner

# Install core dependencies
pip install torch transformers datasets trl peft accelerate dearpygui
```

### GPU-Specific Installation

#### NVIDIA GPU Users (CUDA)

```bash
# Install CUDA PyTorch
pip uninstall torch torchvision torchaudio  # Remove CPU version if present
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install GPU acceleration libraries
pip install bitsandbytes

# Optional: Install Unsloth for 2-5x faster training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

#### AMD GPU Users (ROCm)

```bash
# Install ROCm-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Install dependencies
pip install transformers datasets trl peft accelerate dearpygui
```

#### Intel/Other GPU Users (Vulkan/OpenCL)

```bash
# Install Vulkan and OpenCL detection
pip install vulkan pyopencl

# Windows: For DirectML acceleration (optional)
pip install torch-directml

# Cross-platform: ONNX Runtime for inference acceleration (optional)
pip install onnxruntime-gpu
```

#### Apple Silicon Users (MPS)

```bash
# MPS is built into PyTorch on macOS
pip install torch transformers datasets trl peft accelerate dearpygui
```

### Verifying GPU Detection

Run the included diagnostic tool to verify your GPU is properly detected:

```bash
python check_gpu.py
```

This will check your GPU drivers, PyTorch installation, and backend availability.

## Usage

### Starting the Application

```bash
python NTTuner.py
```

### Basic Training Workflow

1. **Select a Base Model**

   * Choose from the dropdown (includes installed Ollama models and popular options)
   * Or enter a custom HuggingFace model name
2. **Prepare Your Dataset**

   * Format as JSONL with a `text` field per line
   * Drag and drop the file onto the interface, or click Browse
3. **Configure Training Parameters**

   * LoRA Rank: Higher values train more parameters (typically 16-64)
   * Epochs: Number of training passes (start with 1-3)
   * Batch Size: Adjust based on available GPU memory
   * Learning Rate: Usually between 1e-5 and 5e-4
4. **Set Output Options**

   * Choose a name for your fine-tuned model
   * Select output directory
   * Pick quantization level (q5\_k\_m is a good balance)
5. **Start Training**

   * Click "Start Training"
   * Monitor progress in the log window
   * Training runs in the background
6. **Use Your Model**

   * After training completes, the model is automatically imported to Ollama
   * Test it: `ollama run your-model-name`

### Dataset Format

Your training data should be a JSONL file where each line contains a JSON object with a `text` field:

```json
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is machine learning?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMachine learning is a subset of artificial intelligence...<|eot_id|>"}
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nExplain neural networks<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNeural networks are computing systems inspired by biological neural networks...<|eot_id|>"}
```

The exact format depends on your base model's chat template. Consult the model's documentation for the correct formatting.

## Configuration Files

### Saving Configurations

Click "Save Config" to save your current settings as a JSON file. This is useful for:

* Reproducing training runs
* Sharing configurations with others
* Maintaining different setups for different projects

### Loading Configurations

Click "Load Config" to restore previously saved settings. All parameters will be populated automatically.

## Troubleshooting

### GPU Not Detected

If your GPU isn't being recognized:

**For NVIDIA:**
1. Verify drivers are installed: `nvidia-smi`
2. Reinstall PyTorch with CUDA support (see installation above)
3. Run `check_gpu.py` for detailed diagnostics

**For AMD:**
1. Install ROCm-enabled PyTorch
2. Verify ROCm installation: `rocm-smi`

**For Intel/Other:**
1. Install detection libraries: `pip install vulkan pyopencl`
2. The tool will detect your GPU and provide guidance

### Training is Slow

**For NVIDIA GPU users:**
* Install Unsloth for 2-5x speedup
* Ensure CUDA-enabled PyTorch is installed
* Check GPU utilization with `nvidia-smi`

**For AMD GPU users:**
* Ensure ROCm PyTorch is installed
* Training will be slower than NVIDIA+Unsloth but faster than CPU

**For Intel/Other GPU users:**
* GPU is detected but PyTorch training defaults to CPU
* Consider installing torch-directml (Windows) for acceleration
* Or use ONNX Runtime for inference acceleration
* Or use cloud GPU services for training (Vast.ai, RunPod, Google Colab)

**For CPU users:**
* Training on CPU is expected to be extremely slow
* Consider using cloud GPU services
* Use smaller models like TinyLlama for testing

### Out of Memory Errors

If training fails with OOM errors:

* Reduce batch size to 1
* Increase gradient accumulation steps
* Lower max sequence length
* Reduce LoRA rank
* Use a smaller base model

### Model Import Fails

If Ollama import fails:

* Verify Ollama is installed: `ollama --version`
* Check if the GGUF file was created in the output directory
* Try manual import: `cd output_directory && ollama create model-name -f Modelfile`

## System Requirements

### Minimum Requirements

* **OS**: Windows 10/11, Linux, or macOS
* **RAM**: 8GB (16GB recommended)
* **Storage**: 10GB free space for models and outputs
* **Python**: 3.10 or higher

### Recommended Requirements

* **GPU**: Any modern GPU (NVIDIA 8GB+, AMD 8GB+, Intel Arc, Apple M1/M2/M3)
* **RAM**: 16GB or more
* **Storage**: SSD with 50GB+ free space

### Model Size Guidelines

| Model Size | Minimum VRAM | Recommended VRAM | Training Time (est) |
| --- | --- | --- | --- |
| 1B params | 6GB | 8GB | 1-2 hours |
| 3B params | 8GB | 12GB | 2-4 hours |
| 7B params | 12GB | 16GB | 4-8 hours |
| 13B params | 16GB | 24GB | 8-16 hours |

Times are estimates for 1 epoch on 1000 examples with NVIDIA GPU + Unsloth. Other backends will be slower.

## What's Different from Original

### Enhanced GPU Detection

The enhanced version adds comprehensive GPU detection for non-NVIDIA hardware:

* **Vulkan API Detection**: Identifies Intel, AMD, and other GPUs
* **OpenCL Detection**: Universal GPU compatibility layer
* **ROCm Enhancement**: Better AMD GPU configuration
* **Smart Fallback**: Automatic selection of best available backend

### Startup Information

```
GPU DETECTION:
  PyTorch 2.x.x
  OpenCL 3.0
  Vendor: Intel
  Device: Intel(R) UHD Graphics 620
  VRAM: 4.0 GB
  NOTE: Using OpenCL backend - may require PlaidML or ONNX Runtime

FEATURES:
  ✓ NTCompanion JSONL format support
  ✓ Enhanced GPU detection (CUDA/ROCm/MPS/Vulkan/OpenCL)
  ✓ Non-NVIDIA GPU support (Intel, AMD via Vulkan/OpenCL)
  ✓ Dataset validation and statistics
  ✓ VRAM usage estimation
  ✓ Progress tracking with ETA
  ✓ Auto-configuration
  ✓ Advanced GGUF export options
```

### Installation Guidance

The tool now provides specific installation instructions based on detected hardware:

* **AMD GPU**: Suggests ROCm PyTorch installation
* **Intel/Other GPU**: Recommends Vulkan/OpenCL libraries and acceleration options
* **No GPU**: Suggests cloud GPU services

### 100% Backward Compatible

All original functionality is preserved:
* ✅ NVIDIA CUDA support unchanged
* ✅ Unsloth integration unchanged
* ✅ All training features unchanged
* ✅ GGUF export unchanged
* ✅ Ollama integration unchanged

## Advanced Usage

### Custom LoRA Targets

The application targets these modules by default:

* q\_proj, k\_proj, v\_proj, o\_proj
* gate\_proj, up\_proj, down\_proj

For custom targeting, modify the `target_modules` list in the code.

### Manual GGUF Conversion

If you're using CPU training or Unsloth isn't available, you'll need to manually convert to GGUF:

1. Training saves a merged HuggingFace model
2. Install llama.cpp: `git clone https://github.com/ggerganov/llama.cpp`
3. Convert: `python llama.cpp/convert-hf-to-gguf.py merged_model --outtype f16`
4. Quantize: `llama.cpp/llama-quantize model-f16.gguf model-q5_k_m.gguf q5_k_m`
5. Import: `ollama create model-name -f Modelfile`

## Performance Expectations

### Backend Performance Comparison

| Backend | Relative Speed | Notes |
|---------|---------------|-------|
| CUDA + Unsloth | 100% (baseline) | Best performance, NVIDIA only |
| CUDA Standard | 40% | Without Unsloth |
| ROCm | 35-40% | AMD GPUs, native PyTorch |
| MPS | 30-35% | Apple Silicon, unified memory |
| Vulkan (with DirectML) | 15-20% | Windows Intel/AMD |
| OpenCL (with PlaidML) | 15-20% | Various platforms |
| CPU | 5-10% | Slowest but works everywhere |

## Contributing

Contributions are welcome. If you encounter bugs or have feature requests, please open an issue on GitHub.

## License

This project is provided as-is for educational and research purposes. Please respect the licenses of any base models and datasets you use with this tool.

Built with:

* [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
* [Transformers](https://github.com/huggingface/transformers) - Model architecture and training
* [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
* [TRL](https://github.com/huggingface/trl) - Transformer reinforcement learning
* [DearPyGUI](https://github.com/hoffstadt/DearPyGui) - GPU-accelerated GUI framework
* [Ollama](https://ollama.ai) - Local LLM runtime

## Links

* **Repository**: <https://github.com/noosed/nttuner>
* **Issues**: Fork and fix!
* **Ollama**: <https://ollama.ai>
* **Unsloth**: <https://github.com/unslothai/unsloth>


---

Created by [github.com/noosed](https://github.com/noosed)
