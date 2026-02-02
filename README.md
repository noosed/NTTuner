<img width="1000" height="720" alt="image" src="https://github.com/user-attachments/assets/da945fa6-0bdb-4d7c-b209-4e73ca07824e" />
## NTTuner - Professional LLM Fine-Tuning Studio

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm%20%7C%20MPS-orange.svg" alt="GPU Support">
  <img src="https://img.shields.io/badge/Ollama-Integration-purple.svg" alt="Ollama">
</p>

A professional desktop GUI application for fine-tuning large language models with LoRA and deploying them directly to Ollama. Features advanced GGUF export options, multi-GPU support, and seamless integration with the NTCompanion dataset engine.


## 🔗 Related Projects

|Project                                                 |Description                                                                                  |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
|**[NTCompanion](https://github.com/noosed/NTCompanion)**|Professional dataset engine for NTTuner - web scraping, data processing, and JSONL generation|
|**[NTTuner](https://github.com/noosed/NTTuner)**        |This project - LLM fine-tuning GUI with Ollama integration                                   |

-----

## ✨ Features

### Core Training

- **LoRA Fine-tuning** - Efficient parameter-efficient training with configurable rank, alpha, and dropout
- **Multi-GPU Support** - Automatic detection of CUDA (NVIDIA), ROCm (AMD), and MPS (Apple Silicon)
- **Unsloth Integration** - 2-5x training speedup on supported NVIDIA GPUs
- **CPU Fallback** - Full support for CPU-only training
- **Real-time Progress** - Live training metrics with ETA estimation

### Advanced GGUF Export *(New in 2026)*

- **Full Quantization Control** - All llama.cpp quantization types (Q2_K through F32, IQ series, BF16)
- **Batch Export** - Export multiple quantization levels in one operation
- **Preset System** - Quick presets like “All K-Quants”, “Size Ladder”, “IQ Series”
- **Importance Matrix Support** - Use imatrix files for optimized IQ quantization
- **Custom Flags** - Pass-through arguments to llama-quantize
- **Filename Patterns** - Customizable output naming with `{model_name}` and `{quant_type}` variables
- **LoRA-Only Export** - Export adapter without merging (smaller files, runtime application)
- **Auto Ollama Import** - Automatically register exported models with Ollama

### Dataset Support

- **NTCompanion Integration** - Perfect compatibility with NTCompanion JSONL output
- **Multi-format Support** - JSONL, JSON, and CSV datasets
- **Dataset Validation** - Automatic format detection and statistics
- **Preview System** - Inspect dataset entries before training

### User Experience

- **Modern Dark UI** - Clean DearPyGui interface
- **Configuration Management** - Save/load training configurations as JSON
- **Auto-Configuration** - Hardware-aware automatic parameter tuning
- **Detailed Logging** - Comprehensive training logs with timestamps

-----

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Ollama installed ([download here](https://ollama.ai))
- NVIDIA GPU with CUDA (recommended) or AMD GPU with ROCm or Apple Silicon

### Quick Install

```bash
# Clone the repository
git clone https://github.com/noosed/NTTuner.git
cd NTTuner

# Install core dependencies
pip install torch transformers datasets trl peft accelerate dearpygui bitsandbytes
```

### GPU-Specific Installation

<details>
<summary><b>NVIDIA CUDA (Recommended)</b></summary>

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth for 2-5x faster training (optional but recommended)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# For advanced GGUF export, install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j
```

</details>

<details>
<summary><b>AMD ROCm</b></summary>

```bash
# Install ROCm-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

</details>

<details>
<summary><b>Apple Silicon (MPS)</b></summary>

```bash
# PyTorch with MPS support (included in standard install)
pip install torch torchvision torchaudio
```

</details>

### Verify Installation

```bash
python check_gpu.py
```

-----

## 🚀 Quick Start

### 1. Launch NTTuner

```bash
python NTTuner.py
```

### 2. Prepare Your Dataset

Use **[NTCompanion](https://github.com/noosed/NTCompanion)** to create training datasets, or prepare a JSONL file manually:

```jsonl
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is machine learning?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMachine learning is a subset of artificial intelligence...<|eot_id|>"}
```

### 3. Configure Training

|Parameter    |Recommended|Description                                     |
|-------------|-----------|------------------------------------------------|
|LoRA Rank    |32-64      |Higher = more parameters, better quality, slower|
|LoRA Alpha   |64-128     |Usually 2x the rank                             |
|Epochs       |1-3        |More epochs risk overfitting                    |
|Batch Size   |1-2        |Increase if VRAM allows                         |
|Learning Rate|2e-4       |Lower for larger models                         |

### 4. Train and Export

1. Click **Start Training**
1. Monitor progress in the log window
1. Model automatically exports to GGUF and imports to Ollama

### 5. Test Your Model

```bash
ollama run your-model-name
```

-----

## 🔧 Advanced GGUF Export

The new Advanced GGUF Export panel provides full control over the quantization process.

### Enabling Advanced Export

1. Expand the **“Advanced GGUF Export”** section
1. Check **“Use advanced GGUF export instead of default”**
1. Configure your export options

### Quantization Types

|Category          |Types                                   |Use Case                                  |
|------------------|----------------------------------------|------------------------------------------|
|**K-Quants**      |Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0|General purpose, good quality/size balance|
|**I-Quants**      |IQ2_M, IQ3_M, IQ4_XS, IQ4_NL            |Requires imatrix, best quality at size    |
|**Legacy**        |Q4_0, Q4_1, Q5_0, Q5_1                  |Compatibility with older llama.cpp        |
|**Full Precision**|F16, BF16, F32                          |Maximum quality, largest files            |

### Presets

|Preset                   |Outputs                                 |Best For                       |
|-------------------------|----------------------------------------|-------------------------------|
|Standard Quality (Q4_K_M)|Q4_K_M                                  |Daily use, good balance        |
|High Quality (Q5_K_M)    |Q5_K_M                                  |Better quality, slightly larger|
|Size Ladder (Q2→Q8)      |Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0|Testing different sizes        |
|All K-Quants             |All K-quant variants                    |Comprehensive export           |
|IQ Series                |IQ2_M, IQ3_M, IQ4_XS, IQ4_NL            |Best with imatrix              |

### Using Importance Matrix (imatrix)

For IQ quantization types, an importance matrix improves quality:

```bash
# Generate imatrix with llama.cpp
./llama-imatrix -m model-f16.gguf -f calibration_data.txt -o model.imatrix
```

Then specify the `.imatrix` or `.dat` file in the “Importance Matrix” field.

### Filename Patterns

Customize output filenames using variables:

- `{model_name}` - Your output model name
- `{quant_type}` - The quantization type (lowercase)

Example: `{model_name}-{quant_type}` → `my-model-q4_k_m.gguf`

### Export Without Training

To re-export an existing trained model:

1. Set the correct Output Dir and Model Name
1. Configure GGUF export options
1. Click **“Export GGUF Now (existing model)”**

-----

## 📁 Project Structure

```
NTTuner/
├── NTTuner.py              # Main application
├── CUDA_wuda.py            # CUDA diagnostics utility
├── check_gpu.py            # GPU detection script
├── QUICKSTART.md           # Quick start guide
├── README.md               # This file
└── fine_tuned_output/      # Default output directory
    └── your-model/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── training_manifest.json
        └── gguf/
            ├── your-model-q4_k_m.gguf
            ├── your-model-q5_k_m.gguf
            └── Modelfile
```

-----

## 🔍 Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Run diagnostics
python check_gpu.py
```

### Out of Memory (OOM)

- Reduce batch size to 1
- Increase gradient accumulation (effective batch = batch_size × grad_accum)
- Lower max sequence length
- Reduce LoRA rank
- Use a smaller base model

### llama-quantize Not Found

For advanced GGUF export, install llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Add to PATH or specify in NTTuner
export PATH=$PATH:$(pwd)
```

Or specify the full path in the “llama-quantize Path” field.

### Training Very Slow

- **CPU users**: Training on CPU is inherently slow; use Google Colab or cloud GPUs
- **GPU users**:
  - Install Unsloth for 2-5x speedup
  - Verify GPU is being used: check log for “Using backend: CUDA”
  - Monitor with `nvidia-smi`

-----

## 📊 Hardware Guidelines

### VRAM Requirements

|Model Size|Minimum VRAM|Recommended|Training Time*|
|----------|------------|-----------|--------------|
|1B params |4GB         |8GB        |30-60 min     |
|3B params |8GB         |12GB       |1-2 hours     |
|7B params |12GB        |16GB       |2-4 hours     |
|13B params|16GB        |24GB       |4-8 hours     |

*Estimates for 1 epoch, 1000 examples, with Unsloth on RTX 3080

### Recommended Settings by VRAM

|VRAM |Batch Size|Grad Accum|Max Seq Len|LoRA Rank|
|-----|----------|----------|-----------|---------|
|6GB  |1         |4         |256        |16       |
|8GB  |1         |8         |512        |32       |
|12GB |1         |8         |1024       |64       |
|16GB+|2         |8         |2048       |64-128   |

-----

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
1. Create your feature branch (`git checkout -b feature/amazing-feature`)
1. Commit your changes (`git commit -m 'Add amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

-----

## 📜 License

This project is provided as-is for educational and research purposes. Please respect the licenses of any base models and datasets you use.

-----

## 🙏 Acknowledgments

Built with:

- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [Transformers](https://github.com/huggingface/transformers) - Model architecture
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) - Transformer reinforcement learning
- [DearPyGui](https://github.com/hoffstadt/DearPyGui) - GPU-accelerated GUI
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF quantization
- [Ollama](https://ollama.ai) - Local LLM runtime

-----

## 📬 Links

- **NTTuner**: https://github.com/noosed/NTTuner
- **NTCompanion** (Dataset Engine): https://github.com/noosed/NTCompanion
- **Ollama**: https://ollama.ai
- **Unsloth**: https://github.com/unslothai/unsloth
- **llama.cpp**: https://github.com/ggerganov/llama.cpp

-----

<p align="center">
  Created by <a href="https://github.com/noosed">github.com/noosed</a>
</p>