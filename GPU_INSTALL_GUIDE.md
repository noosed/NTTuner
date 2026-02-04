# Quick Install Guide - NTTuner Enhanced

## Choose Your GPU Type

### 🟢 NVIDIA GPU (CUDA)
```bash
# Standard installation
pip install torch transformers datasets trl peft accelerate dearpygui

# For CUDA acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional: 4-bit/8-bit quantization
pip install bitsandbytes

# Optional: 2-5x speedup
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 🔴 AMD GPU (ROCm)
```bash
# Install ROCm-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Install other dependencies
pip install transformers datasets trl peft accelerate dearpygui
```

### 🔵 Intel/Other GPU (Vulkan/OpenCL)
```bash
# Core dependencies
pip install torch transformers datasets trl peft accelerate dearpygui

# For GPU detection
pip install vulkan pyopencl

# Windows: For GPU acceleration
pip install torch-directml

# Alternative: ONNX Runtime (cross-platform)
pip install onnxruntime-gpu
```

### 🍎 Apple Silicon (M1/M2/M3)
```bash
# MPS is built into PyTorch on macOS
pip install torch transformers datasets trl peft accelerate dearpygui
```

### 💻 No GPU (CPU Only)
```bash
# Minimum installation - training will be slow
pip install torch transformers datasets trl peft accelerate dearpygui
```

## Quick Test

After installation, run this to check GPU detection:
```python
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## What You Get

| GPU Type | Detection | Training | Speed |
|----------|-----------|----------|-------|
| NVIDIA + Unsloth | ✅ | ✅ Fast | 100% |
| NVIDIA Standard | ✅ | ✅ Good | 40% |
| AMD ROCm | ✅ | ✅ Good | 35-40% |
| Apple MPS | ✅ | ✅ Good | 30-35% |
| Intel Vulkan | ✅ | ⚠️ CPU* | 5-10% |
| Other OpenCL | ✅ | ⚠️ CPU* | 5-10% |
| CPU Only | ✅ | ✅ Slow | 5-10% |

*GPU detected but requires additional packages for acceleration


```bash
python NTTuner.py
```

The GUI will show your detected GPU and provide specific guidance for your hardware.

## Troubleshooting

**"No GPU detected"**
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with GPU support for your hardware

**"Vulkan/OpenCL detected but slow"**
- Install torch-directml (Windows) or ONNX Runtime for acceleration
- Or use cloud GPU for training

**"Training out of memory"**
- Reduce batch size to 1
- Reduce max sequence length
- Use smaller model
- Enable gradient checkpointing


- github.com/noosed/nttuner
