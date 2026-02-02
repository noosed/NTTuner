# Quick Start Guide - CPU Users

If you're getting the "Unsloth cannot find any torch accelerator" error, this guide is for you!

## The Problem

Unsloth requires a CUDA GPU and won't work on CPU-only systems. This is by design - Unsloth is optimized for fast GPU training.

## The Solution

Use the **ollama_trainer_v2.py** version which supports both GPU and CPU training:

```bash
python ollama_trainer_v2.py
```

This version will:
- ✓ Detect if you have a GPU
- ✓ Use Unsloth if GPU available (2-5x faster)
- ✓ Fall back to standard transformers on CPU
- ✓ Provide clear warnings about performance

## Installation (CPU Users)

```bash
# Install core dependencies (no GPU required)
pip install torch transformers datasets trl peft accelerate dearpygui

# Do NOT install unsloth or bitsandbytes on CPU
```

## Installation (GPU Users)

```bash
# Install everything for maximum speed
pip install torch transformers datasets trl peft accelerate dearpygui bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Important Notes for CPU Training

⚠️ **Training on CPU is EXTREMELY slow** - it may take hours or days for even small models.

### Recommendations:

1. **Use a smaller model**: 
   - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parameters)
   - microsoft/phi-2 (2.7B parameters)
   
2. **Reduce settings**:
   - Batch Size: 1
   - LoRA Rank: 16-32 (lower = faster)
   - Epochs: 1
   - Max Sequence Length: 512 or lower

3. **Use a free GPU instead**:
   - Google Colab (free tier includes GPU)
   - Kaggle Notebooks (free GPU)
   - Lightning.ai (free credits)

### Google Colab Example

```python
# In a Colab notebook (with GPU enabled):
!pip install unsloth transformers datasets trl peft accelerate

# Upload this script and run it
# Or use the Unsloth notebooks: https://github.com/unslothai/unsloth
```

## Testing Your Installation

Run this to check your setup:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

Expected output:
- **CPU**: `CUDA available: False`
- **GPU**: `CUDA available: True` + device name

## Sample Workflow (CPU)

For CPU users wanting to test the tool:

1. Create a tiny test dataset (5-10 examples)
2. Use TinyLlama model
3. Set: batch_size=1, lora_rank=16, epochs=1, max_seq_length=256
4. Expect 10-30 minutes for tiny test

This lets you verify everything works before committing to a real training run on GPU.

## File Comparison

- **ollama_trainer.py** - Original (requires GPU + Unsloth)
- **ollama_trainer_v2.py** - New (works on CPU or GPU)

Use v2 for maximum compatibility!
