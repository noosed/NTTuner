# Quick Start Guide

Get up and running with NTech LLM Tuner in under 5 minutes.

## Prerequisites Check

Before starting, verify you have:
- Python 3.10 or higher: `python --version`
- NVIDIA GPU (optional but recommended): `nvidia-smi`
- Ollama installed: `ollama --version`

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/noosed/nttuner.git
cd ntech-llm-tuner
```

### Step 2: Install Dependencies

**For GPU users (recommended):**
```bash
pip install dearpygui transformers datasets trl peft accelerate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes
```

**For CPU users:**
```bash
pip install dearpygui transformers datasets trl peft accelerate torch
```

### Step 3: Verify GPU (if applicable)

```bash
python check_gpu.py
```

You should see your GPU detected. If not, see the troubleshooting section in README.md.

## Your First Training Run

### Step 1: Launch the Application

```bash
python ollama_trainer_v2.py
```

### Step 2: Configure Training

1. **Select Model**: Choose "TinyLlama/TinyLlama-1.1B-Chat-v1.0" from dropdown
2. **Load Dataset**: Click "Browse" and select `example_dataset.jsonl`
3. **Set Output Name**: Enter "my-first-model"
4. **Leave other settings at defaults**

### Step 3: Start Training

1. Click "Start Training"
2. Watch the log window for progress
3. Training should complete in 5-15 minutes on GPU (much longer on CPU)

### Step 4: Test Your Model

Once training completes:
```bash
ollama run my-first-model
```

Try asking it questions about machine learning (the topic of the example dataset).

## Next Steps

### Create Your Own Dataset

1. Create a `.jsonl` file
2. Each line should be valid JSON with a `text` field
3. Format according to your model's chat template
4. Include 100+ examples for best results

Example line:
```json
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are helpful<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>"}
```

### Adjust Training Parameters

For better results:
- Increase epochs to 2-3
- Increase LoRA rank to 64
- Add more training examples
- Adjust learning rate if needed

### Save Your Configuration

Click "Save Config" to save your settings for future use.

## Common Issues

**GPU not detected:**
Run `check_gpu.py` for diagnostics and follow the instructions.

**Out of memory:**
Reduce batch size to 1 and increase gradient accumulation steps.

**Training very slow:**
You may be on CPU. Consider using a cloud GPU service or Google Colab.

**Model not importing to Ollama:**
Check the output directory for the GGUF file and Modelfile, then manually import:
```bash
cd gguf_export/my-first-model
ollama create my-first-model -f Modelfile
```

## Getting Help

- Check the full [README.md](README.md) for detailed documentation
- Review [troubleshooting section](README.md#troubleshooting) 
- Open an issue on GitHub if you're still stuck


# PyTorch - Choose ONE of the following:
# ---------------------------------------

# Option 1: CPU-only PyTorch (not recommended for training)
# torch>=2.1.0
# torchvision>=0.16.0
# torchaudio>=2.1.0

# Option 2: CUDA 12.1 PyTorch (recommended for NVIDIA GPUs)
# Install separately with:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Option 3: CUDA 11.8 PyTorch (for older GPUs)
# Install separately with:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU acceleration (optional but highly recommended for NVIDIA GPUs)
# ------------------------------------------------------------------
# bitsandbytes>=0.41.0

# Ultra-fast training (optional, requires CUDA GPU)
# -------------------------------------------------
# unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git

# Installation Instructions
# ========================
#
# Basic installation (CPU-only):
#   pip install -r requirements.txt
#   pip install torch torchvision torchaudio
#
# GPU installation (recommended):
#   pip install -r requirements.txt
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#   pip install bitsandbytes
#   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#
# Verify GPU detection:
#   python check_gpu.py


## What's Next?

- Try different base models
- Experiment with LoRA settings
- Create larger, more focused datasets
- Share your fine-tuned models with the community

Happy fine-tuning!
