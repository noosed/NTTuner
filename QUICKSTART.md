# Quick Start

Get NTTuner running in a few minutes.

## Install

```bash
git clone https://github.com/noosed/NTTuner.git
cd NTTuner
pip install torch transformers datasets trl peft accelerate dearpygui bitsandbytes
```

For faster training on NVIDIA GPUs:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Get a Dataset

Use [NTCompanion](https://github.com/noosed/NTCompanion) to create training data from websites, or make a JSONL file manually:

```jsonl
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>"}
```

## Train

1. Run `python NTTuner.py`
1. Pick a model, load your dataset
1. Click Auto-Config
1. Click Start Training
1. Wait for it to finish

## Use Your Model

```bash
ollama run your-model-name
```

## Advanced Export

Want multiple quantization sizes? Expand “Advanced GGUF Export”, check the box, and pick a preset like “Size Ladder” to export Q2 through Q8 in one go.

## Problems?

- GPU not found: Run `python check_gpu.py`
- Out of memory: Reduce batch size to 1
- Slow training: Install Unsloth, or use a cloud GPU

Full docs: <README.md>

-----

Related: [NTCompanion](https://github.com/noosed/NTCompanion) for dataset creation