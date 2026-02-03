<img width="1000" height="720" alt="image" src="https://github.com/user-attachments/assets/da945fa6-0bdb-4d7c-b209-4e73ca07824e" />
# NTTuner

A desktop application for fine-tuning large language models and deploying them to Ollama. Supports NVIDIA, AMD, and Apple Silicon GPUs, with CPU fallback for those without dedicated graphics.

Built to work seamlessly with [NTCompanion](https://github.com/noosed/NTCompanion), which handles dataset creation through web scraping and data processing.


-----

## What It Does

NTTuner takes a base language model (like Llama, Mistral, or Phi) and fine-tunes it on your custom dataset using LoRA. After training, it converts the model to GGUF format and imports it directly into Ollama so you can start using it immediately.

The new Advanced GGUF Export feature gives you full control over quantization—choose from any llama.cpp quant type, export multiple sizes at once, or use importance matrices for better quality at smaller sizes.

-----

## Related Projects

- **[NTCompanion](https://github.com/noosed/NTCompanion)** - Dataset engine for NTTuner. Scrapes websites, processes content, and generates training-ready JSONL files.

-----

## Features

**Training**

- LoRA fine-tuning with configurable rank, alpha, and dropout
- Automatic GPU detection (CUDA, ROCm, MPS)
- Unsloth integration for 2-5x faster training on NVIDIA GPUs
- Real-time progress with loss tracking and ETA
- Auto-configuration based on your hardware

**GGUF Export**

- All llama.cpp quantization types (Q2_K through F32, IQ series, BF16)
- Presets for common workflows (single quant, size ladder, full export)
- Importance matrix support for IQ quantization
- Custom filename patterns
- LoRA-only export option
- Automatic Ollama import

**Dataset Support**

- Native NTCompanion JSONL format
- Also accepts JSON and CSV
- Built-in validation and preview

-----

## Installation

You’ll need Python 3.10+ and [Ollama](https://ollama.ai) installed.

```bash
git clone https://github.com/noosed/NTTuner.git
cd NTTuner

pip install torch transformers datasets trl peft accelerate dearpygui bitsandbytes
```

For NVIDIA GPUs, also install Unsloth for faster training:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

For advanced GGUF export features, you’ll need llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j
```

Verify your GPU is detected:

```bash
python check_gpu.py
```

-----

## Basic Usage

1. Run `python NTTuner.py`
1. Select a base model from the dropdown or enter a HuggingFace model name
1. Load your dataset (JSONL file with a `text` field per line)
1. Click Auto-Config to set parameters based on your hardware
1. Click Start Training
1. When finished, your model is automatically available in Ollama

Test it:

```bash
ollama run your-model-name
```

-----

## Dataset Format

NTTuner expects JSONL files where each line has a `text` field containing the full conversation:

```jsonl
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is Python?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nPython is a programming language...<|eot_id|>"}
```

The easiest way to create these is with [NTCompanion](https://github.com/noosed/NTCompanion), which handles the formatting automatically.

-----

## Advanced GGUF Export

By default, NTTuner exports a single Q5_K_M quantization. For more control:

1. Expand the “Advanced GGUF Export” section
1. Check “Use advanced GGUF export instead of default”
1. Choose a preset or select individual quantization types

**Presets**

|Preset                   |What It Exports                                     |
|-------------------------|----------------------------------------------------|
|Standard Quality (Q4_K_M)|Single Q4_K_M file, good balance of size and quality|
|High Quality (Q5_K_M)    |Single Q5_K_M file, slightly better quality         |
|Size Ladder (Q2→Q8)      |Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0            |
|All K-Quants             |Every K-quant variant                               |
|IQ Series                |IQ2_M, IQ3_M, IQ4_XS, IQ4_NL (best with imatrix)    |

**Quantization Types**

|Type  |Size        |Quality  |Notes                            |
|------|------------|---------|---------------------------------|
|Q2_K  |Smallest    |Lower    |For very constrained environments|
|Q3_K_M|Small       |Moderate |Good for testing                 |
|Q4_K_M|Medium      |Good     |Most common choice               |
|Q5_K_M|Medium-Large|Better   |Recommended for quality          |
|Q6_K  |Large       |High     |Near-F16 quality                 |
|Q8_0  |Larger      |Very High|Minimal quality loss             |
|F16   |Largest     |Maximum  |Full precision                   |

**Using Importance Matrices**

For IQ quantization types, an importance matrix improves output quality. Generate one with llama.cpp:

```bash
./llama-imatrix -m model-f16.gguf -f calibration_data.txt -o model.imatrix
```

Then specify the file in the “Importance Matrix” field.

**Export Existing Models**

To re-export a previously trained model with different quantization settings, set the output directory and model name to match your existing model, configure your export options, and click “Export GGUF Now (existing model)”.

-----

## Configuration Tips

**By VRAM**

|VRAM |Batch Size|Grad Accum|Max Seq Len|LoRA Rank|
|-----|----------|----------|-----------|---------|
|6GB  |1         |4         |256        |16       |
|8GB  |1         |8         |512        |32       |
|12GB |1         |8         |1024       |64       |
|16GB+|2         |8         |2048       |64-128   |

**General Guidelines**

- Start with Auto-Config and adjust from there
- Higher LoRA rank = more trainable parameters = better results but slower
- More epochs can improve quality but risk overfitting (1-3 is usually enough)
- If you run out of memory, reduce batch size first, then sequence length

-----

## Troubleshooting

**GPU not detected**

Run `python check_gpu.py` for diagnostics. Common issues:

- CPU-only PyTorch installed (reinstall with CUDA support)
- Outdated NVIDIA drivers
- CUDA version mismatch

**Out of memory**

- Set batch size to 1
- Reduce max sequence length
- Lower LoRA rank
- Try a smaller base model

**llama-quantize not found**

Advanced GGUF export requires llama.cpp. Either:

- Add llama.cpp to your PATH
- Specify the full path in “llama-quantize Path” field

**Training is slow**

- Install Unsloth for NVIDIA GPUs
- Verify GPU is being used (check log for “Using backend: CUDA”)
- CPU training is inherently slow—consider cloud GPUs

-----

## Project Structure

```
NTTuner/
├── NTTuner.py              # Main application
├── check_gpu.py            # GPU diagnostics
├── CUDA_wuda.py            # CUDA utilities
├── README.md
├── QUICKSTART.md
└── fine_tuned_output/      # Default output location
    └── your-model/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── training_manifest.json
        └── gguf/
            └── your-model-q4_k_m.gguf
```

-----

## Links

- [NTTuner](https://github.com/noosed/NTTuner) - This project
- [NTCompanion](https://github.com/noosed/NTCompanion) - Dataset generation
- [Ollama](https://ollama.ai) - Local LLM runtime
- [Unsloth](https://github.com/unslothai/unsloth) - Training acceleration
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF tools

-----

Created by [noosed](https://github.com/noosed)