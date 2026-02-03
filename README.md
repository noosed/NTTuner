<img width="1000" height="720" alt="image" src="https://github.com/user-attachments/assets/da945fa6-0bdb-4d7c-b209-4e73ca07824e" />
# NTTuner - LLM Fine-Tuning Made Simple

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/noosed/NTTuner)](https://github.com/noosed/NTTuner/stargazers)



**Desktop GUI for fine-tuning Large Language Models with automatic Ollama deployment**

NTTuner is a user-friendly desktop application that makes fine-tuning LLMs accessible to everyone. Whether you're a researcher, developer, or AI enthusiast, NTTuner handles the complexity of LoRA training, GGUF conversion, and Ollama integration through an intuitive graphical interface.

---

## 🌟 Key Features

### 🚀 **One-Click Fine-Tuning**
- **GPU Acceleration**: Automatic CUDA detection with 2-5x speedup via Unsloth
- **CPU Fallback**: Full CPU support (slower but functional on any machine)
- **LoRA Training**: Parameter-efficient fine-tuning with configurable rank/alpha
- **Background Processing**: Non-blocking UI that stays responsive during training

### 🎯 **Seamless Ollama Integration**
- **Auto-Import**: Trained models automatically deploy to your Ollama instance
- **Model Discovery**: Detects and lists all installed Ollama models
- **Built-in Downloader**: Download popular models directly from the interface
- **Custom Models**: Support for any HuggingFace-compatible model

### 💾 **Smart Dataset Management**
- **Drag & Drop**: Drop JSONL files directly onto the interface
- **Format Validation**: Automatic checking for proper formatting
- **Example Templates**: Built-in examples for common chat formats
- **Batch Processing**: Handle datasets of any size

### ⚙️ **Advanced Configuration**
- **Preset Profiles**: Quick-start templates for common use cases
- **Save/Load Configs**: Reuse training configurations as JSON
- **Hyperparameter Control**: Fine-tune learning rate, batch size, epochs, etc.
- **Quantization Options**: Choose from q4_k_m to f16 based on your needs

### 📊 **Real-Time Monitoring**
- **Live Training Logs**: Watch progress as it happens
- **GPU Utilization**: Monitor VRAM and compute usage
- **Loss Tracking**: See training and validation metrics
- **ETA Estimates**: Know how long training will take

---

## 📥 Installation

### Quick Start (5 Minutes)

```bash
# 1. Clone the repository
git clone https://github.com/noosed/NTTuner.git
cd NTTuner

# 2. Install core dependencies
pip install torch transformers datasets trl peft accelerate dearpygui

# 3. Run NTTuner
python NTTuner.py
```

### GPU Installation (Recommended)

For NVIDIA GPU users, get significant speedup with CUDA support:

```bash
# 1. Install CUDA-enabled PyTorch (replace cu121 with your CUDA version)
pip uninstall torch torchvision torchaudio  # Remove CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install GPU acceleration libraries
pip install bitsandbytes

# 3. (Optional) Install Unsloth for 2-5x faster training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 4. Verify GPU detection
python check_gpu.py
```

**CUDA Version Reference:**
- CUDA 11.8: `cu118`
- CUDA 12.1: `cu121`
- CUDA 12.4: `cu124`

Check your CUDA version: `nvcc --version` or `nvidia-smi`

### System Requirements

**Minimum:**
- Python 3.10+
- 8GB RAM
- 10GB free disk space
- Windows 10/11, Linux, or macOS

**Recommended:**
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- 16GB+ RAM
- 50GB+ SSD storage
- CUDA 11.8 or newer

**Model Size Guidelines:**

| Model Size | Min VRAM | Recommended VRAM | Est. Training Time* |
|-----------|----------|------------------|---------------------|
| 1B params | 6GB | 8GB | 1-2 hours |
| 3B params | 8GB | 12GB | 2-4 hours |
| 7B params | 12GB | 16GB | 4-8 hours |
| 13B params | 16GB | 24GB | 8-16 hours |

*For 1 epoch on 1000 examples with typical settings

---

## 🎓 Usage Guide

### Step 1: Prepare Your Dataset

Your training data should be JSONL format with a `text` field per line:

```jsonl
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is machine learning?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMachine learning is a subset of AI that enables systems to learn from data...<|eot_id|>"}
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nExplain neural networks<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNeural networks are computing systems inspired by biological brains...<|eot_id|>"}
```

**Important:** The exact format depends on your base model's chat template (Llama, Mistral, Qwen, etc.). See [Chat Templates](#chat-templates) below.

**Need help creating datasets?** Use [NTCompanion](https://github.com/noosed/NTCompanion) to scrape and format web content automatically!

### Step 2: Configure Training

**Basic Configuration:**

1. **Select Base Model**
   - Choose from dropdown (includes installed Ollama models)
   - Or enter a HuggingFace model name (e.g., `meta-llama/Llama-3.2-3B-Instruct`)

2. **Load Dataset**
   - Drag & drop your JSONL file
   - Or click "Browse" to select

3. **Set Output Name**
   - Give your fine-tuned model a name (e.g., `my-custom-llama`)

**Hyperparameters Explained:**

| Parameter | What It Does | Typical Range | Recommendation |
|-----------|-------------|---------------|----------------|
| **LoRA Rank** | Number of parameters to train | 8-128 | 16-64 (higher = more capacity) |
| **LoRA Alpha** | Scaling factor | Same as rank | Set equal to rank |
| **Epochs** | Training passes over data | 1-10 | Start with 1-3 |
| **Batch Size** | Examples per update | 1-32 | 4 (adjust for VRAM) |
| **Gradient Accumulation** | Updates to accumulate | 1-16 | 4 (increases effective batch) |
| **Learning Rate** | Step size | 1e-6 to 5e-4 | 2e-5 (standard) |
| **Max Sequence Length** | Max tokens per example | 512-4096 | 2048 (model dependent) |
| **Warmup Steps** | Gradual LR increase | 0-500 | 100 (helps stability) |

**Quick Profiles:**

```
Fast Experimentation:
├─ Rank: 16
├─ Epochs: 1
├─ Batch: 4
├─ LR: 2e-5
└─ Time: ~30min (3B model, 500 examples)

Balanced Quality:
├─ Rank: 32
├─ Epochs: 3
├─ Batch: 4
├─ LR: 2e-5
└─ Time: ~2hrs (3B model, 1000 examples)

Maximum Quality:
├─ Rank: 64
├─ Epochs: 5
├─ Batch: 2
├─ LR: 1e-5
└─ Time: ~8hrs (3B model, 2000 examples)
```

### Step 3: Start Training

1. Click **"Start Training"**
2. Monitor progress in the log window
3. Training runs in background (UI stays responsive)
4. On completion:
   - Model is automatically converted to GGUF
   - Imported to Ollama with your chosen name
   - Ready to use immediately

### Step 4: Test Your Model

```bash
# Run your fine-tuned model
ollama run my-custom-llama

# Test it
>>> Tell me about the topics you were trained on
```

---

## 🎨 Chat Templates

Different models use different formatting. Here are the most common:

### Llama 3.1/3.2/3.3 Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi! How can I help?<|eot_id|>
```

### Mistral/Mixtral Format
```
<s>[INST] Hello! [/INST] Hi! How can I help?</s>
```

### Qwen Format
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help?<|im_end|>
```

### Phi-4 Format
```
<|system|>
You are a helpful assistant<|end|>
<|user|>
Hello!<|end|>
<|assistant|>
Hi! How can I help?<|end|>
```

### Gemma Format
```
<bos><start_of_turn>system
You are a helpful assistant<end_of_turn>
<start_of_turn>user
Hello!<end_of_turn>
<start_of_turn>model
Hi! How can I help?<end_of_turn>
```

**Pro Tip:** Use [NTCompanion](https://github.com/noosed/NTCompanion) which automatically formats datasets with the correct template for your chosen model!

---

## 🔧 Advanced Features

### Custom LoRA Target Modules

By default, NTTuner targets these modules:
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention layers)
- `gate_proj`, `up_proj`, `down_proj` (MLP layers)

To customize, edit the `target_modules` list in the code.

### Quantization Options

Choose the right quantization for your use case:

| Quantization | Size | Quality | Speed | Best For |
|--------------|------|---------|-------|----------|
| **q4_k_m** | Smallest | Good | Fastest | Low VRAM, quick inference |
| **q5_k_m** | Medium | Better | Fast | Balanced (recommended) |
| **q6_k** | Larger | Great | Medium | Quality-focused |
| **q8_0** | Large | Excellent | Slower | High quality |
| **f16** | Largest | Perfect | Slowest | Maximum quality, evaluation |

### Manual GGUF Conversion (If Needed)

If auto-conversion fails:

```bash
# 1. Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 2. Convert to GGUF
python llama.cpp/convert-hf-to-gguf.py /path/to/merged_model --outtype f16

# 3. Quantize
./llama.cpp/llama-quantize model-f16.gguf model-q5_k_m.gguf q5_k_m

# 4. Create Modelfile
echo "FROM ./model-q5_k_m.gguf" > Modelfile

# 5. Import to Ollama
ollama create my-model -f Modelfile
```

### Configuration Files

**Save Configuration:**
- Click "Save Config" to save all settings as JSON
- Useful for reproducing training runs
- Share configs with collaborators

**Load Configuration:**
- Click "Load Config" to restore saved settings
- All parameters populate automatically

Example config file:
```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "dataset_path": "/path/to/dataset.jsonl",
  "output_name": "my-model",
  "lora_rank": 32,
  "lora_alpha": 32,
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 2e-5,
  "max_seq_length": 2048
}
```

---

## 🐛 Troubleshooting

### GPU Not Detected

**Symptoms:** Training is extremely slow, GPU not showing in logs

**Solutions:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.version.cuda)  # Check CUDA version
   ```
3. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
4. Run diagnostic: `python check_gpu.py`

### Out of Memory Errors

**Symptoms:** `CUDA out of memory` or similar error

**Solutions:**
1. **Reduce batch size** to 1 or 2
2. **Increase gradient accumulation** to 8 or 16
3. **Lower max sequence length** to 1024 or 512
4. **Reduce LoRA rank** to 8 or 16
5. **Use a smaller base model** (3B instead of 7B)
6. **Enable gradient checkpointing** (automatic in NTTuner)
7. **Use quantized base model** (4-bit loading)

### Training is Very Slow

**For CPU users:**
- CPU training is 10-50x slower than GPU
- Consider using Google Colab (free GPU) or cloud services
- Use smaller models (1B-3B) for experimentation

**For GPU users:**
- Install Unsloth: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
- Verify CUDA PyTorch: `python -c "import torch; print(torch.version.cuda)"`
- Check GPU utilization: `nvidia-smi` (should show high usage during training)
- Ensure batch size isn't too small

### Dataset Format Errors

**Symptoms:** `KeyError: 'text'` or format validation fails

**Solutions:**
1. Verify JSONL format (one JSON object per line)
2. Each line must have a `"text"` field
3. No trailing commas or comments
4. Use proper chat template for your model
5. Validate with: `python -m json.tool < dataset.jsonl`

**Example validation script:**
```python
import json

with open('dataset.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            assert 'text' in obj, f"Line {i}: Missing 'text' field"
            assert len(obj['text']) > 0, f"Line {i}: Empty text"
        except json.JSONDecodeError as e:
            print(f"Line {i}: Invalid JSON - {e}")
```

### Ollama Import Fails

**Symptoms:** Model trains but doesn't appear in Ollama

**Solutions:**
1. Verify Ollama is installed: `ollama --version`
2. Check if Ollama is running: `ollama list`
3. Verify GGUF file exists in output directory
4. Try manual import:
   ```bash
   cd /path/to/output
   echo "FROM ./model.gguf" > Modelfile
   ollama create my-model -f Modelfile
   ```
5. Check Ollama logs for errors

### Model Quality Issues

**Symptoms:** Model gives poor or nonsensical outputs

**Solutions:**
1. **Increase training epochs** to 3-5
2. **Increase LoRA rank** to 32-64
3. **Improve dataset quality** (more examples, better formatting)
4. **Adjust learning rate** (try 1e-5 or 5e-5)
5. **Ensure proper chat template** formatting
6. **Add more diverse examples** to dataset
7. **Check base model is appropriate** for your task

---

## 📚 Best Practices

### Dataset Creation

✅ **Do:**
- Use 500-5000 high-quality examples
- Maintain consistent formatting
- Include diverse examples
- Validate JSON before training
- Test with small dataset first

❌ **Don't:**
- Mix different chat templates
- Include corrupted/truncated examples
- Use extremely long sequences (>4K tokens)
- Train on low-quality data
- Ignore validation errors

### Training Configuration

✅ **Do:**
- Start with recommended defaults
- Save configurations for reproduction
- Monitor training logs for issues
- Test models after training
- Iterate based on results

❌ **Don't:**
- Use learning rate >1e-4 without testing
- Set batch size too high for your VRAM
- Skip warmup steps
- Ignore convergence issues
- Overtrain (too many epochs)

### Model Deployment

✅ **Do:**
- Test thoroughly before production
- Document your training configuration
- Keep original dataset for iteration
- Monitor model performance
- Version your models

❌ **Don't:**
- Deploy untested models
- Delete training data immediately
- Ignore user feedback
- Skip quality evaluation
- Use inappropriate quantization

---

## 🔬 Technical Details

### LoRA (Low-Rank Adaptation)

NTTuner uses LoRA for parameter-efficient fine-tuning:

- **Rank (r)**: Determines trainable parameters (higher = more capacity)
- **Alpha (α)**: Scaling factor (typically set equal to rank)
- **Target Modules**: Specific layers to adapt
- **Trainable Params**: Typically 0.1-1% of full model

**Formula:** Full weight = Frozen base + (LoRA_A × LoRA_B × α/r)

### Training Process

1. **Load Base Model**: HuggingFace model + tokenizer
2. **Prepare LoRA**: Add adapter layers to target modules
3. **Load Dataset**: JSONL → HuggingFace Dataset
4. **Training Loop**: SFTTrainer with specified hyperparameters
5. **Merge Adapters**: Combine LoRA weights with base model
6. **Export GGUF**: Convert to GGUF format for Ollama
7. **Quantize**: Apply selected quantization level
8. **Import**: Add to Ollama model library

### GPU Acceleration

**With Unsloth:**
- Custom CUDA kernels for 2-5x speedup
- Memory-efficient attention
- Optimized backward pass
- Automatic mixed precision

**Standard Training:**
- PyTorch native CUDA
- Flash Attention 2 (if available)
- Gradient checkpointing
- BitsAndBytes quantization

---

## 🤝 Contributing

Contributions welcome! If you find bugs or have feature requests:

1. Check existing issues
2. Create detailed bug report or feature request
3. Include system info, logs, and reproducible steps
4. Submit pull requests with clear descriptions

---

## 📖 Additional Resources

### Official Links
- **Repository**: https://github.com/noosed/NTTuner
- **NTCompanion** (Dataset Creator): https://github.com/noosed/NTCompanion
- **Ollama**: https://ollama.ai
- **Unsloth**: https://github.com/unslothai/unsloth

### Learning Resources
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Chat Templates Guide](https://huggingface.co/docs/transformers/chat_templating)
- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)

### Community
- GitHub Issues: [Report bugs/requests](https://github.com/noosed/NTTuner/issues)
- Ollama Discord: https://discord.gg/ollama

---

## 📄 License

This project is provided as-is for educational and research purposes. Please respect the licenses of any base models and datasets you use.

**Model Licenses:**
- Llama models: [Meta's Community License](https://ai.meta.com/llama/license/)
- Mistral models: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Qwen models: Check [HuggingFace page](https://huggingface.co/Qwen)

---

## 🙏 Acknowledgments

Built with excellent open-source tools:

- **[Unsloth](https://github.com/unslothai/unsloth)** - Fast LLM fine-tuning
- **[Transformers](https://github.com/huggingface/transformers)** - Model architecture
- **[PEFT](https://github.com/huggingface/peft)** - Parameter-efficient training
- **[TRL](https://github.com/huggingface/trl)** - Transformer RL and SFT
- **[DearPyGUI](https://github.com/hoffstadt/DearPyGui)** - GPU-accelerated interface
- **[Ollama](https://ollama.ai)** - Local LLM runtime
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - GGUF conversion

Special thanks to the AI/ML community for making these tools accessible!

---

## ⭐ Support This Project

If NTTuner helps you:
- ⭐ Star the repository
- 🐛 Report bugs and suggest features
- 📖 Improve documentation
- 💬 Share your success stories

**Created by [@noosed](https://github.com/noosed)**

---

## 🚀 Quick Command Reference

```bash
# Installation
git clone https://github.com/noosed/NTTuner.git
cd NTTuner
pip install torch transformers datasets trl peft accelerate dearpygui

# GPU Setup (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Run NTTuner
python NTTuner.py

# Check GPU
python check_gpu.py

# Test trained model
ollama run my-model

# List Ollama models
ollama list

# Remove model
ollama rm my-model
```

Happy fine-tuning! 🎉
Created by [noosed](https://github.com/noosed)
