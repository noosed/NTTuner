# NTTuner Quick Start Guide

Get up and running with NTTuner in under 10 minutes. This guide covers installation, your first fine-tuning run, and common workflows.

## Prerequisites

Before you begin:
- Python 3.10 or higher installed
- 8GB RAM minimum (16GB recommended)
- Ollama installed ([download here](https://ollama.ai))
- For GPU: NVIDIA GPU with 8GB+ VRAM and CUDA drivers

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/noosed/NTTuner.git
cd NTTuner
```

### Step 2: Install Dependencies

**For CPU-only training:**
```bash
pip install torch transformers datasets trl peft accelerate dearpygui
```

**For GPU training (recommended):**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
pip install transformers datasets trl peft accelerate dearpygui bitsandbytes

# Optional: Install Unsloth for 2-5x speedup
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 3: Verify Your Setup

Run the GPU detection script to ensure everything is working:

```bash
python check_gpu.py
```

You should see output indicating whether your GPU is detected and CUDA is available.

## Your First Training Run

### Launch the Application

```bash
python NTTuner.py
```

The GUI will open with several tabs: Settings, Training, and Logs.

### Configure Your Training

1. **Select a Base Model**
   - Open the "Model" dropdown
   - Choose from installed Ollama models or popular options
   - For first-time users, try "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (fast and small)

2. **Prepare Your Dataset**
   
   Create a simple test dataset file named `test_data.jsonl`:
   ```json
   {"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is AI?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAI stands for Artificial Intelligence, the simulation of human intelligence by machines.<|eot_id|>"}
   {"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nExplain machine learning<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMachine learning is a subset of AI where systems learn from data to improve performance.<|eot_id|>"}
   ```

3. **Load Your Dataset**
   - Click "Browse" in the Dataset section
   - Select your `test_data.jsonl` file
   - Or drag and drop the file onto the application window

4. **Configure Training Parameters**
   
   Use these settings for your first run:
   - **LoRA Rank**: 16 (good balance of quality and speed)
   - **LoRA Alpha**: 32 (typically 2x the rank)
   - **Epochs**: 1 (one complete pass through the data)
   - **Batch Size**: 2 (reduce to 1 if you get memory errors)
   - **Learning Rate**: 2e-4 (standard for LoRA)
   - **Max Sequence Length**: 512 (shorter = faster)

5. **Set Output Options**
   - **Model Name**: Choose a name like "my-first-model"
   - **Output Directory**: Leave default or specify a path
   - **Quantization**: q5_k_m (good balance of size and quality)

6. **Start Training**
   - Click "Start Training"
   - Switch to the "Logs" tab to monitor progress
   - Training will run in the background - the UI stays responsive

### Understanding the Training Process

During training, you'll see:
1. **Initialization**: Loading the base model and preparing LoRA adapters
2. **Training**: Progress bars showing epoch and step completion
3. **Merging**: Combining the LoRA weights with the base model
4. **GGUF Conversion**: Converting to Ollama-compatible format
5. **Import**: Automatically adding the model to Ollama

Training time depends on:
- Dataset size (more examples = longer training)
- Model size (larger models = more computation)
- Hardware (GPU is 10-100x faster than CPU)
- Sequence length (longer sequences = more memory)

### Test Your Model

Once training completes, test your new model:

```bash
ollama run my-first-model
```

Try asking questions similar to your training data to see how the model responds.

## Common Workflows

### Workflow 1: Quick Experimentation

Perfect for testing ideas quickly:

```bash
# Use a small model
Model: TinyLlama-1.1B-Chat
Dataset: 100-500 examples
Epochs: 1-2
LoRA Rank: 8-16
Batch Size: 4

# Expected time: 10-30 minutes on GPU
```

### Workflow 2: Quality Fine-Tuning

For production-ready models:

```bash
# Use a capable model
Model: Llama-3.2-3B-Instruct or Mistral-7B-Instruct
Dataset: 1000-10000 examples
Epochs: 2-3
LoRA Rank: 32-64
Batch Size: 2-4

# Expected time: 2-8 hours on GPU
```

### Workflow 3: CPU Training

When GPU isn't available:

```bash
# Use the smallest model possible
Model: TinyLlama-1.1B-Chat
Dataset: 50-200 examples maximum
Epochs: 1
LoRA Rank: 8
Batch Size: 1

# Expected time: 2-6 hours
# Note: Larger models may be impractical on CPU
```

## Working with Datasets

### Dataset Format

NTTuner expects JSONL format with a `text` field. The text should follow your model's chat template.

**Llama 3 format example:**
```json
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n[SYSTEM PROMPT]<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n[USER MESSAGE]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[ASSISTANT RESPONSE]<|eot_id|>"}
```

**Mistral format example:**
```json
{"text": "<s>[INST] [USER MESSAGE] [/INST] [ASSISTANT RESPONSE]</s>"}
```

### Creating Datasets

You can create datasets:
1. **Manually**: Write JSONL files with your training examples
2. **Using NTCompanion**: Scrape websites to automatically generate datasets
3. **From existing data**: Convert CSV, JSON, or text files to JSONL

### Using NTCompanion

NTCompanion is a companion tool for generating datasets from web content:

```bash
# Install NTCompanion dependencies
pip install dearpygui beautifulsoup4 mmh3

# Run NTCompanion
python NTCompanion.py
```

In NTCompanion:
1. Add URLs to scrape
2. Select content type (recipe, tutorial, article, etc.)
3. Choose your target model's chat template
4. Set quality threshold (50 for general, 65+ for high quality)
5. Click "Start Scraping"
6. Use the output `nttuner_dataset.jsonl` in NTTuner

## Saving and Loading Configurations

### Save Your Settings

After configuring training parameters:
1. Click "Save Config" in NTTuner
2. Choose a filename like `llama3-training-config.json`
3. The config includes all parameters, model selection, and paths

### Load Previous Settings

To reuse a configuration:
1. Click "Load Config"
2. Select your saved JSON file
3. All settings will be restored automatically

This is useful for:
- Reproducing training runs
- Sharing configurations with others
- Maintaining different setups for different projects

## Troubleshooting

### GPU Not Detected

**Problem**: Training runs on CPU despite having a GPU.

**Solutions**:
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch installation:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
3. Reinstall CUDA-enabled PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Out of Memory Errors

**Problem**: Training crashes with "CUDA out of memory" error.

**Solutions**:
1. Reduce batch size to 1
2. Reduce max sequence length (try 256 or 128)
3. Lower LoRA rank (try 8 or 16)
4. Use a smaller base model
5. Enable gradient checkpointing (in advanced settings)

### Training is Very Slow

**Problem**: Training takes hours for small datasets.

**For CPU users**:
- CPU training is naturally very slow (10-100x slower than GPU)
- Use the smallest possible model and dataset
- Consider cloud GPU services or Google Colab

**For GPU users**:
1. Verify GPU is actually being used: `nvidia-smi` during training
2. Install Unsloth for 2-5x speedup
3. Increase batch size if you have VRAM headroom
4. Check that CUDA is properly installed

### Model Import Fails

**Problem**: Training completes but model doesn't appear in Ollama.

**Solutions**:
1. Verify Ollama is installed: `ollama --version`
2. Check output directory for GGUF file
3. Manually import:
   ```bash
   cd [output_directory]
   ollama create my-model-name -f Modelfile
   ```

### Poor Model Quality

**Problem**: Fine-tuned model doesn't perform well.

**Causes and solutions**:
1. **Insufficient training data**: Add more examples (aim for 500+)
2. **Too few epochs**: Increase to 2-3 epochs
3. **Poor data quality**: Clean and validate your dataset
4. **Mismatched chat template**: Verify format matches base model
5. **Learning rate too high/low**: Try 1e-4 to 5e-4 range

## Next Steps

### Improve Your Model

- Gather more training data (quality over quantity)
- Use NTCompanion to scale dataset creation
- Experiment with different LoRA ranks and learning rates
- Train for more epochs if underfitting

### Advanced Features

- Custom LoRA target modules (edit code directly)
- Gradient accumulation for larger effective batch sizes
- Mixed precision training for faster training
- Multi-GPU training (requires code modifications)

### Share Your Work

- Export models to HuggingFace
- Share training configurations
- Contribute to the NTTuner community

## Getting Help

If you encounter issues:

1. Check the full README for detailed documentation
2. Review the console logs in the Logs tab
3. Run diagnostic tools: `python check_gpu.py`
4. Open an issue on GitHub with:
   - Your configuration (save and share the JSON)
   - Error messages from the logs
   - System information (GPU, OS, Python version)

## Tips for Success

1. **Start small**: Use a tiny dataset and model to verify your setup works
2. **Monitor training**: Watch the logs for errors or warnings
3. **Save configurations**: Save working configs for future reference
4. **Test incrementally**: Test after each training run before scaling up
5. **Use quality data**: 100 high-quality examples beat 1000 poor ones
6. **Be patient**: Good models take time and iteration to create

## Example: Complete Training Session

Here's a complete example from start to finish:

```bash
# 1. Setup
git clone https://github.com/noosed/NTTuner.git
cd NTTuner
pip install torch transformers datasets trl peft accelerate dearpygui bitsandbytes

# 2. Create test dataset
cat > training_data.jsonl << 'EOF'
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is Python?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nPython is a high-level programming language known for its simplicity and readability.<|eot_id|>"}
{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nExplain variables<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nVariables are containers that store data values in programming.<|eot_id|>"}
EOF

# 3. Launch NTTuner
python NTTuner.py

# 4. In the GUI:
# - Select model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
# - Load dataset: training_data.jsonl
# - Set LoRA rank: 16
# - Set epochs: 1
# - Output name: python-tutor
# - Click "Start Training"

# 5. Wait for training to complete (~15-30 minutes)

# 6. Test your model
ollama run python-tutor
# > What are functions in Python?
```

That's it! You've successfully fine-tuned your first local LLM with NTTuner.

---

**Happy fine-tuning!**

For more details, see the main [README](README.md) and [NTCompanion documentation](https://github.com/noosed/NTCompanion).
