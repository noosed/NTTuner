# NTech LLM Tuner
<img width="20" height="10" alt="image" src="https://github.com/user-attachments/assets/0ccbcb97-22bf-4341-a55c-a911afba879a" />

A desktop GUI application for fine-tuning large language models and deploying them directly to Ollama. Built for ease of use, with support for both GPU and CPU training.

## Overview

NTech LLM Tuner simplifies the process of fine-tuning language models by providing an intuitive interface that handles the complexity of LoRA training, GGUF conversion, and Ollama integration. Whether you're customizing a model for a specific task or experimenting with different training configurations, this tool streamlines the entire workflow.

## Features

### Training Capabilities
- **LoRA Fine-tuning**: Efficient parameter-efficient training with configurable rank and alpha
- **GPU Acceleration**: Automatic detection and utilization of CUDA GPUs
- **CPU Fallback**: Full support for CPU-only training (though significantly slower)
- **Unsloth Integration**: Optional 2-5x speedup with Unsloth library on supported GPUs
- **Background Training**: Non-blocking UI that remains responsive during training

### Model Management
- **Ollama Integration**: Automatically imports models into your local Ollama instance
- **Model Discovery**: Detects and lists all installed Ollama models
- **Download Support**: Built-in downloader for popular Ollama models
- **HuggingFace Support**: Direct integration with HuggingFace model hub

### User Interface
- **Drag and Drop**: Drop dataset files directly onto the interface
- **Configuration Management**: Save and load training configurations as JSON
- **Real-time Logging**: Live training progress and detailed diagnostics
- **Model Browser**: Categorized dropdown of popular models by size and purpose

### Output Options
- **Multiple Quantization Levels**: Choose from q4_k_m to f16 based on your needs
- **Custom Output Directories**: Specify where to save your trained models
- **Automatic GGUF Export**: Converts models to GGUF format for Ollama compatibility

## Installation

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended, but not required)
- Ollama installed ([download here](https://ollama.ai))

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/noosed/ntech-llm-tuner.git
cd ntech-llm-tuner

# Install core dependencies
pip install torch transformers datasets trl peft accelerate dearpygui
```

### GPU Installation (Recommended)

For NVIDIA GPU users, install CUDA-enabled PyTorch and additional acceleration libraries:

```bash
# Install CUDA PyTorch
pip uninstall torch torchvision torchaudio  # Remove CPU version if present
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install GPU acceleration libraries
pip install bitsandbytes

# Optional: Install Unsloth for 2-5x faster training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Verifying GPU Detection

Run the included diagnostic tool to verify your GPU is properly detected:

```bash
python check_gpu.py
```

This will check your NVIDIA drivers, PyTorch installation, and CUDA availability.

## Usage

### Starting the Application

```bash
python ollama_trainer_v2.py
```

### Basic Training Workflow

1. **Select a Base Model**
   - Choose from the dropdown (includes installed Ollama models and popular options)
   - Or enter a custom HuggingFace model name

2. **Prepare Your Dataset**
   - Format as JSONL with a `text` field per line
   - Drag and drop the file onto the interface, or click Browse

3. **Configure Training Parameters**
   - LoRA Rank: Higher values train more parameters (typically 16-64)
   - Epochs: Number of training passes (start with 1-3)
   - Batch Size: Adjust based on available GPU memory
   - Learning Rate: Usually between 1e-5 and 5e-4

4. **Set Output Options**
   - Choose a name for your fine-tuned model
   - Select output directory
   - Pick quantization level (q5_k_m is a good balance)

5. **Start Training**
   - Click "Start Training"
   - Monitor progress in the log window
   - Training runs in the background

6. **Use Your Model**
   - After training completes, the model is automatically imported to Ollama
   - Test it: `ollama run your-model-name`

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
- Reproducing training runs
- Sharing configurations with others
- Maintaining different setups for different projects

### Loading Configurations

Click "Load Config" to restore previously saved settings. All parameters will be populated automatically.

## Troubleshooting

### GPU Not Detected

If your GPU isn't being recognized:

1. Verify drivers are installed: `nvidia-smi`
2. Check if you have CPU-only PyTorch installed
3. Reinstall PyTorch with CUDA support (see GPU Installation above)
4. Run `check_gpu.py` for detailed diagnostics

### Out of Memory Errors

If training fails with OOM errors:

- Reduce batch size to 1
- Increase gradient accumulation steps
- Lower max sequence length
- Reduce LoRA rank
- Use a smaller base model

### Training is Very Slow

For CPU users:
- Training on CPU is expected to be extremely slow
- Consider using Google Colab (free GPU) or cloud services
- Use smaller models like TinyLlama for testing

For GPU users:
- Install Unsloth for 2-5x speedup
- Ensure CUDA-enabled PyTorch is installed
- Check GPU utilization with `nvidia-smi`

### Model Import Fails

If Ollama import fails:
- Verify Ollama is installed: `ollama --version`
- Check if the GGUF file was created in the output directory
- Try manual import: `cd output_directory && ollama create model-name -f Modelfile`

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space for models and outputs
- **Python**: 3.10 or higher

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 3080, etc.)
- **RAM**: 16GB or more
- **Storage**: SSD with 50GB+ free space
- **CUDA**: Version 11.8 or higher

### Model Size Guidelines

| Model Size | Minimum VRAM | Recommended VRAM | Training Time (est) |
|-----------|--------------|------------------|---------------------|
| 1B params | 6GB          | 8GB              | 1-2 hours           |
| 3B params | 8GB          | 12GB             | 2-4 hours           |
| 7B params | 12GB         | 16GB             | 4-8 hours           |
| 13B params| 16GB         | 24GB             | 8-16 hours          |

Times are estimates for 1 epoch on 1000 examples with typical settings.

## Advanced Usage

### Custom LoRA Targets

The application targets these modules by default:
- q_proj, k_proj, v_proj, o_proj
- gate_proj, up_proj, down_proj

For custom targeting, modify the `target_modules` list in the code.

### Manual GGUF Conversion

If you're using CPU training or Unsloth isn't available, you'll need to manually convert to GGUF:

1. Training saves a merged HuggingFace model
2. Install llama.cpp: `git clone https://github.com/ggerganov/llama.cpp`
3. Convert: `python llama.cpp/convert-hf-to-gguf.py merged_model --outtype f16`
4. Quantize: `llama.cpp/llama-quantize model-f16.gguf model-q5_k_m.gguf q5_k_m`
5. Import: `ollama create model-name -f Modelfile`

## Contributing

Contributions are welcome. If you encounter bugs or have feature requests, please open an issue on GitHub.

## License

This project is provided as-is for educational and research purposes. Please respect the licenses of any base models and datasets you use with this tool.

## Acknowledgments

Built with:
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [Transformers](https://github.com/huggingface/transformers) - Model architecture and training
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) - Transformer reinforcement learning
- [DearPyGUI](https://github.com/hoffstadt/DearPyGui) - GPU-accelerated GUI framework
- [Ollama](https://ollama.ai) - Local LLM runtime

## Links

- **Repository**: https://github.com/noosed/ntech-llm-tuner
- **Issues**: https://github.com/noosed/ntech-llm-tuner/issues
- **Ollama**: https://ollama.ai
- **Unsloth**: https://github.com/unslothai/unsloth

---

Created by github.com/noosed
