# Changelog

All notable changes to NTech LLM Tuner will be documented in this file.

## [1.0.0] - 2025-02-01

### Initial Release

#### Features
- GUI application for LLM fine-tuning with DearPyGUI
- LoRA training support with configurable parameters
- Automatic GPU detection and CUDA support
- CPU fallback for systems without GPU
- Ollama integration for model deployment
- HuggingFace model hub integration
- Drag and drop support for dataset files
- Configuration save/load functionality
- Real-time training logs and diagnostics
- Background training with non-blocking UI
- Model download support for popular Ollama models
- Multiple GGUF quantization options
- Custom output directory selection
- Comprehensive GPU diagnostics tool

#### Supported Models
- TinyLlama family
- Llama 3 family
- Mistral family
- Phi family
- Gemma family
- Qwen family
- Any HuggingFace transformer model

#### Supported Features
- LoRA rank: 8-256
- Batch sizes: 1-16
- Gradient accumulation
- Mixed precision training (FP16/BF16)
- Multiple quantization levels (q4_k_m to f16)
- Custom system prompts
- Adjustable learning rates and warmup

#### Known Limitations
- Unsloth requires CUDA GPU
- Large models require significant VRAM
- CPU training is very slow
- Manual GGUF conversion needed without Unsloth on CPU

#### Dependencies
- Python 3.10+
- PyTorch 2.1+
- Transformers 4.36+
- DearPyGUI 1.10+
- See requirements.txt for full list

---

## Future Releases

### Planned Features
- Multi-GPU training support
- Resume training from checkpoints
- Advanced hyperparameter tuning
- Dataset validation and preprocessing
- Model evaluation metrics
- Fine-tuning templates for common tasks
- Web-based interface option
- Distributed training support
