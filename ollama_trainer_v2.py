# -*- coding: utf-8 -*-
"""
NTech LLM Tuner - GUI for fine-tuning LLMs and importing to Ollama
Supports both GPU (Unsloth) and CPU (standard transformers) training
"""
import dearpygui.dearpygui as dpg
import subprocess
import os
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import traceback

# ─── DEPENDENCY CHECKS ────────────────────────────────────────────────

def get_ollama_models():
    """Get list of installed Ollama models"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header
                models = []
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        models.append(model_name)
                return models
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []

def get_popular_models():
    """Get list of popular models for fine-tuning"""
    models = {
        "Ollama Models (Installed)": get_ollama_models(),
        "Popular Ollama Models (Download Available)": [
            "llama3:8b",
            "llama3:70b",
            "mistral:7b",
            "mixtral:8x7b",
            "phi3:mini",
            "gemma:7b",
            "qwen2:7b",
        ],
        "Small Models (CPU-friendly)": [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "stabilityai/stablelm-2-1_6b",
            "Qwen/Qwen2-1.5B-Instruct",
        ],
        "Medium Models (GPU recommended)": [
            "unsloth/llama-3-8b-bnb-4bit",
            "unsloth/mistral-7b-v0.3-bnb-4bit",
            "unsloth/Phi-3-mini-4k-instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        "Large Models (Good GPU required)": [
            "unsloth/llama-3-70b-bnb-4bit",
            "unsloth/mixtral-8x7b-bnb-4bit",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
    }
    
    # Flatten into a single list with separators
    all_models = []
    for category, model_list in models.items():
        if model_list:  # Only add category if it has models
            all_models.append(f"--- {category} ---")
            all_models.extend(model_list)
    
    return all_models

try:
    import torch
    HAS_TORCH = True
    HAS_GPU = torch.cuda.is_available()
    
    # GPU diagnostics
    if HAS_GPU:
        GPU_COUNT = torch.cuda.device_count()
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    else:
        GPU_COUNT = 0
        GPU_NAME = "None"
        GPU_MEMORY = 0
except ImportError:
    HAS_TORCH = False
    HAS_GPU = False
    GPU_COUNT = 0
    GPU_NAME = "None"
    GPU_MEMORY = 0

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except (ImportError, NotImplementedError, RuntimeError) as e:
    HAS_UNSLOTH = False
    UNSLOTH_ERROR = str(e)

try:
    from trl import SFTTrainer
    from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

DEPS_AVAILABLE = HAS_TORCH and HAS_TRANSFORMERS


# ─── CONFIGURATION ────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Training configuration with validation"""
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small default for CPU
    dataset_path: str = ""
    system_prompt: str = "You are a helpful assistant."
    lora_rank: int = 32  # Lower default for CPU
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    epochs: int = 1  # Lower default for testing
    batch_size: int = 1  # Lower default for CPU
    grad_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 512  # Lower default for CPU
    output_name: str = "my-fine-tuned-model"
    output_dir: str = "./gguf_export"  # New: customizable output directory
    quant_method: str = "q5_k_m"
    save_steps: int = 100
    logging_steps: int = 10
    
    def validate(self) -> tuple[bool, str]:
        """Validate configuration"""
        if not self.base_model.strip():
            return False, "Base model name cannot be empty"
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            return False, f"Dataset path not found: {self.dataset_path}"
        if self.lora_rank < 1 or self.lora_rank > 256:
            return False, "LoRA rank must be between 1 and 256"
        if self.epochs < 1:
            return False, "Epochs must be at least 1"
        if self.batch_size < 1:
            return False, "Batch size must be at least 1"
        if not self.output_name.strip():
            return False, "Output name cannot be empty"
        if not self.output_dir.strip():
            return False, "Output directory cannot be empty"
        return True, ""


# ─── TRAINING LOGIC ───────────────────────────────────────────────────

class TrainingManager:
    """Handles training process in background thread"""
    
    def __init__(self, log_callback):
        self.log = log_callback
        self.is_training = False
        self.should_stop = False
        self.use_unsloth = HAS_UNSLOTH and HAS_GPU
        
    def log_message(self, msg: str, error: bool = False):
        """Thread-safe logging"""
        prefix = "ERROR: " if error else ""
        self.log(f"{prefix}{msg}\n")
    
    def load_model_unsloth(self, config: TrainingConfig):
        """Load model using Unsloth (GPU only)"""
        self.log_message(f"Loading model with Unsloth (GPU accelerated)")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer
    
    def load_model_standard(self, config: TrainingConfig):
        """Load model using standard transformers (CPU compatible)"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log_message(f"Loading model with transformers (device: {device})")
        
        if not torch.cuda.is_available():
            self.log_message("[!] WARNING: Training on CPU will be VERY slow!")
            self.log_message("[!] Consider using Google Colab (free GPU) or a cloud service")
        
        # Load model
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # CPU mode - load in FP32
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply LoRA
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        self.log_message(f"Trainable params: {trainable_params:,} / {all_params:,} "
                        f"({100 * trainable_params / all_params:.2f}%)")
        
        return model, tokenizer
    
    def export_to_gguf_unsloth(self, model, tokenizer, config: TrainingConfig, gguf_dir):
        """Export using Unsloth's built-in GGUF export"""
        self.log_message("Exporting to GGUF format (Unsloth method)...")
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=config.quant_method
        )
    
    def export_to_gguf_standard(self, model, tokenizer, config: TrainingConfig, gguf_dir):
        """Export using manual merge + llama.cpp conversion"""
        self.log_message("Saving merged model...")
        
        # Merge LoRA and save
        merged_dir = gguf_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Merge adapter with base model
        model = model.merge_and_unload()
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        
        self.log_message(f"[OK] Merged model saved to: {merged_dir}")
        self.log_message("")
        self.log_message("To convert to GGUF, you need llama.cpp:")
        self.log_message("─" * 60)
        self.log_message("# 1. Install llama.cpp")
        self.log_message("git clone https://github.com/ggerganov/llama.cpp")
        self.log_message("cd llama.cpp")
        self.log_message("make  # or: cmake -B build && cmake --build build --config Release")
        self.log_message("")
        self.log_message("# 2. Convert to GGUF")
        self.log_message(f"python llama.cpp/convert-hf-to-gguf.py {merged_dir} \\")
        self.log_message(f"  --outfile {gguf_dir}/model-f16.gguf --outtype f16")
        self.log_message("")
        self.log_message("# 3. Quantize")
        self.log_message(f"llama.cpp/llama-quantize {gguf_dir}/model-f16.gguf \\")
        self.log_message(f"  {gguf_dir}/{config.quant_method}.gguf {config.quant_method}")
        self.log_message("─" * 60)
        
        # Create instruction file
        instructions = f"""GGUF Conversion Instructions
============================

Your model has been trained and saved as a merged HuggingFace model.
To use it with Ollama, you need to convert it to GGUF format.

Step 1: Install llama.cpp
--------------------------
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# On Windows, use CMake instead:
cmake -B build
cmake --build build --config Release

Step 2: Convert to GGUF (FP16)
-------------------------------
python llama.cpp/convert-hf-to-gguf.py {merged_dir.absolute()} \\
  --outfile {gguf_dir.absolute()}/model-f16.gguf --outtype f16

Step 3: Quantize
----------------
llama.cpp/llama-quantize {gguf_dir.absolute()}/model-f16.gguf \\
  {gguf_dir.absolute()}/{config.quant_method}.gguf {config.quant_method}

Step 4: Import to Ollama
-------------------------
cd {gguf_dir.absolute()}
ollama create {config.output_name} -f Modelfile

Then test with:
ollama run {config.output_name}

Notes:
- If using Windows, paths may need backslashes instead
- Quantization options: q4_k_m (smallest), q5_k_m, q6_k, q8_0, f16 (largest)
- The Modelfile has already been created for you
"""
        (gguf_dir / "CONVERSION_INSTRUCTIONS.txt").write_text(instructions)
        self.log_message(f"[OK] Instructions saved to: {gguf_dir / 'CONVERSION_INSTRUCTIONS.txt'}")
        
    def train(self, config: TrainingConfig):
        """Main training function (runs in background thread)"""
        if not DEPS_AVAILABLE:
            self.log_message("[ERROR] Missing dependencies!", error=True)
            self.log_message("", error=True)
            self.log_message("Install required packages:", error=True)
            self.log_message("  pip install torch transformers datasets trl peft accelerate", error=True)
            self.log_message("", error=True)
            if HAS_GPU:
                self.log_message("For GPU support (faster):", error=True)
                self.log_message("  pip install bitsandbytes", error=True)
            return
            
        try:
            self.is_training = True
            self.log_message("=" * 60)
            self.log_message(">> TRAINING STARTED")
            self.log_message("=" * 60)
            self.log_message(f"Mode: {' Unsloth (GPU)' if self.use_unsloth else '  Standard Transformers'}")
            self.log_message(f"Device: {' CUDA GPU' if HAS_GPU else ' CPU (SLOW!)'}")
            
            if not HAS_UNSLOTH and HAS_GPU:
                self.log_message("")
                self.log_message("[i]  Unsloth not available (using standard transformers)")
                self.log_message("   Install for 2-5x faster training:")
                self.log_message("   pip install unsloth")
            
            self.log_message("=" * 60)
            self.log_message("")
            
            # 1. Load model
            if self.use_unsloth:
                model, tokenizer = self.load_model_unsloth(config)
            else:
                model, tokenizer = self.load_model_standard(config)
            
            if self.should_stop:
                self.log_message("[!] Training cancelled by user")
                return
            
            # 2. Load dataset
            self.log_message(f" Loading dataset: {config.dataset_path}")
            if os.path.isdir(config.dataset_path):
                dataset = load_dataset("json", data_dir=config.dataset_path, split="train")
            else:
                dataset = load_dataset("json", data_files=config.dataset_path, split="train")
            
            self.log_message(f"[OK] Dataset loaded: {len(dataset):,} examples")
            
            if self.should_stop:
                self.log_message("[!] Training cancelled by user")
                return
            
            # 3. Setup trainer
            self.log_message("⚙️  Configuring trainer...")
            output_dir = Path("./training_output") / config.output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate total steps
            total_steps = (len(dataset) * config.epochs) // (config.batch_size * config.grad_accumulation)
            self.log_message(f"   Total training steps: {total_steps:,}")
            
            training_args = TrainingArguments(
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.grad_accumulation,
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                warmup_steps=config.warmup_steps,
                fp16=HAS_GPU and not torch.cuda.is_bf16_supported(),
                bf16=HAS_GPU and torch.cuda.is_bf16_supported(),
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                output_dir=str(output_dir),
                optim="adamw_8bit" if HAS_GPU else "adamw_torch",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=42,
                report_to="none",
                save_total_limit=2,
            )
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=config.max_seq_length,
                args=training_args,
            )
            
            # 4. Train
            self.log_message("")
            self.log_message(">>  Starting training...")
            self.log_message("=" * 60)
            trainer.train()
            self.log_message("=" * 60)
            self.log_message("[OK] Training completed!")
            
            if self.should_stop:
                self.log_message("[!] Training cancelled by user")
                return
            
            # 5. Export
            self.log_message("")
            gguf_dir = Path(config.output_dir) / config.output_name
            gguf_dir.mkdir(parents=True, exist_ok=True)
            
            if self.use_unsloth:
                self.export_to_gguf_unsloth(model, tokenizer, config, gguf_dir)
            else:
                self.export_to_gguf_standard(model, tokenizer, config, gguf_dir)
            
            # 6. Create Modelfile
            self.log_message("")
            self.log_message(" Creating Ollama Modelfile...")
            modelfile_path = gguf_dir / "Modelfile"
            gguf_filename = f"{config.quant_method}.gguf"
            
            modelfile_content = f"""FROM ./{gguf_filename}
SYSTEM "{config.system_prompt}"
PARAMETER temperature 0.75
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
            modelfile_path.write_text(modelfile_content)
            self.log_message(f"[OK] Modelfile created: {modelfile_path}")
            
            # 7. Import to Ollama (only if GGUF exists)
            gguf_file = gguf_dir / gguf_filename
            if gguf_file.exists():
                self.log_message("")
                self.log_message(f">> Importing to Ollama as '{config.output_name}'...")
                result = subprocess.run(
                    ["ollama", "create", config.output_name, "-f", str(modelfile_path)],
                    cwd=str(gguf_dir),
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.log_message("")
                    self.log_message("=" * 60)
                    self.log_message(f"*** SUCCESS! Model '{config.output_name}' is ready!")
                    self.log_message("")
                    self.log_message(f"Test it with:")
                    self.log_message(f"  ollama run {config.output_name}")
                    self.log_message("=" * 60)
                else:
                    self.log_message("")
                    self.log_message(f"[ERROR] Ollama import failed", error=True)
                    self.log_message(f"   {result.stderr}", error=True)
                    self.log_message("")
                    self.log_message(f"Manual import:")
                    self.log_message(f"  cd {gguf_dir}")
                    self.log_message(f"  ollama create {config.output_name} -f Modelfile")
            else:
                self.log_message("")
                self.log_message("=" * 60)
                self.log_message("[OK] Training complete!")
                self.log_message("")
                self.log_message("[!]  Manual GGUF conversion required")
                self.log_message(f"   See: {gguf_dir / 'CONVERSION_INSTRUCTIONS.txt'}")
                self.log_message("=" * 60)
                
        except Exception as e:
            self.log_message("", error=True)
            self.log_message("=" * 60, error=True)
            self.log_message(f"[ERROR] TRAINING FAILED", error=True)
            self.log_message("=" * 60, error=True)
            self.log_message(f"{type(e).__name__}: {str(e)}", error=True)
            self.log_message("", error=True)
            self.log_message("Full traceback:", error=True)
            self.log_message(traceback.format_exc(), error=True)
        finally:
            self.is_training = False
            self.should_stop = False
    
    def start_training(self, config: TrainingConfig):
        """Start training in background thread"""
        if self.is_training:
            self.log_message("[!] Training already in progress", error=True)
            return
            
        valid, error_msg = config.validate()
        if not valid:
            self.log_message(f"[ERROR] Invalid configuration: {error_msg}", error=True)
            return
        
        self.should_stop = False
        thread = threading.Thread(target=self.train, args=(config,), daemon=True)
        thread.start()
    
    def stop_training(self):
        """Request training stop"""
        if self.is_training:
            self.log_message("  Stopping training (may take a moment)...")
            self.should_stop = True


# ─── GUI APPLICATION ──────────────────────────────────────────────────

class OllamaTrainerGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.config = TrainingConfig()
        self.trainer: Optional[TrainingManager] = None
        self.available_models = []
        
    def drag_drop_callback(self, sender, app_data):
        """Handle file drag and drop from file system onto viewport"""
        # When files are dropped on viewport, app_data is a list of file paths
        try:
            if not app_data:
                return
                
            # Handle both list and single string formats
            files = app_data if isinstance(app_data, list) else [app_data]
            
            # Process the first file
            if len(files) > 0:
                dropped_file = files[0]
                
                # Normalize path (handle both forward and backslashes)
                dropped_file = str(dropped_file).replace('\\', '/')
                
                # Check if it's a JSON/JSONL file
                if dropped_file.lower().endswith(('.json', '.jsonl')):
                    # Convert back to Windows path format
                    dropped_file = dropped_file.replace('/', '\\')
                    dpg.set_value("dataset_path", dropped_file)
                    self.append_log(f" Dataset file dropped: {dropped_file}\n")
                else:
                    filename = dropped_file.split('/')[-1]
                    self.append_log(f"[!]️  Invalid file type: {filename}\n")
                    self.append_log(f"   Please drop a .json or .jsonl file\n")
                    
                # Warn if multiple files
                if len(files) > 1:
                    self.append_log(f"[i]  Multiple files dropped, using first file only\n")
        except Exception as e:
            self.append_log(f"[ERROR] Error processing dropped file: {e}\n")
    
    def model_selected_callback(self, sender, app_data):
        """Handle model selection from dropdown"""
        if not app_data.startswith("---"):
            dpg.set_value("base_model", app_data)
            self.append_log(f">> Model selected: {app_data}\n")
    
    def download_model(self):
        """Download the selected model using Ollama"""
        model_name = dpg.get_value("base_model_combo")
        
        # Skip if it's a separator or already downloaded
        if model_name.startswith("---"):
            return
        
        # Check if it's an Ollama model format (contains :)
        if ":" in model_name and "/" not in model_name:
            self.append_log(f">> Downloading model: {model_name}\n")
            self.append_log("This may take several minutes...\n")
            
            def download_thread():
                try:
                    result = subprocess.run(
                        ["ollama", "pull", model_name],
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30 minute timeout
                    )
                    
                    if result.returncode == 0:
                        self.append_log(f"[OK] Model {model_name} downloaded successfully!\n")
                        # Refresh model list
                        self.refresh_models()
                    else:
                        self.append_log(f"[ERROR] Failed to download: {result.stderr}\n")
                except subprocess.TimeoutExpired:
                    self.append_log(f"[ERROR] Download timed out\n")
                except Exception as e:
                    self.append_log(f"[ERROR] Download error: {e}\n")
            
            # Run in background
            thread = threading.Thread(target=download_thread, daemon=True)
            thread.start()
        else:
            self.append_log(f"[i] '{model_name}' is a HuggingFace model - it will be downloaded automatically during training\n")
    
    def refresh_models(self):
        """Refresh the list of available models"""
        self.available_models = get_popular_models()
        if dpg.does_item_exist("base_model_combo"):
            dpg.configure_item("base_model_combo", items=self.available_models)
        self.append_log(" Model list refreshed\n")
        
    def append_log(self, text: str):
        """Append text to log widget"""
        current = dpg.get_value("log")
        dpg.set_value("log", current + text)
        # Auto-scroll to bottom of child window
        try:
            dpg.set_y_scroll("log_window", -1.0)
        except:
            pass  # Ignore if scrolling not available yet
    
    def select_dataset_callback(self, sender, app_data):
        """File dialog callback"""
        selections = app_data.get("selections", {})
        if selections:
            path = list(selections.values())[0]
            dpg.set_value("dataset_path", path)
            self.append_log(f" Selected dataset: {path}\n")
    
    def show_file_dialog(self):
        """Show file picker"""
        dpg.show_item("file_dialog")
    
    def show_output_dir_dialog(self):
        """Show output directory picker"""
        dpg.show_item("output_dir_dialog")
    
    def select_output_dir_callback(self, sender, app_data):
        """Output directory dialog callback"""
        selections = app_data.get("selections", {})
        if selections:
            path = list(selections.values())[0]
            dpg.set_value("output_dir", path)
            self.append_log(f">> Output directory set: {path}\n")
    
    def read_config_from_gui(self) -> TrainingConfig:
        """Read all values from GUI widgets"""
        return TrainingConfig(
            base_model=dpg.get_value("base_model"),
            dataset_path=dpg.get_value("dataset_path"),
            system_prompt=dpg.get_value("system_prompt"),
            lora_rank=dpg.get_value("lora_rank"),
            lora_alpha=dpg.get_value("lora_alpha"),
            lora_dropout=dpg.get_value("lora_dropout"),
            epochs=dpg.get_value("epochs"),
            batch_size=dpg.get_value("batch_size"),
            grad_accumulation=dpg.get_value("grad_accumulation"),
            learning_rate=dpg.get_value("learning_rate"),
            warmup_steps=dpg.get_value("warmup_steps"),
            max_seq_length=dpg.get_value("max_seq_length"),
            output_name=dpg.get_value("output_name"),
            output_dir=dpg.get_value("output_dir"),
            quant_method=dpg.get_value("quant_method"),
            save_steps=dpg.get_value("save_steps"),
            logging_steps=dpg.get_value("logging_steps"),
        )
    
    def start_training_callback(self):
        """Start training button callback"""
        config = self.read_config_from_gui()
        self.trainer.start_training(config)
        dpg.configure_item("btn_start", enabled=False)
        dpg.configure_item("btn_stop", enabled=True)
    
    def stop_training_callback(self):
        """Stop training button callback"""
        self.trainer.stop_training()
        dpg.configure_item("btn_start", enabled=True)
        dpg.configure_item("btn_stop", enabled=False)
    
    def clear_log_callback(self):
        """Clear log output"""
        dpg.set_value("log", "")
    
    def save_config_callback(self):
        """Save current config to JSON - show dialog"""
        dpg.show_item("save_config_dialog")
    
    def save_config_file(self, sender, app_data):
        """Actually save the config file"""
        selections = app_data.get("selections", {})
        if selections:
            filepath = list(selections.values())[0]
            
            # Ensure .json extension
            if not filepath.lower().endswith('.json'):
                filepath += '.json'
            
            try:
                config = self.read_config_from_gui()
                with open(filepath, "w") as f:
                    json.dump(asdict(config), f, indent=2)
                self.append_log(f"[OK] Configuration saved to {filepath}\n")
            except Exception as e:
                self.append_log(f"[ERROR] Failed to save config: {e}\n")
    
    def load_config_callback(self):
        """Load config from JSON"""
        dpg.show_item("load_config_dialog")
    
    def load_config_file(self, sender, app_data):
        """Actually load the config file"""
        selections = app_data.get("selections", {})
        if selections:
            path = list(selections.values())[0]
            try:
                with open(path) as f:
                    data = json.load(f)
                config = TrainingConfig(**data)
                
                # Update all widgets
                for key, value in asdict(config).items():
                    if dpg.does_item_exist(key):
                        dpg.set_value(key, value)
                
                self.append_log(f" Loaded configuration from {path}\n")
            except Exception as e:
                self.append_log(f"[ERROR] Failed to load config: {e}\n")
    
    def create_gui(self):
        """Build the GUI layout"""
        dpg.create_context()
        
        # File dialogs
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.select_dataset_callback,
            tag="file_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".json", color=(150, 255, 150, 255))
            dpg.add_file_extension(".jsonl", color=(150, 255, 150, 255))
        
        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self.select_output_dir_callback,
            tag="output_dir_dialog",
            width=700,
            height=400
        ):
            pass  # Directory selector doesn't need file extensions
        
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.save_config_file,
            tag="save_config_dialog",
            width=700,
            height=400,
            default_filename="config.json"
        ):
            dpg.add_file_extension(".json", color=(150, 255, 150, 255))
        
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.load_config_file,
            tag="load_config_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".json", color=(150, 255, 150, 255))
        
        # Main window
        with dpg.window(tag="main", label="NTech LLM Tuner"):
            dpg.add_text("Fine-tune LLMs and import directly to Ollama", color=(100, 200, 255))
            
            # System status
            if HAS_GPU:
                status_text = f"System: GPU Detected - {GPU_NAME} ({GPU_MEMORY:.1f}GB) | "
            else:
                status_text = "System: CPU Only (No GPU detected) | "
            
            if HAS_UNSLOTH:
                status_text += "Unsloth [OK]"
            elif HAS_GPU:
                status_text += "Unsloth [X] (install for 2x speed)"
            else:
                status_text += "Unsloth [X] (requires GPU)"
            
            dpg.add_text(status_text, color=(150, 150, 150))
            dpg.add_separator()
            
            # Model & Dataset
            with dpg.collapsing_header(label="Model & Dataset", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        label="Base Model",
                        items=self.available_models,
                        default_value=self.config.base_model,
                        tag="base_model_combo",
                        width=320,
                        callback=self.model_selected_callback
                    )
                    dpg.add_button(
                        label="Refresh Models",
                        callback=lambda: self.refresh_models(),
                        width=110,
                        height=23
                    )
                    dpg.add_button(
                        label="Download Model",
                        callback=lambda: self.download_model(),
                        width=110,
                        height=23
                    )
                
                dpg.add_input_text(
                    label="Custom Model (optional)",
                    default_value=self.config.base_model,
                    tag="base_model",
                    width=450,
                    hint="Type HuggingFace model name or select from dropdown above"
                )
                
                dpg.add_text("[i] Tip: Select from dropdown or type custom model name", 
                            color=(150, 150, 150))
                
                dpg.add_spacer(height=10)
                
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        label="Dataset Path",
                        default_value=self.config.dataset_path,
                        tag="dataset_path",
                        width=400
                    )
                    dpg.add_button(label="Browse...", callback=self.show_file_dialog)
                
                # Show tip based on drag-drop availability
                if hasattr(self, 'has_drag_drop') and self.has_drag_drop:
                    dpg.add_text("[i] Tip: Drag & drop .json/.jsonl file or click Browse", 
                                color=(150, 150, 150))
                else:
                    dpg.add_text("[i] Tip: Click Browse to select your dataset file", 
                                color=(150, 150, 150))
                
                dpg.add_input_text(
                    label="System Prompt",
                    default_value=self.config.system_prompt,
                    tag="system_prompt",
                    multiline=True,
                    height=80,
                    width=600
                )
            
            # LoRA Settings
            with dpg.collapsing_header(label="LoRA Configuration", default_open=True):
                dpg.add_slider_int(
                    label="LoRA Rank",
                    default_value=self.config.lora_rank,
                    min_value=8,
                    max_value=256,
                    tag="lora_rank",
                    width=300
                )
                dpg.add_slider_int(
                    label="LoRA Alpha",
                    default_value=self.config.lora_alpha,
                    min_value=8,
                    max_value=512,
                    tag="lora_alpha",
                    width=300
                )
                dpg.add_slider_float(
                    label="LoRA Dropout",
                    default_value=self.config.lora_dropout,
                    min_value=0.0,
                    max_value=0.5,
                    tag="lora_dropout",
                    width=300
                )
            
            # Training Settings
            with dpg.collapsing_header(label="Training Parameters", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(
                        label="Epochs",
                        default_value=self.config.epochs,
                        min_value=1,
                        max_value=10,
                        tag="epochs",
                        width=150
                    )
                    dpg.add_slider_int(
                        label="Batch Size",
                        default_value=self.config.batch_size,
                        min_value=1,
                        max_value=16,
                        tag="batch_size",
                        width=150
                    )
                
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(
                        label="Gradient Accumulation",
                        default_value=self.config.grad_accumulation,
                        min_value=1,
                        max_value=32,
                        tag="grad_accumulation",
                        width=150
                    )
                    dpg.add_input_float(
                        label="Learning Rate",
                        default_value=self.config.learning_rate,
                        tag="learning_rate",
                        width=150,
                        format="%.2e"
                    )
                
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(
                        label="Warmup Steps",
                        default_value=self.config.warmup_steps,
                        min_value=0,
                        max_value=500,
                        tag="warmup_steps",
                        width=150
                    )
                    dpg.add_slider_int(
                        label="Max Sequence Length",
                        default_value=self.config.max_seq_length,
                        min_value=128,
                        max_value=8192,
                        tag="max_seq_length",
                        width=150
                    )
                
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(
                        label="Save Steps",
                        default_value=self.config.save_steps,
                        min_value=10,
                        max_value=1000,
                        tag="save_steps",
                        width=150
                    )
                    dpg.add_slider_int(
                        label="Logging Steps",
                        default_value=self.config.logging_steps,
                        min_value=1,
                        max_value=100,
                        tag="logging_steps",
                        width=150
                    )
            
            # Output Settings
            with dpg.collapsing_header(label="Output Configuration", default_open=True):
                dpg.add_input_text(
                    label="Output Model Name",
                    default_value=self.config.output_name,
                    tag="output_name",
                    width=400
                )
                
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        label="Output Directory",
                        default_value=self.config.output_dir,
                        tag="output_dir",
                        width=400
                    )
                    dpg.add_button(label="Browse...", callback=self.show_output_dir_dialog)
                
                dpg.add_combo(
                    label="Quantization Method",
                    items=["q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"],
                    default_value=self.config.quant_method,
                    tag="quant_method",
                    width=200
                )
            
            dpg.add_separator()
            
            # Control buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start Training",
                    callback=self.start_training_callback,
                    tag="btn_start",
                    width=150,
                    height=40
                )
                dpg.add_button(
                    label="Stop Training",
                    callback=self.stop_training_callback,
                    tag="btn_stop",
                    width=150,
                    height=40,
                    enabled=False
                )
                dpg.add_button(
                    label="Clear Log",
                    callback=self.clear_log_callback,
                    width=100,
                    height=40
                )
                dpg.add_button(
                    label="Save Config",
                    callback=self.save_config_callback,
                    width=120,
                    height=40
                )
                dpg.add_button(
                    label="Load Config",
                    callback=self.load_config_callback,
                    width=120,
                    height=40
                )
            
            dpg.add_separator()
            
            # Log output
            dpg.add_text("Training Log:")
            with dpg.child_window(tag="log_window", height=250, border=True):
                dpg.add_text("", tag="log")
            
            dpg.add_separator()
            
            # GitHub watermark
            dpg.add_text("github.com/noosed", color=(100, 100, 100))
        
        # Setup and run
        dpg.create_viewport(title='NTech LLM Tuner', width=1000, height=900)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
    
    def setup_drag_drop(self):
        """Set up file drop callback after viewport is ready"""
        try:
            dpg.set_viewport_drop_callback(self.drag_drop_callback)
            return True
        except (AttributeError, SystemError):
            return False
    
    def run(self):
        """Run the application"""
        self.trainer = TrainingManager(self.append_log)
        
        # Load available models before creating GUI
        self.available_models = get_popular_models()
        
        # Check if drag-and-drop is available
        self.has_drag_drop = hasattr(dpg, 'set_viewport_drop_callback')
        
        self.create_gui()
        
        # Try to enable drag-and-drop
        drag_drop_enabled = self.setup_drag_drop()
        
        # Startup messages
        self.append_log("=" * 60 + "\n")
        self.append_log("NTech LLM Tuner - Fine-tuning GUI\n")
        self.append_log("=" * 60 + "\n")
        
        # Show detected Ollama models
        ollama_models = get_ollama_models()
        if ollama_models:
            self.append_log(f"[OK] Detected {len(ollama_models)} Ollama model(s):\n")
            for model in ollama_models[:5]:  # Show first 5
                self.append_log(f"  • {model}\n")
            if len(ollama_models) > 5:
                self.append_log(f"  ... and {len(ollama_models) - 5} more\n")
        else:
            self.append_log("[i]  No Ollama models detected (Ollama may not be installed)\n")
        
        self.append_log("\n")
        
        if not DEPS_AVAILABLE:
            self.append_log("[ERROR] Missing dependencies!\n")
            self.append_log("\nInstall required packages:\n")
            self.append_log("  pip install torch transformers datasets trl peft accelerate\n")
            if HAS_GPU:
                self.append_log("  pip install bitsandbytes\n")
        else:
            self.append_log(f"[OK] Dependencies loaded\n")
            
            # Detailed GPU diagnostics
            self.append_log("\n--- GPU Diagnostics ---\n")
            self.append_log(f"PyTorch version: {torch.__version__}\n")
            self.append_log(f"CUDA available: {HAS_GPU}\n")
            
            if HAS_GPU:
                self.append_log(f"CUDA version: {torch.version.cuda}\n")
                self.append_log(f"GPU count: {GPU_COUNT}\n")
                self.append_log(f"GPU name: {GPU_NAME}\n")
                self.append_log(f"GPU memory: {GPU_MEMORY:.2f} GB\n")
                self.append_log(f"cuDNN available: {torch.backends.cudnn.is_available()}\n")
                self.append_log(f"cuDNN version: {torch.backends.cudnn.version()}\n")
            else:
                self.append_log("\n[!] NO GPU DETECTED!\n")
                self.append_log("\nTroubleshooting steps:\n")
                self.append_log("1. Check if NVIDIA drivers are installed:\n")
                self.append_log("   Open Command Prompt and run: nvidia-smi\n")
                self.append_log("   (Should show your RTX 3080)\n\n")
                self.append_log("2. You may have CPU-only PyTorch installed!\n")
                self.append_log("   Uninstall and reinstall with CUDA support:\n")
                self.append_log("   pip uninstall torch torchvision torchaudio\n")
                self.append_log("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n\n")
                self.append_log("3. Restart this application after reinstalling\n")
            
            self.append_log("--- End Diagnostics ---\n\n")
            
            if HAS_UNSLOTH:
                self.append_log(f"[OK] Unsloth available (2-5x faster training)\n")
            elif HAS_GPU:
                self.append_log(f"[i]  Unsloth not installed (install for faster training)\n")
                self.append_log(f"   pip install unsloth\n")
            
            if drag_drop_enabled:
                self.append_log(f"[OK] Drag & drop enabled\n")
            else:
                self.append_log(f"[!]  Drag & drop not available (use Browse button)\n")
            
            if not HAS_GPU:
                self.append_log(f"\n[!]  WARNING: No GPU detected!\n")
                self.append_log(f"   Training will be extremely slow on CPU.\n")
                self.append_log(f"   Consider using Google Colab or a cloud GPU.\n")
        
        self.append_log("\n")
        self.append_log("Ready to train! Configure settings and click 'Start Training'.\n")
        self.append_log("=" * 60 + "\n")
        
        dpg.start_dearpygui()
        dpg.destroy_context()


# ─── MAIN ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = OllamaTrainerGUI()
    app.run()
