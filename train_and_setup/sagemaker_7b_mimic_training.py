# -*- coding: utf-8 -*-
# ===========================
# Enhanced SageMaker LLaMA 3.1 8B Fine-Tuning Script
# Single GPU T4 Support + MIMIC-IV-ED Dataset + Unsloth Optimization ONLY
# ===========================
# 
# MANUAL INSTALLATION REQUIRED:
# pip install unsloth[colab-new]
# 
# Import unsloth FIRST to ensure all optimizations are applied
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Now import other libraries
import os
import time
import json
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from huggingface_hub import login, create_repo
import gc
import psutil
from tqdm import tqdm

# === Enhanced Configuration for LLaMA 3.1 8B + MIMIC-IV-ED ===
CONFIG = {
    'BASE_MODEL_NAME': "meta-llama/Llama-3.1-8B-Instruct",  # ✅ HuggingFace format LLaMA 3.1 8B Instruct Model
    'HF_MODEL_NAME': "ali009eng/llama-8b-mimic-ed-triage",  # ✅ New MIMIC repo for 8B
    'HF_MODEL_ID': "ali009eng/llama-8b-mimic-ed-triage",  # ✅ New MIMIC repo for 8B
    'HF_TOKEN': "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",
    'MAX_SEQ_LENGTH': 2048,  # Increased sequence length for A10G GPU
    'LOAD_IN_4BIT': True,
    'LOAD_IN_8BIT': False,  # Use 4-bit quantization
    'CHUNK_SIZE': 1500,  # Optimized chunks for A10G instance
    'BATCH_SIZE': 2,      # Increased for A10G GPU
    'LEARNING_RATE': 5e-5,  # Standard learning rate for A10G
    'CHECKPOINT_DIR': "./llama_8b_mimic_checkpoints",  # ✅ New checkpoint dir for 8B
    'DATASET_PATH': "./mimic_ed_data.csv",  # ✅ MIMIC data path
    'OUTPUT_DIR': "./llama_8b_mimic_checkpoints/outputs",  # ✅ New output dir for 8B
    'SAVE_STEPS': 150,    # Save more frequently for MIMIC
    'LOGGING_STEPS': 5,   # Standard logging frequency for A10G
    'GRADIENT_ACCUMULATION_STEPS': 16,  # Reduced for A10G GPU
    'WARMUP_STEPS': 200,  # Standard warmup for A10G
    'NUM_EPOCHS_PER_CHUNK': 3,  # Standard epochs for A10G
    'WEIGHT_DECAY': 0.01,  # Standard regularization for A10G
    'LR_SCHEDULER': "cosine",  # Optimized scheduler for 8B
    'WARMUP_RATIO': 0.1,
    'EVAL_STEPS': 75,     # Standard evaluation frequency for A10G
    'SAVE_TOTAL_LIMIT': 5,  # Keep more checkpoints
    'VAL_RATIO': 0.15,    # Increased validation for MIMIC
    'EARLY_STOPPING_PATIENCE': 3,  # More patience for MIMIC
    'EARLY_STOPPING_MIN_DELTA': 0.001,  # Lower threshold
    'GRADIENT_CHECKPOINTING': True,  # ✅ Enable for 8B
    'CPU_OFFLOAD': False,  # Disable CPU offloading for A10G (24GB VRAM)
    'MULTI_GPU': False,   # ✅ Single GPU for ml.g5.xlarge (1x A10G)
    'DDP_BACKEND': None,  # ✅ No DDP for single GPU
    # === NEW: Data Sampling Configuration ===
    'USE_DATA_SAMPLING': True,  # Enable data sampling to avoid training on all 400k rows
    'MAX_TRAINING_ROWS': 50000,  # Maximum rows to use for training (50K instead of 400K)
    'SAMPLING_STRATEGY': "stratified",  # Options: "stratified", "random", "first_n", "last_n"
    'STRATIFY_BY': "acuity",  # Column to stratify by for balanced sampling
    'RANDOM_SEED': 42,  # For reproducible sampling
    
    # === Resume Configuration ===
    'RESUME_FROM_CHECKPOINT': True,  # Enable resuming from last checkpoint
    'LAST_INDEX_FILE': "llama_8b_mimic_checkpoints/last_index.txt",  # File to track last processed row
}





# === Enhanced Callback Classes ===
class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback to prevent overfitting"""
    
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss", float('inf'))
        
        if eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.wait = 0
            print(f"✅ New best validation loss: {eval_loss:.4f}")
        else:
            self.wait += 1
            print(f"⏳ Early stopping counter: {self.wait}/{self.patience}")
            
        if self.wait >= self.patience:
            control.should_training_stop = True
            self.stopped_epoch = state.epoch
            print(f"🛑 Early stopping triggered at epoch {state.epoch}")

class SingleGPUMonitoringCallback(TrainerCallback):
    """Single GPU monitoring callback for 8B model"""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 15 == 0:  # Log every 15 steps
            memory_usage = get_memory_usage()
            gpu_memory = get_gpu_memory_usage()
            gpu_count = torch.cuda.device_count()
            
            print(f"📊 Step {state.global_step}: CPU Memory: {memory_usage:.1f} MB")
            for i in range(gpu_count):
                gpu_mem = torch.cuda.memory_allocated(i) / 1e9
                gpu_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"🖥️ GPU {i}: {gpu_mem:.1f}/{gpu_total:.1f} GB ({gpu_mem/gpu_total*100:.1f}%)")

def setup_environment():
    """Setup SageMaker environment and directories for single GPU 8B training"""
    # Set PyTorch memory allocation to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Create local directories
    os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # Setup Hugging Face
    try:
        login(CONFIG['HF_TOKEN'])
        create_repo(CONFIG['HF_MODEL_ID'], exist_ok=True, repo_type="model")
        print("✅ HuggingFace setup completed for MIMIC-ED 8B model")
    except Exception as e:
        print(f"❌ HuggingFace setup failed: {e}")
        raise
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_gpu_memory = sum([torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)]) / 1e9
        
        print(f"✅ Single GPU detected: {gpu_count} device, {total_gpu_memory:.1f}GB total memory")
        
        # Set memory fraction for 2xlarge instance (more GPU memory)
        torch.cuda.set_per_process_memory_fraction(0.95, 0)  # 95% for 2xlarge instance
        print(f"✅ GPU memory fraction set to 95% for 2xlarge instance")
        
        # Display detailed GPU info
        props = torch.cuda.get_device_properties(0)
        print(f"✅ GPU: {props.name}, Memory: {props.total_memory / 1e9:.1f}GB, Compute: {props.major}.{props.minor}")
        
        # Check if we have enough memory for 8B on 2xlarge instance
        if total_gpu_memory < 20:  # Need at least 20GB for 8B on A10G
            print(f"⚠️ GPU memory might be insufficient for 8B model")
            print(f"💡 Consider using CPU offloading or smaller model")
        else:
            print(f"✅ 2xlarge instance memory sufficient for 8B model")
            
    else:
        print(f"⚠️ No GPU detected! Training will be very slow on CPU")

def get_resume_index():
    """Get the last processed row index"""
    last_index_file = Path(CONFIG['CHECKPOINT_DIR']) / "last_index.txt"
    if last_index_file.exists():
        with open(last_index_file, "r") as f:
            return int(f.read().strip())
    return 0

def save_resume_index(index):
    """Save the current processed row index"""
    last_index_file = Path(CONFIG['CHECKPOINT_DIR']) / "last_index.txt"
    with open(last_index_file, "w") as f:
        f.write(str(index))

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def get_gpu_memory_usage():
    """Get current GPU memory usage across all GPUs"""
    if torch.cuda.is_available():
        total_memory = 0
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.memory_allocated(i)
        return total_memory / 1e9  # GB
    return 0

def clear_memory():
    """Clear memory and garbage collect"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

def sample_mimic_data(df, max_rows, strategy="stratified", stratify_by="acuity", random_seed=42):
    """
    Intelligently sample MIMIC data to create a manageable training dataset
    
    Args:
        df: Full MIMIC dataframe
        max_rows: Maximum number of rows to sample
        strategy: Sampling strategy ("stratified", "random", "first_n", "last_n")
        stratify_by: Column to stratify by for balanced sampling
        random_seed: Random seed for reproducibility
    
    Returns:
        Sampled dataframe with max_rows rows
    """
    print(f"📊 Sampling MIMIC data: {len(df)} total rows -> {max_rows} training rows")
    print(f"🎯 Strategy: {strategy}, Stratify by: {stratify_by}")
    
    if len(df) <= max_rows:
        print("✅ Dataset already within size limit, no sampling needed")
        return df
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    if strategy == "stratified":
        # Stratified sampling to maintain class balance
        if stratify_by in df.columns:
            # Get unique values and their counts
            unique_values = df[stratify_by].value_counts()
            print(f"📈 Distribution before sampling: {dict(unique_values)}")
            
            # Calculate how many samples to take from each class
            sampled_df = pd.DataFrame()
            for value, count in unique_values.items():
                if count > 0:
                    # Proportional sampling from each class
                    sample_size = min(count, int(max_rows * count / len(df)))
                    if sample_size > 0:
                        class_samples = df[df[stratify_by] == value].sample(
                            n=sample_size, random_state=random_seed
                        )
                        sampled_df = pd.concat([sampled_df, class_samples], ignore_index=True)
            
            # If we didn't get enough samples, fill with random samples
            if len(sampled_df) < max_rows:
                remaining = max_rows - len(sampled_df)
                remaining_df = df[~df.index.isin(sampled_df.index)].sample(
                    n=min(remaining, len(df) - len(sampled_df)), random_state=random_seed
                )
                sampled_df = pd.concat([sampled_df, remaining_df], ignore_index=True)
            
            print(f"📊 Distribution after stratified sampling: {dict(sampled_df[stratify_by].value_counts())}")
            
        else:
            print(f"⚠️ Stratify column '{stratify_by}' not found, falling back to random sampling")
            sampled_df = df.sample(n=max_rows, random_state=random_seed)
    
    elif strategy == "random":
        # Random sampling
        sampled_df = df.sample(n=max_rows, random_state=random_seed)
        print("🎲 Random sampling completed")
    
    elif strategy == "first_n":
        # Take first N rows
        sampled_df = df.head(max_rows)
        print("📋 First N sampling completed")
    
    elif strategy == "last_n":
        # Take last N rows
        sampled_df = df.tail(max_rows)
        print("📋 Last N sampling completed")
    
    else:
        print(f"⚠️ Unknown strategy '{strategy}', falling back to random sampling")
        sampled_df = df.sample(n=max_rows, random_state=random_seed)
    
    # Ensure we don't exceed max_rows
    if len(sampled_df) > max_rows:
        sampled_df = sampled_df.head(max_rows)
    
    print(f"✅ Sampling complete: {len(sampled_df)} rows selected for training")
    print(f"💾 Memory saved: {len(df) - len(sampled_df)} rows excluded")
    
    return sampled_df

def load_model_with_unsloth():
    """Load 8B model with Unsloth optimizations for single GPU"""
    try:
        print(f"🚀 Loading 8B base model with Unsloth: {CONFIG['BASE_MODEL_NAME']}")
        
        # Load base model with Unsloth optimizations (no custom device map for 4-bit)
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG['BASE_MODEL_NAME'],
            max_seq_length=CONFIG['MAX_SEQ_LENGTH'],
            dtype=None,
            load_in_4bit=CONFIG['LOAD_IN_4BIT'],
            load_in_8bit=CONFIG['LOAD_IN_8BIT'],
            device_map=None,  # Let Unsloth handle device mapping automatically
            low_cpu_mem_usage=True
        )
        
        # Load from the specific HuggingFace commit (037bea2)
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                base_model, 
                CONFIG['HF_MODEL_NAME'],
                token=CONFIG['HF_TOKEN'],
                revision="037bea2177864ac3c2521776f77a0a17a3cd826b",  # Specific commit
                device_map="auto" if CONFIG['CPU_OFFLOAD'] else None
            )
            print("✅ Successfully loaded LoRA adapter from HuggingFace commit 037bea2 (Chunk 8)")
        except Exception as e:
            print(f"💡 Could not load LoRA adapter: {e}")
            print("🆕 Will create new enhanced LoRA adapters for MIMIC-ED 8B")
            model = base_model
            
            # Apply LoRA to the base model
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None
            )
            print("✅ New LoRA adapters created for MIMIC-ED 8B")
            
        # Set model to training mode and enable gradients
        model.train()
        model.enable_input_require_grads()
        print("✅ Model set to training mode (LoRA parameters are trainable)")
            
        return model, tokenizer
            
    except Exception as e:
        print(f"❌ Error loading 8B model with Unsloth: {e}")
        raise

def validate_and_augment_mimic_data(df):
    """Validate and augment MIMIC-IV-ED training data for 8B model"""
    print("🔍 Validating and augmenting MIMIC-IV-ED data for 8B model...")
    
    # Data validation for MIMIC
    initial_count = len(df)
    valid_data = df.dropna(subset=['acuity', 'chiefcomplaint'])  # Keep rows with essential fields
    print(f"✅ MIMIC data validation: {len(valid_data)}/{initial_count} rows valid")
    
    # Data augmentation - create variations for better generalization
    augmented_data = []
    
    for _, row in tqdm(valid_data.iterrows(), desc="Augmenting MIMIC data", total=len(valid_data)):
        # Original case
        augmented_data.append(row)
        
        # Create slight variations for better generalization
        if pd.notna(row.get('pain')) and np.random.random() < 0.3:
            # Variation with slightly different pain level
            new_row = row.copy()
            if isinstance(row['pain'], (int, float)):
                new_row['pain'] = max(0, min(10, row['pain'] + np.random.choice([-1, 1])))
                augmented_data.append(new_row)
        
        # Variation with slight vital sign changes (within normal ranges)
        if np.random.random() < 0.2:
            new_row = row.copy()
            # Small variations in heart rate (±5 bpm)
            if pd.notna(row.get('heartrate')):
                new_row['heartrate'] = max(40, min(200, row['heartrate'] + np.random.randint(-5, 6)))
                augmented_data.append(new_row)
    
    final_df = pd.DataFrame(augmented_data)
    print(f"✅ MIMIC data augmentation complete: {len(final_df)} total rows (from {len(valid_data)} original)")
    
    return final_df

def format_mimic_patient_info_enhanced(row):
    """Enhanced MIMIC-IV-ED patient information formatting for 8B model"""
    # Build comprehensive patient information from MIMIC data
    patient_info = "Emergency Department Triage Assessment (MIMIC-IV-ED):\n\n"
    
    # Vital signs section
    patient_info += "VITAL SIGNS:\n"
    if pd.notna(row.get('temperature')):
        patient_info += f"- Temperature: {row['temperature']}°C (Normal: 36.5-37.5°C)\n"
    if pd.notna(row.get('heartrate')):
        patient_info += f"- Heart Rate: {row['heartrate']} bpm (Normal: 60-100)\n"
    if pd.notna(row.get('resprate')):
        patient_info += f"- Respiratory Rate: {row['resprate']} breaths/min (Normal: 12-20)\n"
    if pd.notna(row.get('o2sat')):
        patient_info += f"- Oxygen Saturation: {row['o2sat']}% (Normal: 95-100%)\n"
    if pd.notna(row.get('sbp')) and pd.notna(row.get('dbp')):
        patient_info += f"- Blood Pressure: {row['sbp']}/{row['dbp']} mmHg\n"
    if pd.notna(row.get('pain')):
        patient_info += f"- Pain Level: {row['pain']}/10\n"
    
    # Chief complaint
    if pd.notna(row.get('chiefcomplaint')):
        patient_info += f"\nPRESENTING COMPLAINT:\n- Chief Complaint: {row['chiefcomplaint']}\n"
    
    # Additional context
    patient_info += "\nTRIAGE ASSESSMENT QUESTION:\n"
    patient_info += "Based on the above patient information from MIMIC-IV-ED, determine the appropriate ESI (Emergency Severity Index) triage level.\n"
    patient_info += "Consider vital signs, pain level, and chief complaint severity.\n\n"
    
    patient_info += "ESI Triage Levels:\n"
    patient_info += "1: Immediate, life-saving intervention required\n"
    patient_info += "2: High risk situation, rapid medical intervention within 10 minutes\n"
    patient_info += "3: Urgent but stable, medical intervention within 30 minutes\n"
    patient_info += "4: Less urgent, medical intervention within 1 hour\n"
    patient_info += "5: Non-urgent, medical intervention within 2 hours\n\n"
    
    patient_info += "What is the appropriate ESI triage level for this patient?"
    
    return patient_info

def prepare_mimic_dataset_enhanced(start_idx, end_idx, tokenizer, sampled_df=None):
    """Enhanced MIMIC-IV-ED dataset preparation with validation split for 8B"""
    print(f"📊 Loading MIMIC-IV-ED data from rows {start_idx} to {end_idx} for 8B model")
    
    if sampled_df is not None:
        df = sampled_df
    else:
        df = pd.read_csv(CONFIG['DATASET_PATH'])
    
    if start_idx >= len(df):
        raise ValueError("All MIMIC-IV-ED data has already been processed!")
    
    # Validate and augment MIMIC data
    chunk_df = df.iloc[start_idx:end_idx].copy()
    enhanced_chunk = validate_and_augment_mimic_data(chunk_df)
    
    jsonl_path = Path(CONFIG['CHECKPOINT_DIR']) / f"mimic_dataset_7b_{start_idx}_{end_idx}.jsonl"
    
    # Create JSONL data with enhanced MIMIC formatting
    conversations = []
    for idx, (_, row) in enumerate(tqdm(enhanced_chunk.iterrows(), desc="Preparing MIMIC conversations", total=len(enhanced_chunk))):
        user_msg = format_mimic_patient_info_enhanced(row)
        
        # Map MIMIC acuity (1-5) to ESI triage levels
        # MIMIC acuity 1 = highest severity, 5 = lowest severity (same as ESI)
        esi_level = row['acuity']
        assistant_msg = f"Based on the patient's condition from MIMIC-IV-ED data, the appropriate ESI triage level is: {esi_level}"
        
        convo = {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        conversations.append(convo)
        
        if (idx + 1) % 300 == 0:  # Reduced logging frequency for MIMIC
            current_row = start_idx + idx + 1
            print(f"📝 Prepared {idx + 1} MIMIC conversations for 8B")
    
    # Write to JSONL file
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for convo in conversations:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")
    
    print(f"✅ MIMIC-IV-ED dataset prepared for 8B: {len(conversations)} conversations")
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    
    # Format with chat template
    def formatting_prompts_func(examples):
        texts = []
        for convo_list in examples["messages"]:
            text = tokenizer.apply_chat_template(
                convo_list, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True, batch_size=40)  # Smaller batch size for MIMIC
    
    # Add length column for grouping
    def add_length_column(examples):
        examples["length"] = [len(text) for text in examples["text"]]
        return examples
    
    dataset = dataset.map(add_length_column, batched=True)
    
    # Create train/validation split
    train_dataset, val_dataset = create_train_val_split(dataset, CONFIG['VAL_RATIO'])
    
    # Clean up temporary file
    if jsonl_path.exists():
        jsonl_path.unlink()
    
    return train_dataset, val_dataset

def create_train_val_split(dataset, val_ratio=0.15):
    """Create train/validation split"""
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Shuffle and split
    shuffled_dataset = dataset.shuffle(seed=3407)
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, total_size))
    
    print(f"📚 Train set: {len(train_dataset)} samples")
    print(f"🧪 Validation set: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

def setup_lora_with_unsloth(model):
    """Enhanced LoRA configuration for 8B model with Unsloth optimization ONLY"""
    has_lora = any('lora' in name.lower() for name, _ in model.named_parameters())
    
    if not has_lora:
        print("🔧 Applying enhanced LoRA adapters for MIMIC-ED 8B model with Unsloth...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Optimized rank for 8B with Unsloth
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=32,  # Optimized alpha for 8B
            lora_dropout=0.1,  # Low dropout for stability
            bias="none",  # No bias training for efficiency
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=3407,
            use_rslora=False,  # Disabled for stability
            loftq_config=None
        )
        print("✅ Enhanced LoRA adapters applied for MIMIC-ED 8B with Unsloth (r=16, alpha=32)")
    else:
        print("✅ Model already has LoRA adapters")
    
    return model

def get_enhanced_training_args():
    """Get enhanced training arguments for 8B model with single GPU"""
    return TrainingArguments(
        per_device_train_batch_size=CONFIG['BATCH_SIZE'],
        gradient_accumulation_steps=CONFIG['GRADIENT_ACCUMULATION_STEPS'],
        warmup_steps=CONFIG['WARMUP_STEPS'],
        warmup_ratio=CONFIG['WARMUP_RATIO'],
        num_train_epochs=CONFIG['NUM_EPOCHS_PER_CHUNK'],
        learning_rate=CONFIG['LEARNING_RATE'],
        fp16=False,
        bf16=True,
        logging_steps=CONFIG['LOGGING_STEPS'],
        optim="adamw_8bit",
        weight_decay=CONFIG['WEIGHT_DECAY'],
        lr_scheduler_type=CONFIG['LR_SCHEDULER'],
        seed=3407,
        output_dir=CONFIG['OUTPUT_DIR'],
        save_steps=CONFIG['SAVE_STEPS'],
        save_total_limit=CONFIG['SAVE_TOTAL_LIMIT'],
        eval_strategy="steps",
        eval_steps=CONFIG['EVAL_STEPS'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=CONFIG['GRADIENT_CHECKPOINTING'],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=1,  # Reduced for single GPU
        group_by_length=True,
        max_grad_norm=0.5,  # Reduced gradient clipping for 8B
        # Multi-GPU settings
        ddp_backend=CONFIG['DDP_BACKEND'] if CONFIG['MULTI_GPU'] else None,
        dataloader_drop_last=True,  # Important for multi-GPU
    )
def save_and_upload_model_enhanced(model, tokenizer, processed_rows, chunk_num, metrics=None):
    """Enhanced model saving with metrics and metadata for MIMIC-ED 8B"""
    import time  # Fix: Import time at the beginning of the function
    import json
    import datetime
    from pathlib import Path
    
    timestamp = int(time.time())
    local_ckpt_dir = Path(CONFIG['CHECKPOINT_DIR']) / f"mimic_8b_chunk_{chunk_num}_ckpt_{timestamp}"
    
    print(f"💾 Saving enhanced MIMIC-ED 8B model to {local_ckpt_dir}")
    
    # Save model with metadata
    model.save_pretrained(local_ckpt_dir)
    tokenizer.save_pretrained(local_ckpt_dir)
    
    # Save training metadata
    metadata = {
        "chunk_num": chunk_num,
        "processed_rows": processed_rows,
        "timestamp": timestamp,
        "model_size": "8B",
        "dataset": "MIMIC-IV-ED",
        "training_metrics": metrics or {},
        "model_config": {
            "base_model": CONFIG['BASE_MODEL_NAME'],
            "lora_rank": 16,
            "lora_alpha": 32,
            "max_seq_length": CONFIG['MAX_SEQ_LENGTH'],
            "multi_gpu": CONFIG['MULTI_GPU'],
            "enhanced_features": [
                "LLaMA 3.1 8B Model",
                "MIMIC-IV-ED Dataset",
                "Single GPU Training",
                "Optimized LoRA (r=16, alpha=32)",
                "Gradient Checkpointing",
                "CPU Offloading",
                "Memory Optimized",
                "Enhanced MIMIC prompts"
            ]
        }
    }
    
    with open(local_ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Enhanced MIMIC-ED 8B model saved locally")
    
    # Upload to HuggingFace with FIXED verification
    try:
        print("🚀 Uploading MIMIC-ED 8B model to HuggingFace Hub...")
        
        from huggingface_hub import HfApi, list_repo_commits
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique marker to force new commit
        marker_file = local_ckpt_dir / f"chunk_{chunk_num}_marker_{timestamp_str}.txt"
        with open(marker_file, "w") as f:
            f.write(f"Chunk {chunk_num} - {processed_rows} rows - {timestamp_str}\n")
            f.write(f"Training metrics: {metrics}\n")
        
        api = HfApi(token=CONFIG['HF_TOKEN'])
        
        # Upload with unique commit message
        commit_msg = f"Chunk {chunk_num} - {processed_rows} rows - {timestamp_str}"
        api.upload_folder(
            folder_path=str(local_ckpt_dir),
            repo_id=CONFIG['HF_MODEL_ID'],
            commit_message=commit_msg,
            token=CONFIG['HF_TOKEN']
        )
        
        # CRITICAL: Actually verify upload succeeded
        time.sleep(5)  # Wait for HF to process
        commits = list_repo_commits(CONFIG['HF_MODEL_ID'], token=CONFIG['HF_TOKEN'])
        latest_commit = commits[0].title if commits else ""
        
        if f"Chunk {chunk_num}" in latest_commit and timestamp_str in latest_commit:
            print(f"✅ UPLOAD VERIFIED: {latest_commit}")
            # Update resume index ONLY after successful upload verification
            save_resume_index(processed_rows)
            print(f"📝 Updated last_index.txt to: {processed_rows}")
            
            # Clean up local checkpoint to save space
            if local_ckpt_dir.exists():
                import shutil
                shutil.rmtree(local_ckpt_dir)
                print("🧹 Local checkpoint cleaned up")
        else:
            raise Exception(f"Upload verification failed! Latest: {latest_commit}")
            
    except Exception as e:
        print(f"❌ UPLOAD FAILED: {e}")
        # Revert progress to prevent data loss
        previous_rows = processed_rows - CONFIG['CHUNK_SIZE']
        save_resume_index(max(0, previous_rows))
        print(f"🔄 Reverted last_index.txt to: {max(0, previous_rows)}")
        print("💡 Model is saved locally, you can try uploading manually later")
       
        raise Exception(f"Stopping training - upload failed for chunk {chunk_num}")       
        
def train_chunk_with_unsloth(model, tokenizer, start_idx, end_idx, chunk_num, sampled_df=None):
    """Enhanced training on a specific chunk of MIMIC-IV-ED data for 8B model using Unsloth ONLY"""
    print(f"🚀 Starting Unsloth training for MIMIC-ED 8B chunk {chunk_num}: rows {start_idx} to {end_idx}")
    
    # Prepare enhanced MIMIC dataset with validation split
    train_dataset, val_dataset = prepare_mimic_dataset_enhanced(start_idx, end_idx, tokenizer, sampled_df)
    
    # Setup enhanced LoRA with Unsloth
    model = setup_lora_with_unsloth(model)
    
    # Enable training mode with Unsloth optimizations
    FastLanguageModel.for_training(model)
    print("✅ Unsloth training mode enabled")
    
    # Get enhanced training arguments
    training_args = get_enhanced_training_args()
    
    # Create enhanced trainer with Unsloth optimizations
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG['MAX_SEQ_LENGTH'],
        packing=True,  # Unsloth's efficient packing
        args=training_args,
    )
    
    # Add enhanced callbacks
    early_stopping = EarlyStoppingCallback(
        patience=CONFIG['EARLY_STOPPING_PATIENCE'],
        min_delta=CONFIG['EARLY_STOPPING_MIN_DELTA']
    )
    single_gpu_monitoring = SingleGPUMonitoringCallback()
    
    trainer.add_callback(early_stopping)
    trainer.add_callback(single_gpu_monitoring)
    
    # Train the model with Unsloth optimizations
    print("🚀 Starting Unsloth training for MIMIC-ED 8B model...")
    train_result = trainer.train()
    
    # Get training metrics
    metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
    
    # Save and upload enhanced model
    save_and_upload_model_enhanced(model, tokenizer, end_idx, chunk_num, metrics)
    
    # Clear memory
    clear_memory()
    
    return model, tokenizer

def main():
    """Main training function with continuous processing and enhancements for MIMIC-ED 8B using Unsloth ONLY"""
    
    print("🚀 Starting Enhanced LLaMA 3.1 8B Fine-tuning on SageMaker with MIMIC-IV-ED using Unsloth ONLY")
    print("✨ Features: 8B Model, ml.g4dn.2xlarge Instance, MIMIC-IV-ED Dataset, Unsloth Optimized LoRA")
    
    # Setup environment
    setup_environment()
    
    # Get resume point
    start_idx = get_resume_index()
    
    # Load data info and apply sampling
    df = pd.read_csv(CONFIG['DATASET_PATH'])
    print(f"📊 MIMIC-IV-ED Dataset: {len(df)} total rows")
    
    # Apply data sampling to reduce to manageable size
    if CONFIG['USE_DATA_SAMPLING']:
        df = sample_mimic_data(
            df, 
            max_rows=CONFIG['MAX_TRAINING_ROWS'],
            strategy=CONFIG['SAMPLING_STRATEGY'],
            stratify_by=CONFIG['STRATIFY_BY'],
            random_seed=CONFIG['RANDOM_SEED']
        )
    
    total_rows = len(df)
    if start_idx >= total_rows:
        print("🎉 All MIMIC-IV-ED data has already been processed!")
        return
    
    print(f"📊 After sampling: {total_rows} rows for training")
    # Resume from last_index.txt (should be 12,000 - end of chunk 8)
    start_idx = 0
    if CONFIG['RESUME_FROM_CHECKPOINT'] and os.path.exists(CONFIG['LAST_INDEX_FILE']):
        try:
            with open(CONFIG['LAST_INDEX_FILE'], 'r') as f:
                start_idx = int(f.read().strip())
            print(f"🔄 Resuming from row: {start_idx} (starting chunk 9)")
        except Exception as e:
            print(f"💡 Could not read last index: {e}, starting from beginning")
            start_idx = 0
    else:
        print("💡 No resume file found, starting from beginning")
        start_idx = 0
    
    # Load model and tokenizer with Unsloth
    model, tokenizer = load_model_with_unsloth()
    
    # Set chat template with Unsloth
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    print("✅ Unsloth chat template applied")
    
    chunk_num = 9  # Start from chunk 9 (after chunk 8 completion)
    current_idx = start_idx  # This should be 12000
    print(f"🔍 DEBUG: start_idx = {start_idx}, current_idx = {current_idx}, chunk_num = {chunk_num}")
    
    # Store sampled dataframe for reuse in training
    sampled_dataframe = df
    
    try:
        while current_idx < total_rows:
            # Calculate chunk boundaries
            end_idx = min(current_idx + CONFIG['CHUNK_SIZE'], total_rows)
            
            print(f"📦 Processing MIMIC-ED 8B chunk {chunk_num}: rows {current_idx} to {end_idx}")
            print(f"💾 Memory usage: {get_memory_usage():.1f} MB, GPU: {get_gpu_memory_usage():.1f} GB")
            
            # Train on this chunk with Unsloth
            model, tokenizer = train_chunk_with_unsloth(model, tokenizer, current_idx, end_idx, chunk_num, sampled_dataframe)
            
            # Update progress
            current_idx = end_idx
            chunk_num += 1
            
            # Progress report
            progress = (current_idx / total_rows) * 100
            print(f"📈 Progress: {progress:.1f}% ({current_idx}/{total_rows} rows)")
            
            # Check if we should continue
            if current_idx < total_rows:
                print(f"⏳ {total_rows - current_idx} rows remaining. Continuing to next MIMIC-ED 8B chunk...")
                time.sleep(8)  # Pause between chunks for MIMIC
            else:
                print("🎉 All MIMIC-ED 8B training with Unsloth completed!")
                
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise
    finally:
        # Final cleanup
        clear_memory()
        print("🏁 Training session ended")

if __name__ == "__main__":
    # Run Unsloth-only main training for MIMIC-ED 8B with LLaMA 3.1
    main()
