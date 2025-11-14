# -*- coding: utf-8 -*-
# ===========================
# Enhanced SageMaker LLaMA 3.2 7B Fine-Tuning Script
# Optimized for 7B model with advanced memory management
# ===========================

import os
import time
import json
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from huggingface_hub import login, create_repo
import gc
import psutil
import logging
from tqdm import tqdm

# === Enhanced Configuration for LLaMA 3.2 7B ===
CONFIG = {
    'BASE_MODEL_NAME': "unsloth/llama-3.2-7b-bnb-4bit",  # ✅ Upgraded to 7B
    'HF_MODEL_NAME': "ali009eng/llama-7b-triage-classifier",  # ✅ New repo for 7B
    'HF_MODEL_ID': "ali009eng/llama-7b-triage-classifier",  # ✅ New repo for 7B
    'HF_TOKEN': "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",
    'MAX_SEQ_LENGTH': 2048,
    'LOAD_IN_4BIT': True,
    'CHUNK_SIZE': 2000,  # Smaller chunks for 7B memory management
    'BATCH_SIZE': 1,      # Reduced batch size for 7B
    'LEARNING_RATE': 5e-5,  # Lower learning rate for 7B stability
    'CHECKPOINT_DIR': "./llama_7b_checkpoints",  # ✅ New checkpoint dir
    'DATASET_PATH': "./esi_data.csv",
    'OUTPUT_DIR': "./llama_7b_checkpoints/outputs",  # ✅ New output dir
    'SAVE_STEPS': 200,    # Save less frequently to save memory
    'LOGGING_STEPS': 5,   # Log every 5 steps
    'GRADIENT_ACCUMULATION_STEPS': 32,  # Increased for effective batch size
    'WARMUP_STEPS': 300,  # More warmup for 7B
    'NUM_EPOCHS_PER_CHUNK': 2,  # Fewer epochs per chunk for 7B
    'WEIGHT_DECAY': 0.01,  # Reduced regularization for 7B
    'LR_SCHEDULER': "cosine",  # Simpler scheduler for 7B
    'WARMUP_RATIO': 0.1,
    'EVAL_STEPS': 100,    # Evaluate less frequently
    'SAVE_TOTAL_LIMIT': 3,  # Keep fewer checkpoints
    'VAL_RATIO': 0.1,
    'EARLY_STOPPING_PATIENCE': 2,  # Less patience for 7B
    'EARLY_STOPPING_MIN_DELTA': 0.002,  # Higher threshold
    'GRADIENT_CHECKPOINTING': True,  # ✅ Enable for 7B
    'CPU_OFFLOAD': True,  # ✅ Enable CPU offloading
}

# Setup logging for SageMaker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./llama_7b_checkpoints/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Enhanced Callback Classes ===
class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback to prevent overfitting"""
    
    def __init__(self, patience=2, min_delta=0.002):
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
            logger.info(f" New best validation loss: {eval_loss:.4f}")
        else:
            self.wait += 1
            logger.info(f" Early stopping counter: {self.wait}/{self.patience}")
            
        if self.wait >= self.patience:
            control.should_training_stop = True
            self.stopped_epoch = state.epoch
            logger.info(f" Early stopping triggered at epoch {state.epoch}")

class MemoryMonitoringCallback(TrainerCallback):
    """Memory monitoring callback for 7B model"""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 20 == 0:  # Log every 20 steps
            memory_usage = get_memory_usage()
            gpu_memory = get_gpu_memory_usage()
            logger.info(f" Step {state.global_step}: CPU Memory: {memory_usage:.1f} MB, GPU Memory: {gpu_memory:.1f} GB")

def setup_environment():
    """Setup SageMaker environment and directories"""
    # Create local directories
    os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # Setup Hugging Face
    try:
        login(CONFIG['HF_TOKEN'])
        create_repo(CONFIG['HF_MODEL_ID'], exist_ok=True, repo_type="model")
        logger.info(" HuggingFace setup completed for 7B model")
    except Exception as e:
        logger.error(f" HuggingFace setup failed: {e}")
        raise
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f" GPU detected: {gpu_count} devices, {gpu_memory:.1f}GB total memory")
        
        # Set memory fraction for 7B model
        torch.cuda.set_per_process_memory_fraction(0.85)  # Reduced for 7B
        logger.info(" GPU memory fraction set to 85% for 7B model")
        
        # Display GPU info
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f}GB")
    else:
        logger.warning(" No GPU detected! Training will be very slow on CPU")

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
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9  # GB
    return 0

def clear_memory():
    """Clear memory and garbage collect"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_model_with_proper_fallback():
    """Load 7B model with proper LoRA adapter handling"""
    try:
        logger.info(f"Loading 7B base model: {CONFIG['BASE_MODEL_NAME']}")
        
        # Load base model with CPU offloading for 7B
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG['BASE_MODEL_NAME'],
            max_seq_length=CONFIG['MAX_SEQ_LENGTH'],
            dtype=None,
            load_in_4bit=CONFIG['LOAD_IN_4BIT'],
            device_map="auto" if CONFIG['CPU_OFFLOAD'] else None,  # ✅ Auto device mapping
            low_cpu_mem_usage=True  # ✅ Low CPU memory usage
        )
        
        # Try to load existing LoRA adapter
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                base_model, 
                CONFIG['HF_MODEL_NAME'],
                token=CONFIG['HF_TOKEN'],
                device_map="auto" if CONFIG['CPU_OFFLOAD'] else None
            )
            logger.info(" Successfully loaded existing LoRA adapter for 7B")
        except Exception as e:
            logger.info(f"Could not load LoRA adapter: {e}")
            logger.info("Will create new enhanced LoRA adapters for 7B")
            model = base_model
            
        return model, tokenizer
            
    except Exception as e:
        logger.error(f"Error loading 7B model: {e}")
        raise

def validate_and_augment_data(df):
    """Validate and augment training data for 7B model"""
    logger.info("🔍 Validating and augmenting data for 7B model...")
    
    # Data validation
    initial_count = len(df)
    valid_data = df.dropna()
    logger.info(f" Data validation: {len(valid_data)}/{initial_count} rows valid")
    
    # Data augmentation - create variations of the same case
    augmented_data = []
    
    for _, row in tqdm(valid_data.iterrows(), desc="Augmenting data", total=len(valid_data)):
        # Original case
        augmented_data.append(row)
        
        # Create slight variations for better generalization
        if row['pain_level'] > 0 and np.random.random() < 0.3:  # Reduced augmentation for 7B
            # Variation with slightly different pain level
            new_row = row.copy()
            new_row['pain_level'] = max(0, min(10, row['pain_level'] + np.random.choice([-1, 1])))
            augmented_data.append(new_row)
        
        # Variation with slight vital sign changes (within normal ranges)
        if np.random.random() < 0.2:  # Reduced chance for 7B
            new_row = row.copy()
            # Small variations in heart rate (±5 bpm)
            new_row['heart_rate'] = max(40, min(200, row['heart_rate'] + np.random.randint(-5, 6)))
            augmented_data.append(new_row)
    
    final_df = pd.DataFrame(augmented_data)
    logger.info(f" Data augmentation complete: {len(final_df)} total rows (from {len(valid_data)} original)")
    
    return final_df

def format_patient_info_enhanced(row):
    """Enhanced patient information formatting for 7B model"""
    # Add triage level descriptions for better context
    triage_descriptions = {
        1: "Immediate, life-saving intervention required",
        2: "High risk situation, rapid medical intervention within 10 minutes",
        3: "Urgent but stable, medical intervention within 30 minutes",
        4: "Less urgent, medical intervention within 1 hour",
        5: "Non-urgent, medical intervention within 2 hours"
    }
    
    return (
        f"Emergency Department Triage Assessment:\n\n"
        f"PATIENT DEMOGRAPHICS:\n"
        f"- Name: {row['patient_name']}\n"
        f"- Age: {row['age']} years\n"
        f"- Gender: {row['gender']}\n\n"
        f"PRESENTING COMPLAINT:\n"
        f"- Chief Complaint: {row['chief_complaint']}\n\n"
        f"VITAL SIGNS:\n"
        f"- Heart Rate: {row['heart_rate']} bpm (Normal: 60-100)\n"
        f"- Respiratory Rate: {row['respiratory_rate']} breaths/min (Normal: 12-20)\n"
        f"- Oxygen Saturation: {row['oxygen_saturation']}% (Normal: 95-100%)\n"
        f"- Temperature: {row['temperature']}°C (Normal: 36.5-37.5°C)\n"
        f"- Blood Pressure: {row['blood_pressure']} mmHg\n"
        f"- Pain Level: {row['pain_level']}/10\n\n"
        f"TRIAGE ASSESSMENT QUESTION:\n"
        f"Based on the above patient information, determine the appropriate ESI (Emergency Severity Index) triage level.\n"
        f"Consider vital signs, pain level, and chief complaint severity.\n\n"
        f"ESI Triage Levels:\n"
        f"1: Immediate, life-saving intervention required\n"
        f"2: High risk situation, rapid medical intervention within 10 minutes\n"
        f"3: Urgent but stable, medical intervention within 30 minutes\n"
        f"4: Less urgent, medical intervention within 1 hour\n"
        f"5: Non-urgent, medical intervention within 2 hours\n\n"
        f"What is the appropriate ESI triage level for this patient?"
    )

def prepare_dataset_enhanced(start_idx, end_idx, tokenizer):
    """Enhanced dataset preparation with validation split for 7B"""
    logger.info(f"Loading data from rows {start_idx} to {end_idx} for 7B model")
    
    df = pd.read_csv(CONFIG['DATASET_PATH'])
    
    if start_idx >= len(df):
        raise ValueError("All data has already been processed!")
    
    # Validate and augment data
    chunk_df = df.iloc[start_idx:end_idx].copy()
    enhanced_chunk = validate_and_augment_data(chunk_df)
    
    jsonl_path = Path(CONFIG['CHECKPOINT_DIR']) / f"triage_dataset_7b_{start_idx}_{end_idx}.jsonl"
    
    # Create JSONL data with enhanced formatting
    conversations = []
    for idx, (_, row) in enumerate(tqdm(enhanced_chunk.iterrows(), desc="Preparing conversations", total=len(enhanced_chunk))):
        user_msg = format_patient_info_enhanced(row)
        assistant_msg = f"Based on the patient's condition, the appropriate ESI triage level is: {row['triage_level']}"
        
        convo = {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        conversations.append(convo)
        
        if (idx + 1) % 500 == 0:  # Reduced logging frequency for 7B
            current_row = start_idx + idx + 1
            logger.info(f"Prepared {idx + 1} conversations for 7B")
    
    # Write to JSONL file
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for convo in conversations:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")
    
    logger.info(f"Dataset prepared for 7B: {len(conversations)} conversations")
    
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
    
    dataset = dataset.map(formatting_prompts_func, batched=True, batch_size=50)  # Smaller batch size for 7B
    
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

def create_train_val_split(dataset, val_ratio=0.1):
    """Create train/validation split"""
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Shuffle and split
    shuffled_dataset = dataset.shuffle(seed=3407)
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, total_size))
    
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f" Validation set: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

def setup_lora_enhanced(model):
    """Enhanced LoRA configuration for 7B model"""
    has_lora = any('lora' in name.lower() for name, _ in model.named_parameters())
    
    if not has_lora:
        logger.info("Applying enhanced LoRA adapters for 7B model...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Reduced rank for 7B memory efficiency
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
                # Removed lm_head for 7B memory efficiency
            ],
            lora_alpha=32,  # Reduced alpha for 7B
            lora_dropout=0.1,  # Reduced dropout for 7B
            bias="none",  # No bias training for 7B
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,  # Disabled for 7B stability
            loftq_config=None
        )
        logger.info("✅ Enhanced LoRA adapters applied for 7B (r=16, alpha=32)")
    else:
        logger.info("✅ Model already has LoRA adapters")
    
    return model

def get_enhanced_training_args():
    """Get enhanced training arguments for 7B model"""
    return TrainingArguments(
        per_device_train_batch_size=CONFIG['BATCH_SIZE'],
        gradient_accumulation_steps=CONFIG['GRADIENT_ACCUMULATION_STEPS'],
        warmup_steps=CONFIG['WARMUP_STEPS'],
        warmup_ratio=CONFIG['WARMUP_RATIO'],
        num_train_epochs=CONFIG['NUM_EPOCHS_PER_CHUNK'],
        learning_rate=CONFIG['LEARNING_RATE'],
        fp16=True,
        bf16=False,
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
        gradient_checkpointing=CONFIG['GRADIENT_CHECKPOINTING'],  # ✅ Enable for 7B
        ddp_find_unused_parameters=False,
        dataloader_num_workers=1,  # Reduced for 7B
        group_by_length=True,
        max_grad_norm=0.5,  # Reduced gradient clipping for 7B
    )

def save_and_upload_model_enhanced(model, tokenizer, processed_rows, chunk_num, metrics=None):
    """Enhanced model saving with metrics and metadata for 7B"""
    timestamp = int(time.time())
    local_ckpt_dir = Path(CONFIG['CHECKPOINT_DIR']) / f"7b_chunk_{chunk_num}_ckpt_{timestamp}"
    
    logger.info(f"Saving enhanced 7B model to {local_ckpt_dir}")
    
    # Save model with metadata
    model.save_pretrained(local_ckpt_dir)
    tokenizer.save_pretrained(local_ckpt_dir)
    
    # Save training metadata
    metadata = {
        "chunk_num": chunk_num,
        "processed_rows": processed_rows,
        "timestamp": timestamp,
        "model_size": "7B",
        "training_metrics": metrics or {},
        "model_config": {
            "base_model": CONFIG['BASE_MODEL_NAME'],
            "lora_rank": 16,
            "lora_alpha": 32,
            "max_seq_length": CONFIG['MAX_SEQ_LENGTH'],
            "enhanced_features": [
                "LLaMA 3.2 7B Model",
                "Optimized LoRA (r=16, alpha=32)",
                "Gradient Checkpointing",
                "CPU Offloading",
                "Memory Optimized",
                "Enhanced prompts"
            ]
        }
    }
    
    with open(local_ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Update resume index
    save_resume_index(processed_rows)
    
    logger.info(f"✅ Enhanced 7B model saved locally")
    
    # Upload to HuggingFace with better commit messages
    try:
        logger.info("Uploading 7B model to HuggingFace Hub...")
        
        commit_message = (
            f"LLaMA 3.2 7B Chunk {chunk_num} completed - {processed_rows} rows processed\n"
            f"Training metrics: {metrics}\n"
            f"Optimized LoRA config: r=16, alpha=32\n"
            f"Features: Gradient Checkpointing, CPU Offloading, Memory Optimized"
        )
        
        model.push_to_hub(
            CONFIG['HF_MODEL_ID'],
            token=CONFIG['HF_TOKEN'],
            commit_message=commit_message
        )
        
        tokenizer.push_to_hub(
            CONFIG['HF_MODEL_ID'],
            token=CONFIG['HF_TOKEN'],
            commit_message=f"7B tokenizer for chunk {chunk_num}"
        )
        
        logger.info("✅ Enhanced 7B model uploaded to HuggingFace Hub successfully!")
        
        # Clean up local checkpoint to save space
        if local_ckpt_dir.exists():
            import shutil
            shutil.rmtree(local_ckpt_dir)
            logger.info("Local checkpoint cleaned up")
        
    except Exception as e:
        logger.error(f" Failed to upload to HuggingFace: {e}")
        logger.info("Model is saved locally, you can try uploading manually later")

def train_chunk_enhanced(model, tokenizer, start_idx, end_idx, chunk_num):
    """Enhanced training on a specific chunk of data for 7B model"""
    logger.info(f" Starting enhanced training for 7B chunk {chunk_num}: rows {start_idx} to {end_idx}")
    
    # Prepare enhanced dataset with validation split
    train_dataset, val_dataset = prepare_dataset_enhanced(start_idx, end_idx, tokenizer)
    
    # Setup enhanced LoRA
    model = setup_lora_enhanced(model)
    
    # Enable training mode
    FastLanguageModel.for_training(model)
    
    # Get enhanced training arguments
    training_args = get_enhanced_training_args()
    
    # Create enhanced trainer with callbacks
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG['MAX_SEQ_LENGTH'],
        packing=True,
        args=training_args,
    )
    
    # Add enhanced callbacks
    early_stopping = EarlyStoppingCallback(
        patience=CONFIG['EARLY_STOPPING_PATIENCE'],
        min_delta=CONFIG['EARLY_STOPPING_MIN_DELTA']
    )
    memory_monitoring = MemoryMonitoringCallback()
    
    trainer.add_callback(early_stopping)
    trainer.add_callback(memory_monitoring)
    
    # Train the model
    logger.info("Starting enhanced training for 7B model...")
    train_result = trainer.train()
    
    # Get training metrics
    metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
    
    # Save and upload enhanced model
    save_and_upload_model_enhanced(model, tokenizer, end_idx, chunk_num, metrics)
    
    # Clear memory
    clear_memory()
    
    return model, tokenizer

def main():
    """Main training function with continuous processing and enhancements for 7B"""
    logger.info("🚀 Starting Enhanced LLaMA 3.2 7B Fine-tuning on SageMaker")
    logger.info("✨ Features: 7B Model, Optimized LoRA, Gradient Checkpointing, CPU Offloading")
    
    # Setup environment
    setup_environment()
    
    # Get resume point
    start_idx = get_resume_index()
    
    # Load data info
    df = pd.read_csv(CONFIG['DATASET_PATH'])
    total_rows = len(df)
    
    if start_idx >= total_rows:
        logger.info("🎉 All data has already been processed!")
        return
    
    logger.info(f"📊 Dataset: {total_rows} total rows")
    logger.info(f"🔄 Resuming from row: {start_idx}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_with_proper_fallback()
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")
    
    chunk_num = 1
    current_idx = start_idx
    
    try:
        while current_idx < total_rows:
            # Calculate chunk boundaries
            end_idx = min(current_idx + CONFIG['CHUNK_SIZE'], total_rows)
            
            logger.info(f"📦 Processing enhanced 7B chunk {chunk_num}: rows {current_idx} to {end_idx}")
            logger.info(f"💾 Memory usage: {get_memory_usage():.1f} MB, GPU: {get_gpu_memory_usage():.1f} GB")
            
            # Train on this chunk with enhancements
            model, tokenizer = train_chunk_enhanced(model, tokenizer, current_idx, end_idx, chunk_num)
            
            # Update progress
            current_idx = end_idx
            chunk_num += 1
            
            # Progress report
            progress = (current_idx / total_rows) * 100
            logger.info(f"📈 Progress: {progress:.1f}% ({current_idx}/{total_rows} rows)")
            
            # Check if we should continue
            if current_idx < total_rows:
                logger.info(f"⏳ {total_rows - current_idx} rows remaining. Continuing to next enhanced 7B chunk...")
                time.sleep(10)  # Longer pause between chunks for 7B
            else:
                logger.info("🎉 All enhanced 7B training completed!")
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        # Final cleanup
        clear_memory()
        logger.info("Training session ended")

if __name__ == "__main__":
    # Run enhanced main training for 7B
    main()
