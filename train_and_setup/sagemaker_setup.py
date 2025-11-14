#!/usr/bin/env python3
"""
SageMaker Setup Script for LLaMA Training
Quick setup and verification for your GPU environment
"""

import os
import sys
import subprocess
import torch

def check_gpu():
    """Check GPU availability and info"""
    print("🔍 Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU detected: {gpu_count} devices")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
        # Check current device
        current_device = torch.cuda.current_device()
        print(f"🎯 Using GPU: {current_device}")
        
        return True
    else:
        print("❌ No GPU detected!")
        return False

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "transformers==4.35.0",
        "datasets>=3.4.1,<4.0.0",
        "accelerate peft trl",
        "bitsandbytes xformers==0.0.29.post3",
        "unsloth triton cut_cross_entropy unsloth_zoo",
        "pandas numpy scikit-learn",
        "tqdm psutil sentencepiece protobuf",
        "huggingface_hub>=0.34.0 hf_transfer"
    ]
    
    for package in packages:
        print(f"Installing: {package}")
        try:
            subprocess.run(f"pip install --quiet {package}", shell=True, check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    dirs = [
        "./llama_checkpoints",
        "./llama_checkpoints/outputs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def check_dataset():
    """Check if dataset exists"""
    print("📊 Checking dataset...")
    
    if os.path.exists("./esi_data.csv"):
        print("✅ Dataset found: esi_data.csv")
        
        # Quick dataset info
        try:
            import pandas as pd
            df = pd.read_csv("./esi_data.csv")
            print(f"  📋 Shape: {df.shape}")
            print(f"  🔢 Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print("❌ Dataset not found: esi_data.csv")
        print("   Please upload your CSV file to the notebook")
        return False

def main():
    """Main setup function"""
    print("🚀 SageMaker LLaMA Training Setup")
    print("=" * 50)
    
    # Check GPU
    gpu_ok = check_gpu()
    if not gpu_ok:
        print("⚠️  Warning: No GPU detected. Training will be very slow!")
    
    # Install dependencies
    deps_ok = install_dependencies()
    if not deps_ok:
        print("❌ Dependency installation failed!")
        return False
    
    # Create directories
    create_directories()
    
    # Check dataset
    dataset_ok = check_dataset()
    if not dataset_ok:
        print("⚠️  Dataset not ready. Please upload esi_data.csv")
    
    print("\n" + "=" * 50)
    if gpu_ok and deps_ok and dataset_ok:
        print("🎉 Setup complete! Ready to train!")
        print("Run the training script: sagemaker_llama_training.py")
    else:
        print("⚠️  Setup completed with warnings. Check issues above.")
    
    return True

if __name__ == "__main__":
    main()

