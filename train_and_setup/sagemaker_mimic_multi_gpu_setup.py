# -*- coding: utf-8 -*-
# ===========================
# SageMaker Multi-GPU T4 + MIMIC-IV-ED Training Setup Script
# Optimized for 7B model with multi-GPU T4 support
# ===========================

import os
import subprocess
import sys
import torch
import psutil

def print_header():
    """Print setup header"""
    print("=" * 70)
    print("🚀 SageMaker Multi-GPU T4 + MIMIC-IV-ED Training Setup")
    print("Optimized for LLaMA 3.2 7B with multi-GPU T4 support")
    print("=" * 70)

def check_multi_gpu():
    """Check multi-GPU availability and memory"""
    print("\n🔍 Checking Multi-GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_gpu_memory = sum([torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)]) / 1e9
        
        print(f"✅ Multi-GPU detected: {gpu_count} devices")
        print(f"🎯 Total GPU Memory: {total_gpu_memory:.1f} GB")
        
        # Display detailed GPU info for each device
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
        
        # Check if we have enough memory for 7B on multiple GPUs
        if total_gpu_memory < 25:
            print(f"\n⚠️  Warning: Total GPU memory ({total_gpu_memory:.1f}GB) might be insufficient for 7B model")
            print("   Consider using CPU offloading or smaller model")
            print("   Recommended: At least 25GB total for 7B model")
        else:
            print(f"\n✅ Multi-GPU memory sufficient for 7B model ({total_gpu_memory:.1f}GB total)")
            
        # Check for T4 GPUs specifically
        t4_count = sum(1 for i in range(gpu_count) if "T4" in torch.cuda.get_device_properties(i).name)
        if t4_count > 0:
            print(f"✅ T4 GPUs detected: {t4_count}/{gpu_count}")
        else:
            print("⚠️  No T4 GPUs detected, but other GPU types will work")
            
        return True
    else:
        print("❌ No GPU detected! Training will be very slow on CPU")
        print("   Consider using a multi-GPU instance")
        return False

def install_dependencies():
    """Install optimized dependencies for multi-GPU 7B model"""
    print("\n📦 Installing dependencies for Multi-GPU 7B model...")
    
    # Core PyTorch with CUDA support
    print("Installing: torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    result = subprocess.run([
        "pip", "install", "--quiet", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 installed")
    else:
        print("❌ Failed to install PyTorch")
        print(result.stderr)
        return False
    
    # Transformers with specific version for 7B
    print("Installing: transformers==4.35.0")
    result = subprocess.run([
        "pip", "install", "--quiet", "transformers==4.35.0"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ transformers==4.35.0 installed")
    else:
        print("❌ Failed to install transformers")
        print(result.stderr)
        return False
    
    # Datasets with specific version
    print("Installing: datasets>=3.4.1,<4.0.0")
    result = subprocess.run([
        "pip", "install", "--quiet", "datasets>=3.4.1,<4.0.0"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ datasets>=3.4.1,<4.0.0 installed")
    else:
        print("❌ Failed to install datasets")
        print(result.stderr)
        return False
    
    # HuggingFace Hub
    print("Installing: huggingface_hub>=0.34.0")
    result = subprocess.run([
        "pip", "install", "--quiet", "huggingface_hub>=0.34.0"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ huggingface_hub>=0.34.0 installed")
    else:
        print("❌ Failed to install huggingface_hub")
        print(result.stderr)
        return False
    
    # Accelerate for distributed training
    print("Installing: accelerate")
    result = subprocess.run([
        "pip", "install", "--quiet", "accelerate"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ accelerate installed")
    else:
        print("❌ Failed to install accelerate")
        print(result.stderr)
        return False
    
    # PEFT for LoRA
    print("Installing: peft")
    result = subprocess.run([
        "pip", "install", "--quiet", "peft"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ peft installed")
    else:
        print("❌ Failed to install peft")
        print(result.stderr)
        return False
    
    # TRL for training
    print("Installing: trl")
    result = subprocess.run([
        "pip", "install", "--quiet", "trl"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ trl installed")
    else:
        print("❌ Failed to install trl")
        print(result.stderr)
        return False
    
    # Bitsandbytes for quantization
    print("Installing: bitsandbytes")
    result = subprocess.run([
        "pip", "install", "--quiet", "bitsandbytes"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ bitsandbytes installed")
    else:
        print("❌ Failed to install bitsandbytes")
        print(result.stderr)
        return False
    
    # Xformers for memory efficiency
    print("Installing: xformers")
    result = subprocess.run([
        "pip", "install", "--quiet", "xformers"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ xformers installed")
    else:
        print("❌ Failed to install xformers")
        print(result.stderr)
        return False
    
    # Triton for optimization
    print("Installing: triton")
    result = subprocess.run([
        "pip", "install", "--quiet", "triton"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ triton installed")
    else:
        print("❌ Failed to install triton")
        print(result.stderr)
        return False
    
    # TQDM for progress bars
    print("Installing: tqdm")
    result = subprocess.run([
        "pip", "install", "--quiet", "tqdm"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ tqdm installed")
    else:
        print("❌ Failed to install tqdm")
        print(result.stderr)
        return False
    
    # PSUTIL for memory monitoring
    print("Installing: psutil")
    result = subprocess.run([
        "pip", "install", "--quiet", "psutil"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ psutil installed")
    else:
        print("❌ Failed to install psutil")
        print(result.stderr)
        return False
    
    # Unsloth for optimized training
    print("Installing: unsloth")
    result = subprocess.run([
        "pip", "install", "--quiet", "unsloth"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ unsloth installed")
    else:
        print("❌ Failed to install unsloth")
        print(result.stderr)
        return False
    
    # Unsloth Zoo for model loading
    print("Installing: unsloth_zoo")
    result = subprocess.run([
        "pip", "install", "--quiet", "unsloth_zoo"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ unsloth_zoo installed")
    else:
        print("❌ Failed to install unsloth_zoo")
        print(result.stderr)
        return False
    
    return True

def setup_multi_gpu_optimizations():
    """Setup multi-GPU optimizations for 7B model"""
    print("\n💾 Setting up multi-GPU optimizations for 7B model...")
    
    # Set environment variables for multi-GPU optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'  # Enable NCCL debugging
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand for AWS
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P for AWS
    
    # Set memory fraction for each GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            torch.cuda.set_per_process_memory_fraction(0.8, i)  # 80% per GPU
            print(f"✅ GPU {i} memory fraction set to 80%")
    
    print("✅ Multi-GPU optimizations configured")

def create_mimic_directories():
    """Create necessary directories for MIMIC-IV-ED 7B training"""
    print("\n📁 Creating directories for MIMIC-IV-ED 7B training...")
    
    directories = [
        "./llama_7b_mimic_checkpoints",
        "./llama_7b_mimic_checkpoints/outputs",
        "./mimic_data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def verify_multi_gpu_installation():
    """Verify all packages are installed correctly for multi-GPU"""
    print("\n🔍 Verifying multi-GPU installation...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ NCCL available: {torch.cuda.nccl.is_available()}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not installed")
        return False
    
    try:
        import huggingface_hub
        print(f"✅ HuggingFace Hub: {huggingface_hub.__version__}")
    except ImportError:
        print("❌ HuggingFace Hub not installed")
        return False
    
    try:
        import accelerate
        print(f"✅ Accelerate: {accelerate.__version__}")
    except ImportError:
        print("❌ Accelerate not installed")
        return False
    
    try:
        import peft
        print(f"✅ PEFT: {peft.__version__}")
    except ImportError:
        print("❌ PEFT not installed")
        return False
    
    try:
        import trl
        print(f"✅ TRL: {trl.__version__}")
    except ImportError:
        print("❌ TRL not installed")
        return False
    
    try:
        import bitsandbytes
        print(f"✅ Bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        print("❌ Bitsandbytes not installed")
        return False
    
    try:
        import xformers
        print(f"✅ Xformers: {xformers.__version__}")
    except ImportError:
        print("❌ Xformers not installed")
        return False
    
    try:
        import unsloth
        print("✅ Unsloth installed")
    except ImportError:
        print("❌ Unsloth not installed")
        return False
    
    try:
        import unsloth_zoo
        print("✅ Unsloth Zoo installed")
    except ImportError:
        print("❌ Unsloth Zoo not installed")
        return False
    
    return True

def setup_mimic_data_preparation():
    """Setup instructions for MIMIC-IV-ED data preparation"""
    print("\n📊 MIMIC-IV-ED Data Preparation Setup...")
    
    print("✅ MIMIC-IV-ED data directories created")
    print("\n📋 Next steps for MIMIC-IV-ED data:")
    print("1. Download MIMIC-IV-ED from PhysioNet (requires credentials)")
    print("2. Extract and process the following tables:")
    print("   - edstays: Patient stay information")
    print("   - triage: Triage data with acuity levels (1-5)")
    print("   - vitalsign: Vital signs during ED stay")
    print("   - diagnosis: ICD diagnosis codes")
    print("3. Create a combined CSV file named 'mimic_ed_data.csv'")
    print("4. Ensure the CSV has these columns:")
    print("   - acuity: Triage acuity (1-5, where 1=highest severity)")
    print("   - chiefcomplaint: Patient's chief complaint")
    print("   - temperature, heartrate, resprate, o2sat, sbp, dbp, pain")
    print("5. Place the CSV file in the project root directory")

def main():
    """Main setup function for multi-GPU MIMIC-IV-ED training"""
    print_header()
    
    # Check multi-GPU
    gpu_available = check_multi_gpu()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed!")
        return False
    
    # Setup multi-GPU optimizations
    setup_multi_gpu_optimizations()
    
    # Create directories
    create_mimic_directories()
    
    # Setup MIMIC data preparation
    setup_mimic_data_preparation()
    
    # Verify installation
    if not verify_multi_gpu_installation():
        print("❌ Installation verification failed!")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 SageMaker Multi-GPU T4 + MIMIC-IV-ED Setup Complete!")
    print("=" * 70)
    print("\n📋 Next Steps:")
    print("1. Prepare your MIMIC-IV-ED dataset as 'mimic_ed_data.csv'")
    print("2. Run the MIMIC training script: python sagemaker_7b_mimic_training.py")
    print("3. Monitor training progress in the logs")
    print("\n⚠️  Important Notes for Multi-GPU 7B Model:")
    print("- Training will be faster with multiple GPUs")
    print("- Memory usage will be distributed across GPUs")
    print("- Each GPU will use 80% of its memory")
    print("- Monitor GPU utilization across all devices")
    print("- MIMIC-IV-ED data provides real clinical triage scenarios")
    print("- Ensure compliance with PhysioNet data use agreement")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
