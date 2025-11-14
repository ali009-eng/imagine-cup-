# -*- coding: utf-8 -*-
# ===========================
# SageMaker LLaMA 3.2 7B Training Setup Script
# Optimized for 7B model with advanced memory management
# ===========================

import os
import subprocess
import sys
import torch
import psutil

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🚀 SageMaker LLaMA 3.2 7B Training Setup")
    print("Optimized for 7B model with advanced memory management")
    print("=" * 60)

def check_gpu():
    """Check GPU availability and memory"""
    print("\n🔍 Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU detected: {gpu_count} devices")
        print(f"🎯 Total GPU Memory: {gpu_memory:.1f} GB")
        
        # Display detailed GPU info
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Check if we have enough memory for 7B
        if gpu_memory < 15:
            print("⚠️  Warning: GPU memory might be insufficient for 7B model")
            print("   Consider using CPU offloading or smaller model")
        else:
            print("✅ GPU memory sufficient for 7B model")
            
        return True
    else:
        print("❌ No GPU detected! Training will be very slow on CPU")
        print("   Consider using a GPU instance")
        return False

def install_dependencies():
    """Install optimized dependencies for 7B model"""
    print("\n📦 Installing dependencies for 7B model...")
    
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

def setup_memory_optimizations():
    """Setup memory optimizations for 7B model"""
    print("\n💾 Setting up memory optimizations for 7B model...")
    
    # Set environment variables for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Set memory fraction for GPU
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)
        print("✅ GPU memory fraction set to 85%")
    
    print("✅ Memory optimizations configured")

def create_directories():
    """Create necessary directories for 7B training"""
    print("\n📁 Creating directories for 7B training...")
    
    directories = [
        "./llama_7b_checkpoints",
        "./llama_7b_checkpoints/outputs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def verify_installation():
    """Verify all packages are installed correctly"""
    print("\n🔍 Verifying installation...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
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

def main():
    """Main setup function"""
    print_header()
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed!")
        return False
    
    # Setup memory optimizations
    setup_memory_optimizations()
    
    # Create directories
    create_directories()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed!")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 SageMaker LLaMA 3.2 7B Setup Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Upload your 'esi_data.csv' file to the SageMaker instance")
    print("2. Run the 7B training script: python sagemaker_llama_7b_training.py")
    print("3. Monitor training progress in the logs")
    print("\n⚠️  Important Notes for 7B Model:")
    print("- Training will be slower than 3B model")
    print("- Memory usage will be higher")
    print("- Consider using CPU offloading if GPU memory is insufficient")
    print("- Monitor GPU memory usage during training")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
