# ===========================================
# 🚨 SAGEMAKER LLAMA TRAINING - DEPENDENCY FIX
# ===========================================

print("🚨 STARTING NUCLEAR RESET FOR SAGEMAKER...")
print("="*60)

# STEP 1: CLEAN SLATE
print("🧹 STEP 1: Clean Slate - Removing conflicting packages...")
print("-" * 40)

packages_to_remove = [
    "transformers", "huggingface_hub", "tokenizers", "trl", "peft", 
    "unsloth", "unsloth_zoo", "datasets", "pyarrow"
]

for package in packages_to_remove:
    try:
        print(f"🗑️  Removing {package}...")
        !pip uninstall -y {package}
    except:
        print(f"⚠️  {package} not found or already removed")

print("�� Clearing pip cache...")
!pip cache purge

print("✅ Clean slate completed!")
print()

# STEP 2: INSTALL EXACT WORKING VERSIONS
print("�� STEP 2: Installing exact working versions...")
print("-" * 40)

# PyTorch with CUDA 11.8
print("�� Installing PyTorch with CUDA 11.8...")
!pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Core ML libraries (force reinstall, no deps)
print("�� Installing Core ML libraries...")
!pip install transformers==4.51.3 --force-reinstall --no-deps
!pip install huggingface_hub==0.34.4 --force-reinstall --no-deps
!pip install tokenizers==0.21.4 --force-reinstall --no-deps

# LoRA and training
print("🔥 Installing LoRA and training libraries...")
!pip install trl==0.18.0 peft==0.17.1

# Unsloth (from GitHub)
print("�� Installing Unsloth from GitHub...")
!pip install git+https://github.com/unslothai/unsloth.git
!pip install git+https://github.com/unslothai/unsloth-zoo.git

print("✅ All packages installed!")
print()

# STEP 3: VERIFY INSTALLATION
print("🧪 STEP 3: Verifying installation...")
print("-" * 40)

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        print(f"✅ Current GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers error: {e}")

try:
    import huggingface_hub
    print(f"✅ HuggingFace Hub: {huggingface_hub.__version__}")
except Exception as e:
    print(f"❌ HuggingFace Hub error: {e}")

try:
    import tokenizers
    print(f"✅ Tokenizers: {tokenizers.__version__}")
except Exception as e:
    print(f"❌ Tokenizers error: {e}")

try:
    import trl
    print(f"✅ TRL: {trl.__version__}")
except Exception as e:
    print(f"❌ TRL error: {e}")

try:
    import peft
    print(f"✅ PEFT: {peft.__version__}")
except Exception as e:
    print(f"❌ PEFT error: {e}")

try:
    import unsloth
    print(f"✅ Unsloth: {unsloth.__version__}")
except Exception as e:
    print(f"❌ Unsloth error: {e}")

try:
    import unsloth_zoo
    print(f"✅ Unsloth Zoo: {unsloth_zoo.__version__}")
except Exception as e:
    print(f"❌ Unsloth Zoo error: {e}")

