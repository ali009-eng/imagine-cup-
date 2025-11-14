# Create dependency fix file in current directory
import os

# Create the dependency fix file
dependency_fix_content = """# DEPENDENCY FIX FOR SAGEMAKER LLAMA TRAINING

## 🚨 THE PROBLEM
SageMaker base environment has conflicting packages that cause:
- transformers version conflicts (4.35.0 vs 4.51.3)
- huggingface_hub mismatches
- pip dependency resolver warnings
- Import errors with aria.configuration_aria
- CUDA compatibility issues

## 🔥 THE NUCLEAR RESET SOLUTION

### STEP 1: Clean Slate
```bash
pip uninstall -y transformers huggingface_hub tokenizers trl peft unsloth unsloth_zoo datasets pyarrow
pip cache purge
```

### STEP 2: Install Exact Working Versions
```bash
# PyTorch with CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Core ML libraries (force reinstall, no deps)
pip install transformers==4.51.3 --force-reinstall --no-deps
pip install huggingface_hub==0.34.4 --force-reinstall --no-deps
pip install tokenizers==0.21.4 --force-reinstall --no-deps

# LoRA and training
pip install trl==0.18.0 peft==0.17.1

# Unsloth (from GitHub)
pip install git+https://github.com/unslothai/unsloth.git
pip install git+https://github.com/unslothai/unsloth-zoo.git
```

## ✅ WORKING VERSION COMBINATION
- torch==2.6.0
- transformers==4.51.3
- huggingface_hub==0.34.4
- tokenizers==0.21.4
- trl==0.18.0
- peft==0.17.1
- unsloth (latest from GitHub)
- unsloth_zoo (latest from GitHub)

## 🎯 WHY THIS WORKS
- --force-reinstall --no-deps prevents dependency conflicts
- Exact version pinning ensures compatibility
- GitHub packages get latest features
- CUDA 11.8 is stable for SageMaker T4

## �� USAGE
Run this fix BEFORE starting training to avoid dependency hell!

## 📝 NOTES
- This was discovered after multiple failed attempts
- The fix ensures stable training environment
- Always clear pip cache before reinstalling
- Test imports after each major package installation
"""

# Write to file in current directory
with open('./DEPENDENCY_FIX.md', 'w') as f:
    f.write(dependency_fix_content)

print("✅ DEPENDENCY_FIX.md created in current folder!")
print("📁 File location:", os.path.abspath('./DEPENDENCY_FIX.md'))

# Verify file was created
if os.path.exists('./DEPENDENCY_FIX.md'):
    size = os.path.getsize('./DEPENDENCY_FIX.md')
    print(f"📏 File size: {size} bytes")
    print("🎯 Ready to download to your laptop!")
else:
    print(" File creation failed!")