# 🚀 LLaMA 3B Fine-Tuning on AWS SageMaker

**Medical Triage Classification Model with Enhanced Training**

## ✨ What You Get

- **Enhanced LLaMA 3B model** trained on your medical triage data
- **Advanced LoRA configuration** (r=32, alpha=64) for better performance
- **Data augmentation** for improved generalization
- **Early stopping** to prevent overfitting
- **Validation split** for better model evaluation
- **Automatic HuggingFace upload** after each training chunk
- **Continuous training** until all data is processed

## 🎯 Prerequisites

### 1. AWS SageMaker Access
- ✅ **ml.g4dn.xlarge instance** approved (you have this!)
- ✅ **$200 AWS credits** available
- ✅ **Student account** benefits

### 2. Your Data
- ✅ **esi_data.csv** - Your 50,002 row triage dataset
- ✅ **HuggingFace token** - For model uploads

## 🚀 Quick Start on SageMaker

### Step 1: Create Notebook Instance
1. Go to **AWS SageMaker Console**
2. Click **"Create notebook instance"**
3. **Name:** `llama-triage-training`
4. **Instance type:** `ml.g4dn.xlarge` (your approved GPU!)
5. **Volume size:** `20 GB`
6. Click **"Create notebook instance"**

### Step 2: Wait for Launch
- **Status:** "InService" (takes 5-10 minutes)
- **Green checkmark** = ready to use!

### Step 3: Open Jupyter
1. Click **"Open Jupyter"**
2. Click **"New"** → **"Python 3"**
3. Name it: `llama_training.ipynb`

### Step 4: Upload Your Data
1. Click **upload button** (📁 icon)
2. Select your `esi_data.csv` file
3. Wait for upload to complete

### Step 5: Run Training
1. **Copy the training script** from `sagemaker_llama_training.py`
2. **Paste into a cell** in your notebook
3. **Run the cell** to start training!

## 📁 Files You Need

### 1. **`sagemaker_llama_training.py`** (Main Script)
- ✅ **SageMaker-ready** paths (`./` instead of `/dbfs/`)
- ✅ **No Google Colab** dependencies
- ✅ **All your training logic** preserved
- ✅ **Enhanced LoRA** configuration

### 2. **`sagemaker_setup.py`** (Optional Setup)
- Quick environment verification
- Dependency installation
- GPU and dataset checks

### 3. **`requirements_sagemaker.txt`** (Dependencies)
- All required packages
- Compatible versions

## 🔧 Configuration

### Your Current Settings:
```python
CONFIG = {
    'BASE_MODEL_NAME': "unsloth/llama-3.2-3b-bnb-4bit",
    'HF_MODEL_NAME': "ali009eng/llama-3b-triage-classifier",
    'CHUNK_SIZE': 3000,        # Process 3000 rows at a time
    'BATCH_SIZE': 2,           # Stable training
    'LEARNING_RATE': 1e-4,     # Good convergence
    'NUM_EPOCHS_PER_CHUNK': 3, # Train each chunk for 3 epochs
}
```

### Training Process:
1. **Load 3000 rows** from your dataset
2. **Train for 3 epochs** on that chunk
3. **Save and upload** model to HuggingFace
4. **Move to next 3000 rows**
5. **Repeat until all 50,002 rows** are processed

## 💰 Cost Estimation

### ml.g4dn.xlarge (Your Instance):
- **Hourly rate:** ~$0.50-1.00/hour
- **70 hours training:** $35-70
- **Your credits:** $200
- **Remaining:** $130-165 ✅

## 📊 What Happens During Training

### Real-Time Monitoring:
- **Memory usage** updates
- **Training loss** per step
- **Chunk progress** indicators
- **GPU utilization** status
- **Validation metrics**

### Automatic Saves:
- **Checkpoint** after each chunk
- **HuggingFace upload** after each chunk
- **Progress tracking** for resuming

## 🎉 Expected Results

### After Full Training:
- **Enhanced LLaMA 3B model** specialized for medical triage
- **High accuracy** on ESI level classification
- **Medical context awareness** from enhanced prompts
- **Robust performance** from data augmentation
- **Ready for LangChain integration**

## 🚨 Troubleshooting

### Common Issues:

#### 1. **"No GPU detected"**
- Check instance type is `ml.g4dn.xlarge`
- Restart notebook instance

#### 2. **"Dataset not found"**
- Upload `esi_data.csv` to notebook
- Check file path is correct

#### 3. **"Out of memory"**
- Reduce `BATCH_SIZE` to 1
- Reduce `CHUNK_SIZE` to 2000

#### 4. **"Import errors"**
- Run setup script first
- Check all dependencies installed

## 🔄 Resuming Training

### If Training Interrupts:
1. **Check last processed row** in logs
2. **Restart notebook instance**
3. **Run training script again**
4. **Automatically resumes** from last checkpoint

## 📈 Next Steps After Training

### 1. **Test Your Model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load your fine-tuned model
model = PeftModel.from_pretrained(
    base_model, 
    "ali009eng/llama-3b-triage-classifier"
)
```

### 2. **LangChain Integration:**
```python
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool

# Create your medical triage agent
```

### 3. **Deploy:**
- **HuggingFace Spaces** for demo
- **AWS Lambda** for API
- **Local server** for testing

## 🎯 Success Metrics

### Training Completion:
- ✅ **All 50,002 rows processed**
- ✅ **Model uploaded to HuggingFace**
- ✅ **Training logs saved**
- ✅ **Checkpoints available**

### Model Quality:
- **Low validation loss**
- **Consistent triage predictions**
- **Medical context understanding**
- **Robust to data variations**

## 🚀 Ready to Start?

**Your SageMaker environment is ready!**

**Just upload your data and run the training script!**

**You'll have a professional medical triage AI in no time!** 🎉

---

**Need help? Check the logs above or restart the notebook instance!**

