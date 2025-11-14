# LLaMA 3.2 7B Triage Classification Training

## 🚀 Enhanced Training for LLaMA 3.2 7B Model

This repository contains optimized training scripts for fine-tuning LLaMA 3.2 7B on emergency department triage classification tasks.

## 📋 Features

### 🎯 Model Specifications
- **Base Model**: LLaMA 3.2 7B (unsloth/llama-3.2-7b-bnb-4bit)
- **Quantization**: 4-bit quantization for memory efficiency
- **LoRA Configuration**: r=16, alpha=32 (optimized for 7B)
- **Sequence Length**: 2048 tokens
- **Training Method**: Supervised Fine-Tuning (SFT)

### 🚀 Advanced Optimizations
- **Gradient Checkpointing**: Enabled for memory efficiency
- **CPU Offloading**: Automatic device mapping for large models
- **Memory Monitoring**: Real-time GPU and CPU memory tracking
- **Early Stopping**: Prevents overfitting with configurable patience
- **Data Augmentation**: Enhanced training data with variations
- **Validation Split**: 10% validation set for monitoring
- **Continuous Training**: Resume from any checkpoint

### 💾 Memory Management
- **GPU Memory Fraction**: 85% utilization
- **Batch Size**: 1 (with gradient accumulation)
- **Gradient Accumulation**: 32 steps for effective batch size
- **Chunk Processing**: 2000 samples per chunk
- **Automatic Cleanup**: Memory clearing between chunks

## 📁 File Structure

```
train_and_setup/
├── sagemaker_llama_7b_training.py    # Main 7B training script
├── sagemaker_7b_setup.py             # Setup script for 7B
├── requirements_7b.txt               # Dependencies for 7B
├── esi_data.csv                      # Your triage dataset
└── README_7B.md                      # This file
```

## 🛠️ Setup Instructions

### 1. SageMaker Environment Setup

```bash
# Run the setup script
python sagemaker_7b_setup.py
```

### 2. Upload Your Data

Make sure your `esi_data.csv` file is in the same directory as the training script.

### 3. Start Training

```bash
# Run the 7B training script
python sagemaker_llama_7b_training.py
```

## ⚙️ Configuration

### Key Parameters for 7B Model

```python
CONFIG = {
    'BASE_MODEL_NAME': "unsloth/llama-3.2-7b-bnb-4bit",
    'CHUNK_SIZE': 2000,                    # Smaller chunks for 7B
    'BATCH_SIZE': 1,                       # Reduced batch size
    'LEARNING_RATE': 5e-5,                # Lower learning rate
    'GRADIENT_ACCUMULATION_STEPS': 32,     # Increased for effective batch
    'GRADIENT_CHECKPOINTING': True,        # Enabled for memory efficiency
    'CPU_OFFLOAD': True,                   # Enable CPU offloading
}
```

### LoRA Configuration

```python
# Optimized LoRA settings for 7B
r=16,                    # Reduced rank for memory efficiency
alpha=32,                # Reduced alpha
lora_dropout=0.1,        # Reduced dropout
bias="none",             # No bias training
use_rslora=False,        # Disabled for stability
```

## 📊 Training Process

### 1. Data Processing
- **Validation**: Remove missing values
- **Augmentation**: Create variations for better generalization
- **Formatting**: Enhanced prompts with medical context
- **Splitting**: 90% train, 10% validation

### 2. Model Loading
- **Base Model**: Load LLaMA 3.2 7B with 4-bit quantization
- **LoRA Adapters**: Apply optimized LoRA configuration
- **Device Mapping**: Automatic CPU/GPU distribution

### 3. Training Loop
- **Chunk Processing**: Process data in 2000-sample chunks
- **Memory Monitoring**: Track GPU and CPU usage
- **Checkpointing**: Save every 200 steps
- **Early Stopping**: Stop if no improvement for 2 epochs

### 4. Model Saving
- **Local Checkpoints**: Save to `./llama_7b_checkpoints/`
- **HuggingFace Upload**: Automatic upload after each chunk
- **Metadata**: Save training metrics and configuration

## 🔍 Monitoring

### Memory Usage
- **GPU Memory**: Monitored every 20 steps
- **CPU Memory**: Tracked continuously
- **Automatic Cleanup**: Memory clearing between chunks

### Training Metrics
- **Loss**: Training and validation loss
- **Progress**: Percentage completion
- **Time**: Estimated time remaining

### Logs
- **File**: `./llama_7b_checkpoints/training.log`
- **Console**: Real-time progress updates
- **Errors**: Detailed error reporting

## ⚠️ Important Notes

### Memory Requirements
- **Minimum GPU Memory**: 15GB recommended
- **CPU Memory**: 32GB+ recommended
- **Storage**: 50GB+ for checkpoints

### Training Time
- **Per Chunk**: ~2-4 hours (depending on hardware)
- **Full Dataset**: ~20-40 hours for 50K samples
- **Resume**: Can resume from any checkpoint

### Cost Considerations
- **SageMaker**: Higher cost than 3B model
- **GPU Hours**: ~2x more than 3B model
- **Storage**: More checkpoints and larger model

## 🚀 Deployment

### HuggingFace Model
- **Repository**: `ali009eng/llama-7b-triage-classifier`
- **Access**: Public repository
- **Versioning**: Automatic version control

### Local Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3.2-7b-bnb-4bit")
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3.2-7b-bnb-4bit")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "ali009eng/llama-7b-triage-classifier")
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `CHUNK_SIZE` to 1000
   - Enable `CPU_OFFLOAD`
   - Reduce `BATCH_SIZE` to 1

2. **Slow Training**
   - Check GPU utilization
   - Verify CUDA installation
   - Monitor memory usage

3. **Dependency Conflicts**
   - Use the provided `requirements_7b.txt`
   - Run setup script again
   - Clear pip cache

### Performance Tips

1. **Memory Optimization**
   - Use gradient checkpointing
   - Enable CPU offloading
   - Monitor memory usage

2. **Training Speed**
   - Use larger GPU instances
   - Optimize data loading
   - Use mixed precision training

## 📈 Expected Results

### Performance Metrics
- **Accuracy**: 85-95% on triage classification
- **Training Time**: 20-40 hours for full dataset
- **Model Size**: ~4GB (4-bit quantized)

### Quality Improvements
- **Better Reasoning**: 7B model shows improved reasoning
- **Medical Knowledge**: Enhanced medical domain understanding
- **Consistency**: More consistent triage decisions

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in `./llama_7b_checkpoints/training.log`
3. Monitor memory usage during training

## 📝 License

This project is for educational and research purposes. Please ensure compliance with LLaMA model license terms.
