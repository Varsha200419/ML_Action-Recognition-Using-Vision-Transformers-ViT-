# Action Recognition with Vision Transformer (ViT)

**Coursework Implementation - Video Action Recognition using TimeSFormer**

This project implements video action recognition using Vision Transformer (ViT) adapted for video data through TimeSFormer architecture. The system recognizes 25 different actions from the HMDB dataset.

## 🎯 Coursework Requirements Met

- ✅ **Pre-trained TimeSFormer** from `facebook/timesformer-base-finetuned-k400`
- ✅ **HMDB Dataset** with 1,250 videos (50 per category)
- ✅ **8×224×224 clips** with 1/32 sampling rate
- ✅ **80/20 train/validation split**
- ✅ **TensorBoard integration** for training progress visualization
- ✅ **Top-1 and Top-5 accuracy** calculation
- ✅ **Confusion matrix** visualization and analysis
- ✅ **Top-5 predictions analysis** for sample videos
- ✅ **Hyperparameter tuning** experiments
- 🎯 **Target Performance**: 45% top-1, 75% top-5 accuracy

## 🚀 Quick Start

### 📦 Installation Options

#### Option 1: Standard Installation
```bash
pip install -r requirements.txt
```

#### Option 2: If Network Issues (Colab/Server)
```bash
# Use minimal requirements
pip install -r requirements_minimal.txt

# Or use robust installer
python install_dependencies.py
```

#### Option 3: Manual Core Installation
```bash
# Essential packages only
pip install torch torchvision transformers tensorboard numpy matplotlib Pillow tqdm PyYAML
```

### 🏃‍♂️ Running the Project

#### For Google Colab:
```python
!python train_colab.py
```

#### For Local Development:
```bash
python main.py
```

### 🔧 Troubleshooting Installation

**Network Connection Issues:**
- Use `requirements_minimal.txt` for limited connectivity
- Run `python install_dependencies.py` for robust installation with retries
- Install core packages manually if needed

**Memory Issues:**
- Reduce batch size in `config_colab.yaml`
- Use minimal requirements to save space

**Import Errors:**
- Check Python version (3.8+ required)
- Verify virtual environment activation
- Try installing packages individually

## 📊 TensorBoard Visualization

Monitor training progress in real-time:
```bash
tensorboard --logdir=logs/tensorboard
```

View metrics:
- Training/Validation Loss
- Top-1 and Top-5 Accuracy
- Learning Rate Schedules
- Model Comparisons

## 🏗️ Project Structure

```
Action_Recognition_ViT/
├── models/                    # Model architectures
│   ├── model_timesformer.py  # Pre-trained TimeSFormer (ViT for video)
│   ├── model_vit_separate.py # Standard ViT adaptation
├── src/                      # Core implementation
│   ├── train.py             # Training pipeline with TensorBoard
│   ├── dataloader.py        # HMDB dataset loader (1/32 sampling)
│   ├── evaluate.py          # Evaluation and analysis
│   └── utils.py             # Utility functions
├── config.yaml              # Training configuration
├── train_colab.py           # Colab-ready training script
└── requirements.txt         # Dependencies
```

## 📈 Training Features

- **Multi-Model Support**: TimeSFormer, ViT, CNN-LSTM
- **Experiment Management**: Optimizer, learning rate, batch size tuning
- **Real-time Monitoring**: TensorBoard integration
- **Comprehensive Analysis**: Top-5 predictions, confusion matrices
- **Progress Tracking**: Coursework target monitoring (45%/75%)

## 📁 Results and Output Locations

### 🎯 Where Your Results Are Saved

After training completion, all results are organized in these directories:

```
📊 RESULTS DIRECTORY STRUCTURE:
├── results/
│   ├── confusion_matrices/              # 🎨 Confusion matrix visualizations
│   │   ├── timesformer_confusion_matrix.png
│   │   ├── vit_confusion_matrix.png
│   │   ├── cnn_lstm_confusion_matrix.png
│   │   └── optimizer_adam_confusion_matrix.png
│   ├── training_logs/                   # 📝 Detailed training progress
│   │   ├── timesformer_training.log
│   │   ├── vit_training.log
│   │   └── cnn_lstm_training.log
│   ├── experiment_results.json         # 📋 Complete experiment data
│   ├── comprehensive_analysis.png      # 🔍 Multi-model comparison
│   └── experiment_report.txt           # 📄 Human-readable summary
│
├── models/                             # 💾 Trained model weights
│   ├── timesformer_best.pth           # 🏆 Best TimeSFormer model
│   ├── vit_best.pth                   # 🏆 Best ViT model
│   └── cnn_lstm_best.pth              # 🏆 Best CNN-LSTM model
│
├── logs/
│   ├── tensorboard/                    # 📈 TensorBoard files
│   │   ├── timesformer_experiment/     # TimeSFormer training logs
│   │   ├── vit_experiment/             # ViT training logs
│   │   └── cnn_lstm_experiment/        # CNN-LSTM training logs
│   └── training_logs/                  # 📊 Text-based logs
│       ├── training_progress.log
│       └── experiment_summary.log
```

### 🔍 Key Files for Coursework Submission

| **File** | **Purpose** | **Location** |
|----------|-------------|--------------|
| 🎯 **Main Results** | TimeSFormer confusion matrix | `results/confusion_matrices/timesformer_confusion_matrix.png` |
| 📊 **Model Comparison** | All models side-by-side | `results/comprehensive_analysis.png` |
| 📝 **Performance Summary** | Text-based results | `results/experiment_report.txt` |
| 📈 **Training Progress** | TensorBoard visualization | `logs/tensorboard/timesformer_experiment/` |
| 💾 **Best Model** | Trained model weights | `models/timesformer_best.pth` |
| 📋 **All Metrics** | JSON data format | `results/experiment_results.json` |

### 💡 How to Access Results

**View Training Summary:**
```bash
# Text summary
cat results/experiment_report.txt

# JSON results  
python -c "import json; print(json.dumps(json.load(open('results/experiment_results.json')), indent=2))"
```

**View Confusion Matrices:**
```bash
# Open images (Windows)
start results/confusion_matrices/timesformer_confusion_matrix.png
start results/comprehensive_analysis.png
```

**Launch TensorBoard:**
```bash
tensorboard --logdir logs/tensorboard
# Open: http://localhost:6006
```

**Load Trained Models:**
```python
from models.model_timesformer import TimeSFormerActionRecognition
model, checkpoint = TimeSFormerActionRecognition.load_model('models/timesformer_best.pth')
print(f"Best accuracy: {checkpoint['metrics']['best_val_accuracy']:.2f}%")
```

## 🔍 Analysis Outputs

1. **Confusion Matrices**: For each model and experiment
2. **Top-5 Analysis**: Sample predictions with confidence scores
3. **Training Curves**: Loss and accuracy progression
4. **Model Comparison**: Performance across different configurations
5. **TensorBoard Logs**: Interactive training visualization

## 🎥 Video Processing

- **Input Format**: 8 frames × 224×224 pixels
- **Sampling Rate**: 1/32 (TimeSFormer specification)
- **Preprocessing**: Frame padding, temporal reversal, flickering
- **Data Augmentation**: Random horizontal flip, color jitter

## 📋 Usage Examples

### Run Complete Training:
```python
trainer = ActionRecognitionTrainer("config.yaml")
trainer.run_experiments()
```

### Analyze Top-5 Predictions:
```python
trainer.analyze_top5_predictions(model, val_loader, num_samples=10)
```

### View Results:
- Check `results/` for confusion matrices and analysis
- Open TensorBoard for interactive training visualization
- Review `logs/` for detailed training logs

## 🏆 Performance Targets

- **Minimum Top-1 Accuracy**: 45%
- **Minimum Top-5 Accuracy**: 75%
- **Model**: Pre-trained TimeSFormer adapted for HMDB
- **Dataset**: 25 action categories, 80/20 split
