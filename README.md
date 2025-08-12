# 🎥 Video Action Recognition using Vision Transformer (ViT & TimeSFormer)

This project implements video action recognition on the HMDB_simp dataset using state-of-the-art Vision Transformer (ViT) and TimeSFormer models. The system is designed to classify videos into 25 distinct human action categories, providing a robust and configurable pipeline for training, evaluation, and visualization.

---

## 🎯 Overview

- **Purpose:** Automatically recognize human actions in video clips using transformer-based deep learning models.
- **Models Used:**  
  - **Vision Transformer (ViT)**
  - **TimeSFormer**  
- **Action Categories:** 25 classes (e.g., brush_hair, cartwheel, chew, climb, fencing, golf, pushup, ride_bike, shoot_bow, smile, throw, wave, etc.)

---

## 🚀 Features

- Multiple model architectures: ViT & TimeSFormer
- Configurable training pipeline via YAML config files
- Evaluation metrics: Top-1/Top-5 accuracy, confusion matrix
- Automatic saving of logs, model weights, and confusion matrices
- Visualization tools for loss curves and metrics
- Streamlit-based interactive web application for demo and inference

---

## 📁 Project Structure

```
configs/                # Configuration files (YAML)
models/                 # Model implementations
src/                    # Training, evaluation, dataset, transforms, utilities
Results/                # Experiment outputs (logs, weights, confusion matrices)
app.py                  # Streamlit web application
requirements.txt        # Dependencies
```

---

## 🛠️ Installation

**Prerequisites:**  
- Python 3.8+
- PyTorch 1.9+ (GPU recommended)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

**Setup:**
```bash
# Clone the repository
git clone https://github.com/Varsha200419/ML_Action-Recognition-Using-Vision-Transformers-ViT-.git
cd ML_Action-Recognition-Using-Vision-Transformers-ViT-

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install streamlit opencv-python
```

---

## 📊 Dataset

- **HMDB-51** (simplified as HMDB_simp):  
  - 25 action categories  
  - Preprocessed frames stored in `/Results/HMDB_simp_processed`

---

## 🏋️ Training

Train a model with configurable options (model, optimizer, learning rate, etc.):

```bash
python src/train.py --model facebook/timesformer-base-finetuned-k400
```
- Model options: `facebook/timesformer-base-finetuned-k400`, `google/vit-base-patch16-224-in21k`, etc.
- Configure hyperparameters in `configs/coursework_config.yaml`.

---

## 🔍 Evaluation

Evaluate a trained model and generate metrics:

```bash
python src/evaluate.py --model-path Results/lr - 0.001/model_final.pth
```
- Outputs: Accuracy, Top-5 accuracy, confusion matrix, classification report.

---

## 🌐 Web Application

Launch the interactive Streamlit web app for demo and inference:

```bash
streamlit run app.py
```
- Features: Upload video, run inference, view predicted action, visualize metrics.

---

## 📊 Quick Start Guide

1. **Install dependencies**  
2. **Train a model**  
3. **Evaluate results**  
4. **Launch the web app**

---

## 📚 References

- [TimeSFormer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
- [HMDB-51 Dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

## 👨‍💻 Author

Varsha200419

---

## 📝 License

This project is licensed under
