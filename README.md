# Action Recognition using Vision Transformer (ViT)

This project implements video action recognition using state-of-the-art Vision Transformer (ViT) and TimeSFormer models, trained on a simplified HMDB-51 dataset (`HMDB_simp`) with 25 action categories. The goal is to accurately classify human actions in short video clips using transformer-based deep learning architectures.

---

🚀 **Features**
- Multiple model architectures: ViT, TimeSFormer
- Configurable training pipeline (hyperparameters, optimizers, learning rates)
- Evaluation metrics: accuracy, confusion matrix
- Visualization tools: training/validation loss curves
- Streamlit-based interactive web application for inference and visualization

---

📁 **Project Structure**
```
├── configs/           # Model and training configuration files
├── models/            # Pretrained and trained model weights
├── src/               # Source code (training, evaluation, data processing)
├── Results/           # training_log, .pth and visualizations
├── result/            # Images (confusion matrix, loss curves)
├── app.py          # Streamlit web application
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

🛠️ **Installation**

**Prerequisites:**
- Python 3.8+
- PyTorch 1.9+
- GPU recommended (CUDA)

**Setup:**
```powershell
# Clone the repository
git clone https://github.com/Varsha200419/Action_Recognition_ViT.git
cd Action_Recognition_ViT

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install streamlit opencv-python
```

---

📊 **Dataset**

- **HMDB-51 (Simplified):**
	- 25 action categories (e.g., walk, run, jump, wave, etc.)
	- Preprocessed and split for training/validation/testing

---

🏋️ **Training**

Train a model using the following command:
```powershell
python src/train.py timesformer --model vit --optimizer adam --lr 0.001
```
- Options: `--model` (vit, timesformer), `--optimizer` (adam, sgd), `--lr` (learning rate)

---

🔍 **Evaluation**

Evaluate a trained model:
```powershell
python src/evaluate.py timesformer --model-path models/best_model.pth
```
- Outputs accuracy, confusion matrix, and saves visualizations to `Results/` and `assets/`

---



---

🌐 **Web Application**

Launch the interactive web app for real-time inference and visualization:
```powershell
streamlit run webapp.py
```
- Upload video clips and get action predictions
- View model performance metrics and visualizations

---

📊 **Quick Start Guide**
1. **Install dependencies**
2. **Train a model**
3. **Evaluate results**
4. **Launch the web app**

---

📚 **References**
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [TimeSFormer](https://arxiv.org/abs/2102.05095)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

---

👨‍💻 **Author**
- Varsha200419

---

## License

This project is licensed under the MIT License.
