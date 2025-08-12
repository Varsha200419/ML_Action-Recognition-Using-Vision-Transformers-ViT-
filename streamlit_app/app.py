import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Try to import custom models with error handling
try:
    from models.vit_model import get_timesformer_model
    MODEL_AVAILABLE = True
except ImportError as e:
    st.error(f"Cannot import model: {e}")
    MODEL_AVAILABLE = False

# Try to import transformers
try:
    from transformers import AutoFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.error("transformers library not available")
    TRANSFORMERS_AVAILABLE = False

# HMDB-51 action classes
HMDB_CATEGORIES = [
    'brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 
    'dive', 'draw_sword', 'dribble', 'drink', 'eat', 'fall_floor', 'fencing', 
    'flic_flac', 'golf', 'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball', 
    'kiss', 'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup', 
    'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow', 
    'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault', 'stand', 
    'swing_baseball', 'sword', 'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave'
]

@st.cache_resource
def load_model(model_path):
    """Load the trained Timesformer model"""
    if not MODEL_AVAILABLE:
        st.error("Model loading not available - missing dependencies")
        return None, None, None, None
        
    if not TRANSFORMERS_AVAILABLE:
        st.error("Transformers library not available")
        return None, None, None, None
        
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'classifier.weight' in state_dict:
                num_classes = state_dict['classifier.weight'].shape[0]
            else:
                num_classes = 25
        else:
            num_classes = 25
        
        model = get_timesformer_model(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        try:
            from transformers import VideoMAEImageProcessor
            extractor = VideoMAEImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        except:
            extractor = AutoFeatureExtractor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        
        return model, extractor, device, num_classes
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def extract_frames(video_path, num_frames=8, method='uniform'):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) + [total_frames-1] * (num_frames - total_frames)
    else:
        if method == 'uniform':
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        elif method == 'middle_focused':
            start_frame = int(total_frames * 0.1)
            end_frame = int(total_frames * 0.9)
            frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_resized)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames), frame_indices

def predict_action(model, extractor, frames, device, categories):
    try:
        frame_list = [frames[i] for i in range(len(frames))]
        inputs = extractor(frame_list, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            top10_probs, top10_indices = torch.topk(probabilities[0], min(10, len(categories)))
            top10_predictions = [(categories[idx.item()], prob.item()) for prob, idx in zip(top10_probs, top10_indices)]
            all_probabilities = probabilities[0].cpu().numpy()
        
        return predicted_class_idx, confidence, top10_predictions, all_probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, None

def create_frame_grid(frames, max_frames=8):
    num_frames = min(len(frames), max_frames)
    cols = 4
    rows = (num_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        if i < num_frames:
            axes[row, col].imshow(frames[i])
            axes[row, col].set_title(f'Frame {i+1}')
        else:
            axes[row, col].axis('off')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="Action Recognition with Timesformer", page_icon="🎬", layout="wide")
    st.title("🎬 Video Action Recognition")
    st.markdown("*Powered by Timesformer Vision Transformer*")
    
    if not MODEL_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        st.error("Missing dependencies. Please install required packages.")
        st.stop()
    
    with st.sidebar:
        st.header("Model Information")
        st.info("This app uses a fine-tuned Timesformer model for video action recognition.")
        default_model_path = "Results/lr - 0.001/SGD/model_final.pth"
        model_path = st.text_input("Model Path:", value=default_model_path)
        
        if st.button("Load Model"):
            if os.path.exists(model_path):
                with st.spinner("Loading model..."):
                    model, extractor, device, num_classes = load_model(model_path)
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.extractor = extractor
                        st.session_state.device = device
                        st.session_state.num_classes = num_classes
                        st.success(f"Model loaded successfully! Classes: {num_classes}")
                    else:
                        st.error("Failed to load model")
            else:
                st.error("Model file not found!")
    
    if 'model' not in st.session_state:
        st.warning("Please load a model first using the sidebar.")
        return
    
    st.header("Upload Video for Action Recognition")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.video(uploaded_file)
        frames, frame_indices = extract_frames(video_path, num_frames=8, method='uniform')
        st.info(f"Selected frame indices: {frame_indices.tolist()}")
        st.pyplot(create_frame_grid(frames))
        
        if st.button("🔍 Analyze Action", type="primary"):
            with st.spinner("Analyzing video..."):
                categories = HMDB_CATEGORIES[:st.session_state.num_classes]
                pred_idx, confidence, top10_predictions, all_probs = predict_action(
                    st.session_state.model, st.session_state.extractor,
                    frames, st.session_state.device, categories
                )
                if pred_idx is not None:
                    st.success(f"Prediction: {categories[pred_idx].replace('_',' ').title()} ({confidence:.2%})")
    
        try:
            os.unlink(video_path)
        except:
            pass

if __name__ == "__main__":
    main()
