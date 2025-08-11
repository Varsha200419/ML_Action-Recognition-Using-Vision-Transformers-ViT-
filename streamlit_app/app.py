import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import sys
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
project_root = Path(_file_).parent
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
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get number of classes from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'classifier.weight' in state_dict:
                num_classes = state_dict['classifier.weight'].shape[0]
            else:
                num_classes = 25  # Default fallback
        else:
            num_classes = 25
        
        # Create model with correct number of classes
        model = get_timesformer_model(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load feature extractor for Timesformer
        try:
            from transformers import VideoMAEImageProcessor
            extractor = VideoMAEImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        except:
            # Fallback to AutoFeatureExtractor
            extractor = AutoFeatureExtractor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        
        return model, extractor, device, num_classes
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def extract_frames(video_path, num_frames=8, method='uniform'):
    """Extract frames from video with different sampling methods"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames < num_frames:
        # If video has fewer frames than needed, repeat last frame
        frame_indices = list(range(total_frames)) + [total_frames-1] * (num_frames - total_frames)
    else:
        if method == 'uniform':
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        elif method == 'middle_focused':
            # Focus more on middle part of video (where action typically occurs)
            start_frame = int(total_frames * 0.1)  # Skip first 10%
            end_frame = int(total_frames * 0.9)    # Skip last 10%
            frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        else:
            # Default to uniform
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to 224x224
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_resized)
        else:
            # Use last valid frame if reading fails
            if frames:
                frames.append(frames[-1])
            else:
                # Create black frame as fallback
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames), frame_indices

def predict_action(model, extractor, frames, device, categories):
    """Predict action from video frames"""
    try:
        # Prepare input for Timesformer model
        # frames shape: (num_frames, height, width, channels)
        
        # Convert frames to list format expected by the processor
        frame_list = [frames[i] for i in range(len(frames))]
        
        # Process frames using the video processor
        # The processor expects videos as a list of frames
        inputs = extractor(frame_list, return_tensors="pt")
        
        # Move inputs to device
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            # Forward pass through the Timesformer model
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predictions
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            
            # Get top 10 predictions for better analysis
            top10_probs, top10_indices = torch.topk(probabilities[0], min(10, len(categories)))
            top10_predictions = []
            for i, (prob, idx) in enumerate(zip(top10_probs, top10_indices)):
                if idx.item() < len(categories):
                    action_name = categories[idx.item()]
                    top10_predictions.append((action_name, prob.item()))
            
            # Also get all probabilities for analysis
            all_probabilities = probabilities[0].cpu().numpy()
        
        return predicted_class_idx, confidence, top10_predictions, all_probabilities
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        # Print more detailed error info for debugging
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None, None

def create_frame_grid(frames, max_frames=8):
    """Create a grid of frames for visualization"""
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
    st.set_page_config(
        page_title="Action Recognition with Timesformer",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Video Action Recognition")
    st.markdown("*Powered by Timesformer Vision Transformer*")
    
    # Check dependencies
    if not MODEL_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        st.error("Missing dependencies. Please install required packages.")
        st.stop()
        return
    
    # Sidebar for model information
    with st.sidebar:
        st.header("Model Information")
        st.info("This app uses a fine-tuned Timesformer model for video action recognition.")
        
        # Model path input
        default_model_path = "Results/lr - 0.001/SGD/timesformer_lr000100_f8_s32_20250805_134346.pth"
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
    
    # Main content area
    if 'model' not in st.session_state:
        st.warning("Please load a model first using the sidebar.")
        return
    
    st.header("Upload Video for Action Recognition")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze the action"
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display video
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(uploaded_file)
        
        with col2:
            st.subheader("Video Information")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            st.write(f"*FPS:* {fps:.2f}")
            st.write(f"*Frames:* {frame_count}")
            st.write(f"*Duration:* {duration:.2f}s")
        
        # Extract and display frames
        st.subheader("Frame Extraction Settings")
        col1, col2 = st.columns(2)
        with col1:
            frame_method = st.selectbox(
                "Frame sampling method:",
                ["uniform", "middle_focused"],
                help="Uniform: evenly spaced frames, Middle-focused: focus on middle 80% of video"
            )
        with col2:
            num_frames = st.slider("Number of frames:", 4, 16, 8)
        
        st.subheader("Extracted Frames")
        with st.spinner("Extracting frames..."):
            frames, frame_indices = extract_frames(video_path, num_frames=num_frames, method=frame_method)
        
        # Show frame indices info
        st.info(f"Selected frame indices: {frame_indices.tolist()}")
        
        # Show frame grid
        fig = create_frame_grid(frames)
        st.pyplot(fig)
        
        # Predict action
        if st.button("🔍 Analyze Action", type="primary"):
            with st.spinner("Analyzing video..."):
                # Use appropriate categories based on model
                categories = HMDB_CATEGORIES[:st.session_state.num_classes]
                
                pred_idx, confidence, top10_predictions, all_probs = predict_action(
                    st.session_state.model, 
                    st.session_state.extractor,
                    frames,
                    st.session_state.device,
                    categories
                )
                
                if pred_idx is not None:
                    st.success("Analysis Complete!")
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("🎯 Predicted Action")
                        if pred_idx < len(categories):
                            predicted_action = categories[pred_idx]
                            st.write(f"{predicted_action.replace('_', ' ').title()}")
                            st.write(f"Confidence: *{confidence:.2%}*")
                        else:
                            st.error("Prediction index out of range")
                    
                    with col2:
                        st.subheader("📊 Top 10 Predictions")
                        for i, (action, prob) in enumerate(top10_predictions, 1):
                            st.write(f"{i}. *{action.replace('_', ' ').title()}*: {prob:.2%}")
                    
                    # Confidence visualization
                    st.subheader("📈 Confidence Distribution")
                    actions = [action.replace('_', ' ').title() for action, _ in top10_predictions]
                    probs = [prob for _, prob in top10_predictions]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(actions, probs, color=plt.cm.viridis(np.linspace(0, 1, len(actions))))
                    ax.set_xlabel('Confidence')
                    ax.set_title('Top 10 Action Predictions')
                    ax.set_xlim(0, 1)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probs):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{prob:.2%}', ha='left', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show pullup specific analysis if available
                    if 'pullup' in categories:
                        pullup_idx = categories.index('pullup')
                        pullup_confidence = all_probs[pullup_idx] * 100
                        st.info(f"*Pullup confidence*: {pullup_confidence:.2f}%")
                
                else:
                    st.error("Failed to analyze video")
        
        # Clean up temporary file
        try:
            os.unlink(video_path)
        except:
            pass
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit and Timesformer • Action Recognition System</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if _name_ == "_main_":
    main()