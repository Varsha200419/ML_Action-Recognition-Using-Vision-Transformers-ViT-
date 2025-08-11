# frame_preprocess.py
# Script to sample and pad frames for each video folder

import os
import numpy as np
from PIL import Image, ImageEnhance
import random
import yaml

def normalize_frames(frames):
    """Normalize pixel values to range [0, 1]."""
    return [frame / 255.0 for frame in frames]

def random_flip(frames):
    """Randomly flip frames horizontally."""
    if random.random() > 0.5:
        return [np.fliplr(frame) for frame in frames]
    return frames

def random_crop(frames, crop_size=200):
    """Randomly crop frames to a smaller size."""
    cropped_frames = []
    for frame in frames:
        # Convert to uint8 if normalized
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)
            
        h, w, _ = frame_uint8.shape
        if h < crop_size or w < crop_size:
            # If frame is smaller than crop size, just resize
            cropped_frame = np.array(Image.fromarray(frame_uint8).resize((224, 224)))
        else:
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            cropped_frame = frame_uint8[top:top+crop_size, left:left+crop_size]
            cropped_frame = np.array(Image.fromarray(cropped_frame).resize((224, 224)))
        cropped_frames.append(cropped_frame)
    return cropped_frames

def to_grayscale(frames):
    """Convert frames to grayscale."""
    gray_frames = []
    for frame in frames:
        # Convert to uint8 if normalized
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)
            
        gray_img = Image.fromarray(frame_uint8).convert('L')
        gray_frames.append(np.array(gray_img))
    return gray_frames

def adjust_brightness(frames, factor=1.2):
    """Randomly adjust brightness of frames."""
    bright_frames = []
    for frame in frames:
        # Convert to uint8 if normalized
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)
            
        img = Image.fromarray(frame_uint8)
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor if random.random() > 0.5 else 1.0)
        bright_frames.append(np.array(bright_img))
    return bright_frames

def temporal_reversal(frames):
    """Reverse the order of frames."""
    if random.random() > 0.5:
        return frames[::-1]
    return frames

def frame_averaging(frames):
    """Average adjacent frames to create smoother transitions."""
    if len(frames) < 2:
        return frames
        
    avg_frames = []
    for i in range(len(frames)):
        if i == 0:
            avg_frames.append(frames[i])
        else:
            # Average frames with proper data type handling
            frame1 = frames[i-1].astype(np.float64)
            frame2 = frames[i].astype(np.float64)
            avg_frame = (frame1 + frame2) / 2
            
            # Maintain original data type
            if frames[i].max() <= 1.0:
                avg_frame = avg_frame.astype(np.float64)
            else:
                avg_frame = avg_frame.astype(np.uint8)
                
            avg_frames.append(avg_frame)
    return avg_frames

def flicker_frames(frames, max_delta=40):
    """Simulate flickering by randomly adjusting brightness of frames."""
    flickered = []
    for frame in frames:
        delta = random.randint(-max_delta, max_delta)
        
        # Handle different data types
        if frame.max() <= 1.0:  # Normalized frames
            # Convert delta to normalized range
            delta_norm = delta / 255.0
            frame = np.clip(frame + delta_norm, 0, 1).astype(np.float64)
        else:  # Regular uint8 frames
            frame = np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            
        flickered.append(frame)
    return flickered

def pad_frames(frames, num_frames):
    """Pad frames by duplicating and augmenting existing frames if not enough."""
    while len(frames) < num_frames:
        # Select a random frame and apply simple augmentations
        base_frame = random.choice(frames).copy()
        
        # Apply simple augmentations that don't change dimensions
        if random.random() > 0.5:
            base_frame = np.fliplr(base_frame)  # horizontal flip
        
        # Random brightness adjustment
        if random.random() > 0.5 and base_frame.max() <= 1.0:
            base_frame = np.clip(base_frame * random.uniform(0.8, 1.2), 0, 1)
        elif random.random() > 0.5:
            base_frame = np.clip(base_frame * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
            
        frames.append(base_frame)
    return frames

def process_video_frames(video_folder, num_frames=8, sampling_rate=32, frame_size=224, save_dir=None):
    # List all jpg frames in order
    frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
    total_frames = len(frame_files)
    indices = [i * sampling_rate for i in range(num_frames)]
    frames = []
    for idx in indices:
        actual_idx = min(idx, total_frames - 1)
        frame_path = os.path.join(video_folder, frame_files[actual_idx]) if total_frames > 0 else None
        if frame_path and os.path.exists(frame_path):
            img = Image.open(frame_path).resize((frame_size, frame_size))
            frames.append(np.array(img))
        else:
            # Pad with zeros if no frame exists
            frames.append(np.zeros((frame_size, frame_size, 3), dtype=np.uint8))
    
    # Apply preprocessing methods (keep them as RGB for consistency)
    frames = normalize_frames(frames)
    frames = random_flip(frames)
    frames = random_crop(frames, crop_size=200)
    frames = adjust_brightness(frames, factor=1.2)
    frames = temporal_reversal(frames)
    frames = frame_averaging(frames)
    frames = flicker_frames(frames)
    frames = pad_frames(frames, num_frames)

    # Optionally save processed frames
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            # Ensure frame is uint8 for saving
            if frame.max() <= 1.0:
                frame_to_save = (frame * 255).astype(np.uint8)
            else:
                frame_to_save = frame.astype(np.uint8)
            out_path = os.path.join(save_dir, f'frame_{i:02d}.jpg')
            Image.fromarray(frame_to_save).save(out_path)
    return frames

def get_dataset_config():
    with open("configs/coursework_config.yaml") as f:
        config = yaml.safe_load(f)
    return config['dataset']['root_dir'], config['dataset']['categories']

# Example usage: process all video folders in a dataset
if __name__ == "__main__":
    dataset_root, categories = get_dataset_config()
    output_root = os.path.join("results", "HMDB_simp_processed")  # Save in results folder
    num_frames = 8
    sampling_rate = 32
    frame_size = 224
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    for category in categories:
        cat_path = os.path.join(dataset_root, category)
        if os.path.isdir(cat_path):
            print(f"Processing category: {category}")
            for video_folder in os.listdir(cat_path):
                video_path = os.path.join(cat_path, video_folder)
                if os.path.isdir(video_path):
                    save_dir = os.path.join(output_root, category, video_folder)
                    process_video_frames(video_path, num_frames, sampling_rate, frame_size, save_dir)
    print("Frame preprocessing complete. All processed frames saved in results folder.")
