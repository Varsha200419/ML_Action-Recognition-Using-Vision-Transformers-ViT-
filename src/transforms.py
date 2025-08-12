import random
import numpy as np
import cv2

# Data transformations and augmentations
# Add your augmentation functions here

def preprocess_and_augment(frames, crop_size=(224,224), target_len=8):
    # Resize frames
    frames = [cv2.resize(frame, crop_size) if frame.shape[:2] != crop_size else frame for frame in frames]
    # Random horizontal flip
    if random.random() > 0.5:
        frames = [np.fliplr(frame) for frame in frames]
    # Pad if not enough frames
    while len(frames) < target_len:
        frames.append(frames[-1].copy())
    frames = frames[:target_len]
    # Normalize
    frames = [frame / 255.0 if frame.max() > 1 else frame for frame in frames]
    return frames
