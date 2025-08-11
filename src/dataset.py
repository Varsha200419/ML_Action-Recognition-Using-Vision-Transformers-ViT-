# dataset.py
# Custom Dataset & DataLoader for HMDB

import os
import torch
from torch.utils.data import Dataset
import cv2
import random
import numpy as np
import yaml
from torchvision import transforms

class VideoAugmentation:
    """Enhanced video-specific data augmentation to combat overfitting"""
    def __init__(self, training=True):
        self.training = training
        self.brightness_range = 0.3
        self.contrast_range = 0.3
        self.saturation_range = 0.3
        self.noise_std = 0.05
        self.rotation_angle = 15
        self.crop_scale = 0.8
    
    def apply_augmentation(self, frames):
        """Apply more aggressive augmentation to video frames"""
        if not self.training:
            return frames
        
        # Random brightness/contrast/saturation (more aggressive)
        if random.random() < 0.7:  # Increased probability
            brightness_factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
            contrast_factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
            
            frames = frames * brightness_factor
            frames = (frames - 0.5) * contrast_factor + 0.5
            frames = torch.clamp(frames, 0, 1)
        
        # Random noise (more aggressive)
        if random.random() < 0.5:  # Increased probability
            noise = torch.randn_like(frames) * self.noise_std
            frames = torch.clamp(frames + noise, 0, 1)
        
        # Random horizontal flip
        if random.random() < 0.5:
            frames = torch.flip(frames, [3])  # Flip width dimension
            
        # Random rotation simulation (basic geometric transform)
        if random.random() < 0.3:
            # Simple channel swapping as a form of color augmentation
            if random.random() < 0.5:
                frames = frames[:, [2, 1, 0], :, :]  # RGB to BGR
        
        # Random temporal shifts (drop/duplicate frames)
        if random.random() < 0.3:
            num_frames = frames.shape[0]
            if num_frames > 2:
                # Randomly drop one frame and duplicate another
                drop_idx = random.randint(0, num_frames-1)
                dup_idx = random.randint(0, num_frames-1)
                while dup_idx == drop_idx:
                    dup_idx = random.randint(0, num_frames-1)
                
                frames[drop_idx] = frames[dup_idx]
        
        return frames

class HMDBDataset(Dataset):
    def __init__(self, root_dir=None, categories=None, num_frames=8, frame_size=224, sampling_rate=32, mode='train', use_processed=True):
        # Initialize augmentation
        self.augmentation = VideoAugmentation(training=(mode == 'train'))
        
        if root_dir is None or categories is None:
            with open("configs/coursework_config.yaml") as f:
                config = yaml.safe_load(f)
            original_root_dir = config['dataset']['root_dir']
            categories = config['dataset']['categories']
            
            # Check if processed data exists and use it if available
            if use_processed:
                processed_root_dir = os.path.join("results", "HMDB_simp_processed")
                if os.path.exists(processed_root_dir) and root_dir is None:
                    root_dir = processed_root_dir
                    print(f"Using processed dataset from: {root_dir}")
                else:
                    root_dir = original_root_dir
                    print(f"Processed dataset not found, using original: {root_dir}")
            else:
                root_dir = original_root_dir
                print(f"Using original dataset: {root_dir}")
        
        self.root_dir = root_dir
        self.categories = categories
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sampling_rate = sampling_rate
        self.mode = mode
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for idx, cat in enumerate(self.categories):
            cat_dir = os.path.join(self.root_dir, cat)
            if os.path.exists(cat_dir):
                for video_dir in os.listdir(cat_dir):
                    video_path = os.path.join(cat_dir, video_dir)
                    if os.path.isdir(video_path):
                        # Check if directory contains frame images
                        frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                        if len(frame_files) > 0:
                            samples.append((video_path, idx))
        
        # Use fixed seed for reproducible splits
        random.seed(42)
        random.shuffle(samples)
        
        print(f"Total samples found: {len(samples)}")
        
        # Check dataset size and adjust splits accordingly
        total_samples = len(samples)
        samples_per_class = total_samples / len(self.categories)
        
        print(f"Samples per class: {samples_per_class:.1f}")
        
        if total_samples < 500:
            # Very small dataset - use conservative splits
            print("WARNING: Small dataset detected - using conservative splits (60/20/20)")
            train_end = int(0.6 * total_samples)
            val_end = int(0.8 * total_samples)
        elif total_samples < 1000:
            # Medium dataset - standard splits  
            print("Medium dataset - using standard splits (70/15/15)")
            train_end = int(0.7 * total_samples)
            val_end = int(0.85 * total_samples)
        else:
            # Large dataset - can use more aggressive test split
            print("Large dataset - using test-heavy splits (70/10/20)")
            train_end = int(0.7 * total_samples)
            val_end = int(0.8 * total_samples)
        
        if self.mode == 'train':
            samples = samples[:train_end]
        elif self.mode == 'val':
            samples = samples[train_end:val_end]
        elif self.mode == 'test':
            samples = samples[val_end:]
        elif self.mode == 'all':
            # Return all samples for cross-validation
            print("Using ALL samples for cross-validation")
            pass
        else:
            # If mode not specified, return all samples (backward compatibility)
            pass
        
        print(f"Dataset split - {self.mode.upper()}: {len(samples)} samples")
        
        # Warning for very small test sets
        if self.mode == 'test' and len(samples) < 50:
            print("WARNING: Very small test set - results may not be reliable!")
            
        return samples

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, video_dir_path):
        # Get all frame files and sort them
        frame_files = sorted([f for f in os.listdir(video_dir_path) if f.endswith('.jpg')])
        total_frames = len(frame_files)
        
        if total_frames == 0:
            raise ValueError(f"No frame files found in {video_dir_path}")
        
        frames = []
        
        # Check if this is processed data (exactly 8 frames named frame_XX.jpg)
        is_processed = (total_frames == 8 and 
                       frame_files[0].startswith('frame_') and 
                       frame_files[-1].startswith('frame_'))
        
        if is_processed:
            # For processed data, load all frames sequentially
            for frame_file in frame_files[:self.num_frames]:
                frame_path = os.path.join(video_dir_path, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # Processed frames are already 224x224, but resize to be safe
                    frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                    frames.append(frame)
                else:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
        else:
            # For original data, use sampling logic
            indices = [min(i * self.sampling_rate, total_frames-1) for i in range(self.num_frames)]
            for idx in indices:
                frame_path = os.path.join(video_dir_path, frame_files[idx])
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                    frames.append(frame)
                else:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
        
        # Convert to tensors
        frames = [torch.tensor(f).permute(2,0,1).float()/255. for f in frames]
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_dir_path, label = self.samples[idx]
        frames = self._sample_frames(video_dir_path)
        
        # Apply augmentation if in training mode
        frames = self.augmentation.apply_augmentation(frames)
        
        return {'pixel_values': frames, 'labels': label}
