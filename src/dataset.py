import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import yaml

def load_config(config_path='configs/coursework_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_preprocessed_clips(results_dir, categories, clip_size=8):
    clips = []
    labels = []
    for category in categories:
        category_dir = os.path.join(results_dir, category)
        if not os.path.exists(category_dir):
            continue
        for video_folder in os.listdir(category_dir):
            video_dir = os.path.join(category_dir, video_folder)
            if not os.path.isdir(video_dir):
                continue
            for clip_folder in os.listdir(video_dir):
                clip_dir = os.path.join(video_dir, clip_folder)
                if not os.path.isdir(clip_dir):
                    continue
                clip_frames = []
                for i in range(clip_size):
                    frame_path = os.path.join(clip_dir, f"frame_{i:02d}.jpg")
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        frame = frame.astype(np.float32) / 255.0  # Normalize
                        clip_frames.append(frame)
                if len(clip_frames) == clip_size:
                    clips.append(np.stack(clip_frames))  # Shape: (clip_size, H, W, C)
                    labels.append(category)
    return clips, labels

def get_train_val_split(test_size=0.2, random_state=42):
    config = load_config()
    clips, labels = load_preprocessed_clips(
        config['experiments']['results_dir'],
        config['dataset']['categories'],
        clip_size=config['dataset']['num_frames']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        clips, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return X_train, X_val, y_train, y_val

import os
import cv2
import numpy as np
from glob import glob
from transforms import preprocess_and_augment
from torch.utils.data import Dataset
from preprocessing import preprocess_video_frames

# Dataset handling
# Add your dataset class and loading logic here

class HMDBDataset:
    def __init__(self, root_dir, clip_len=8, crop_size=(224, 224), transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.transform = transform if transform is not None else preprocess_and_augment
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for category in sorted(os.listdir(self.root_dir)):
            category_path = os.path.join(self.root_dir, category)
            if not os.path.isdir(category_path):
                continue
            for video_folder in sorted(os.listdir(category_path)):
                video_path = os.path.join(category_path, video_folder)
                if not os.path.isdir(video_path):
                    continue
                frame_files = sorted(glob(os.path.join(video_path, '*.jpg')))
                if len(frame_files) >= self.clip_len:
                    samples.append({'category': category, 'video': video_folder, 'frames': frame_files})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_files = sample['frames']
        # Sample frames at 1/32 rate as per TimeSFormer paper
        step = max(1, len(frame_files) // self.clip_len)
        selected_frames = [frame_files[i] for i in range(0, len(frame_files), step)][:self.clip_len]
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in selected_frames]
        frames = self.transform(frames, crop_size=self.crop_size, target_len=self.clip_len)
        label = sample['category']
        return np.stack(frames), label

class HMDBVideoDataset(Dataset):
    def __init__(self, root_dir, categories=None, num_frames=8, sampling_rate=32, frame_size=224, split='train', val_split=0.2, random_seed=42):
        self.root_dir = root_dir
        self.categories = categories if categories is not None else sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.split = split
        self.samples = []
        self.labels = []
        self._collect_samples()
        self._split_data(val_split, random_seed)

    def _collect_samples(self):
        for label_idx, category in enumerate(self.categories):
            category_path = os.path.join(self.root_dir, category)
            for video_folder in sorted(os.listdir(category_path)):
                video_path = os.path.join(category_path, video_folder)
                if os.path.isdir(video_path):
                    self.samples.append((video_path, label_idx))
                    self.labels.append(label_idx)

    def _split_data(self, val_split, random_seed):
        np.random.seed(random_seed)
        indices = np.arange(len(self.samples))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - val_split))
        if self.split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        video_path, label = self.samples[self.indices[idx]]
        frames = preprocess_video_frames(video_path, num_frames=self.num_frames, sampling_rate=self.sampling_rate, frame_size=self.frame_size)
        frames = np.stack(frames)  # Shape: (num_frames, H, W, C)
        return frames, label

# Example usage:
# train_dataset = HMDBVideoDataset(root_dir='path/to/HMDB_simp', split='train')
# val_dataset = HMDBVideoDataset(root_dir='path/to/HMDB_simp', split='val')
