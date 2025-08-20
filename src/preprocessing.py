import os
import cv2
import yaml
import random
import numpy as np

def load_config(config_path='configs/coursework_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def augment_frame(frame, aug_cfg):
    # Random Cropping
    if aug_cfg.get('random_crop', False) and random.random() < 0.5:
        h, w = frame.shape[:2]
        crop_h, crop_w = int(h * 0.9), int(w * 0.9)
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)
        frame = frame[y:y+crop_h, x:x+crop_w]
        frame = cv2.resize(frame, (w, h))

    # Horizontal Flipping
    if aug_cfg.get('random_flip', False) and random.random() < 0.5:
        frame = cv2.flip(frame, 1)

    # Brightness Adjustment
    if aug_cfg.get('brightness', False) and random.random() < 0.5:
        value = random.randint(-30, 30)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.int16)
        hsv[...,2] = np.clip(hsv[...,2] + value, 0, 255)
        hsv = hsv.astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Gaussian Noise
    if aug_cfg.get('gaussian_noise', False) and random.random() < 0.5:
        noise = np.random.normal(0, 10, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # RGB Channel Swapping
    if aug_cfg.get('channel_swap', False) and random.random() < 0.5:
        frame = frame[..., [2, 1, 0]]  # BGR <-> RGB

    # Flicker Simulation
    if aug_cfg.get('flicker', False) and random.random() < 0.2:
        flicker = random.randint(-20, 20)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.int16)
        hsv[...,2] = np.clip(hsv[...,2] + flicker, 0, 255)
        hsv = hsv.astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame

def normalize_frame(frame):
    # Normalize to [0, 1] as in TimeSFormer
    return frame.astype(np.float32) / 255.0

def process_images(image_dir, output_dir, aug_cfg, clip_size=8, frame_size=(224, 224), sample_rate=32):
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    frames = []
    # Temporal sampling
    sampled_indices = [idx for idx in range(len(images)) if idx % sample_rate == 0]
    # Temporal Reversal
    if aug_cfg.get('temporal_reversal', False) and random.random() < 0.5:
        sampled_indices = sampled_indices[::-1]
    for idx in sampled_indices:
        img_path = os.path.join(image_dir, images[idx])
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame = cv2.resize(frame, frame_size)
        frame = augment_frame(frame, aug_cfg)
        frame = normalize_frame(frame)
        # Frame Dropping
        if aug_cfg.get('frame_drop', False) and random.random() < 0.1:
            continue
        # Frame Duplication
        if aug_cfg.get('frame_duplicate', False) and random.random() < 0.1:
            frames.append(frame)
        frames.append(frame)
    # Frame Averaging (motion blur)
    if aug_cfg.get('frame_average', False) and random.random() < 0.5 and len(frames) > 1:
        for i in range(1, len(frames)):
            frames[i] = cv2.addWeighted(frames[i], 0.5, frames[i-1], 0.5, 0)
    print(f"Extracted {len(frames)} frames from {os.path.basename(image_dir)}")
    # Frame Padding
    while len(frames) < clip_size:
        frames.append(np.zeros((*frame_size, 3), dtype=np.float32))
    # Save clips
    num_clips = 0
    for i in range(0, len(frames) - clip_size + 1, clip_size):
        clip = frames[i:i+clip_size]
        clip_dir = os.path.join(output_dir, f"clip_{i//clip_size:04d}")
        os.makedirs(clip_dir, exist_ok=True)
        for j, frame in enumerate(clip):
            cv2.imwrite(os.path.join(clip_dir, f"frame_{j:02d}.jpg"), (frame * 255).astype(np.uint8))
        num_clips += 1
    print(f"Saved {num_clips} clips to {output_dir}")

def preprocess_video_frames(video_path, num_frames=8, sampling_rate=32, frame_size=224):
    """
    Loads and preprocesses frames from a video folder.
    Returns a list of processed frames.
    """
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
    selected_frames = []
    step = max(1, len(frame_files) // num_frames)
    for i in range(0, len(frame_files), step):
        if len(selected_frames) >= num_frames:
            break
        frame_path = os.path.join(video_path, frame_files[i])
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame = frame.astype(np.float32) / 255.0
            selected_frames.append(frame)
    # Frame Padding
    while len(selected_frames) < num_frames:
        selected_frames.append(np.zeros((frame_size, frame_size, 3), dtype=np.float32))
    return selected_frames

def main():
    config = load_config()
    dataset_root = config['dataset']['root_dir']
    categories = config['dataset']['categories']
    aug_cfg = config.get('augmentation', {})
    results_dir = config['experiments']['results_dir']

    os.makedirs(results_dir, exist_ok=True)

    for category in categories:
        print(f"Processing category: {category}")
        category_path = os.path.join(dataset_root, category)
        output_category_dir = os.path.join(results_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)
        if not os.path.exists(category_path):
            print(f"Category path not found: {category_path}")
            continue
        video_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
        print(f"Found {len(video_folders)} video folders in {category}")
        for video_folder in video_folders:
            image_dir = os.path.join(category_path, video_folder)
            output_video_dir = os.path.join(output_category_dir, video_folder)
            os.makedirs(output_video_dir, exist_ok=True)
            print(f"Processing image folder: {video_folder}")
            process_images(image_dir, output_video_dir, aug_cfg)
    print("Frame preprocessing complete. All processed clips saved in results folder.")

if __name__ == "__main__":
    main()