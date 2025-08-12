import os
import cv2
import yaml
import random
import numpy as np

def load_config(config_path='configs/coursework_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def augment_frame(frame, aug_cfg):
    # Example augmentations (add your own as needed)
    if aug_cfg.get('random_flip', False) and random.random() < 0.5:
        frame = cv2.flip(frame, 1)
    if aug_cfg.get('brightness', False):
        value = random.randint(-30, 30)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.int16)  # Fix: avoid overflow
        hsv[...,2] = np.clip(hsv[...,2] + value, 0, 255)
        hsv = hsv.astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Add more augmentations as needed
    return frame

def normalize_frame(frame):
    # Normalize to [0, 1] as in TimeSFormer
    return frame.astype(np.float32) / 255.0

def process_images(image_dir, output_dir, aug_cfg, clip_size=8, frame_size=(224, 224), sample_rate=32):  # Changed sample_rate to 32
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    frames = []
    for idx, img_name in enumerate(images):
        if idx % sample_rate == 0:
            img_path = os.path.join(image_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            frame = cv2.resize(frame, frame_size)
            frame = augment_frame(frame, aug_cfg)
            frame = normalize_frame(frame)  # Add normalization
            frames.append(frame)
    print(f"Extracted {len(frames)} frames from {os.path.basename(image_dir)}")
    # Save clips
    num_clips = 0
    for i in range(0, len(frames) - clip_size + 1, clip_size):
        clip = frames[i:i+clip_size]
        clip_dir = os.path.join(output_dir, f"clip_{i//clip_size:04d}")
        os.makedirs(clip_dir, exist_ok=True)
        for j, frame in enumerate(clip):
            cv2.imwrite(os.path.join(clip_dir, f"frame_{j:02d}.jpg"), frame)
        num_clips += 1
    print(f"Saved {num_clips} clips to {output_dir}")

def preprocess_video_frames(video_path, num_frames=8, sampling_rate=32, frame_size=224):
    """
    Loads and preprocesses frames from a video folder.
    Returns a list of processed frames.
    """
    import os
    import cv2
    import numpy as np
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
    # Pad if not enough frames
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