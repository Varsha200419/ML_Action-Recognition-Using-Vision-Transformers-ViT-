# Updated training script using HMDBVideoDataset and correct path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
from torch.utils.data import DataLoader
from dataset import HMDBVideoDataset
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification, AutoModelForImageClassification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from PIL import Image

def load_config(config_path='configs/coursework_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Parse command line for model name
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Model name or path')
    args = parser.parse_args()

    config = load_config()
    DATA_PATH = config['dataset']['root_dir']  # <-- Use root_dir for preprocessed frames
    CATEGORIES = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    NUM_FRAMES = config['dataset']['num_frames']
    FRAME_SIZE = config['dataset']['frame_size']
    SAMPLING_RATE = config['dataset']['sampling_rate']
    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']
    LEARNING_RATE = config['training']['learning_rate']
    VAL_SPLIT = config['training'].get('val_split', 0.2)

    # Use command line model name if provided, else config value
    model_name = args.model if args.model else config['model']['model_name']

    train_dataset = HMDBVideoDataset(
        root_dir=DATA_PATH,
        categories=CATEGORIES,
        num_frames=NUM_FRAMES,
        sampling_rate=SAMPLING_RATE,
        frame_size=FRAME_SIZE,
        split='train',
        val_split=VAL_SPLIT
    )
    val_dataset = HMDBVideoDataset(
        root_dir=DATA_PATH,
        categories=CATEGORIES,
        num_frames=NUM_FRAMES,
        sampling_rate=SAMPLING_RATE,
        frame_size=FRAME_SIZE,
        split='val',
        val_split=VAL_SPLIT
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if "vit" in model_name.lower():
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=len(CATEGORIES),
            ignore_mismatched_sizes=True
        )
        is_vit = True
    else:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForVideoClassification.from_pretrained(
            model_name,
            num_labels=len(CATEGORIES),
            ignore_mismatched_sizes=True
        )
        is_vit = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    def process_batch(batch):
        pixel_values, labels = batch
        videos = pixel_values
        pil_videos = []
        for video in videos:  # video: (num_frames, C, H, W)
            frames = []
            for frame in video:  # frame: (C, H, W)
                np_frame = frame.cpu().numpy()
                # If shape is (1, 224, 224) or (1, 1, 224), squeeze extra dims
                if np_frame.shape[0] == 1:
                    np_frame = np_frame.squeeze(0)
                # If grayscale, convert to RGB by stacking
                if np_frame.ndim == 2:
                    np_frame = np.stack([np_frame]*3, axis=-1)
                elif np_frame.shape[0] == 3:
                    np_frame = np_frame.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                # Convert to uint8 if needed
                if np_frame.max() <= 1.0:
                    np_frame = (np_frame * 255).astype(np.uint8)
                else:
                    np_frame = np_frame.astype(np.uint8)
                pil_frame = Image.fromarray(np_frame)
                frames.append(pil_frame)
            pil_videos.append(frames)
        inputs = extractor(pil_videos, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        return inputs, labels

    def evaluate(model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                inputs, labels = process_batch(batch)
                outputs = model(**inputs)
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        # Top-1 accuracy
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        # Top-5 accuracy
        top5 = np.argsort(-logits.cpu().numpy(), axis=1)[:, :5]
        top5_acc = np.mean([label in top5_row for label, top5_row in zip(all_labels, top5)])
        print(f"Top-1 Accuracy: {acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}")
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        ConfusionMatrixDisplay(cm, display_labels=CATEGORIES).plot()
        plt.show()
        return acc, top5_acc, cm

    def visualize_top5(logits, labels, all_preds, loader, CATEGORIES, num_samples=5):
        # Randomly select samples
        idxs = np.random.choice(len(labels), num_samples, replace=False)
        for idx in idxs:
            top5 = np.argsort(-logits[idx])[:5]
            print(f"True: {CATEGORIES[labels[idx]]}, Pred: {CATEGORIES[all_preds[idx]]}, Top-5: {[CATEGORIES[i] for i in top5]}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = process_batch(batch)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc, val_top5, all_preds, all_labels, logits = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val Top-5 Acc={val_top5:.4f}')
        visualize_top5(logits, all_labels, all_preds, val_loader, CATEGORIES, num_samples=5)

if __name__ == "__main__":
    main()
