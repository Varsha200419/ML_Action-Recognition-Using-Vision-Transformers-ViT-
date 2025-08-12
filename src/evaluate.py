# Evaluation script
# Add your evaluation logic here

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
from dataset import HMDBVideoDataset
from transformers import AutoFeatureExtractor
from models.timesformer_model import get_timesformer_model
from models.vit_base_model import get_vit_model
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from PIL import Image

def process_batch(batch, extractor, device):
    pixel_values, labels = batch
    pil_videos = []
    for video in pixel_values:
        frames = []
        for frame in video:
            np_frame = frame.cpu().numpy()
            if np_frame.shape[0] == 1:
                np_frame = np_frame.squeeze(0)
            if np_frame.ndim == 2:
                np_frame = np.stack([np_frame]*3, axis=-1)
            elif np_frame.shape[0] == 3:
                np_frame = np_frame.transpose(1, 2, 0)
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

def evaluate(model, val_loader, extractor, device, categories):
    model.eval()
    all_preds, all_labels = [], []
    all_logits = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = process_batch(batch, extractor, device)
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    all_logits = np.concatenate(all_logits)
    top1 = (np.array(all_preds) == np.array(all_labels)).mean()
    top5 = np.mean([label in np.argsort(-logit)[:5] for label, logit in zip(all_labels, all_logits)])
    print(f"Top-1 Accuracy: {top1:.3f}, Top-5 Accuracy: {top5:.3f}")

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=range(len(categories)), zero_division=0)
    print("+--------------+---------------+-------------+----------+------------+")
    print("| Class Name   |   Class Index |   Precision |   Recall |   F1 Score |")
    print("+--------------+---------------+-------------+----------+------------+")
    for idx, cname in enumerate(categories):
        print(f"| {cname:<12} | {idx:^13} | {precision[idx]:^11.3f} | {recall[idx]:^8.3f} | {f1[idx]:^10.3f} |")
    print("+--------------+---------------+-------------+----------+------------+")

if __name__ == "__main__":
    with open("configs/coursework_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    DATA_PATH = config['dataset']['root_dir']
    CATEGORIES = config['dataset']['categories']
    NUM_FRAMES = config['dataset']['num_frames']
    FRAME_SIZE = config['dataset']['frame_size']
    SAMPLING_RATE = config['dataset']['sampling_rate']
    BATCH_SIZE = config['training']['batch_size']
    VAL_SPLIT = config['training'].get('val_split', 0.2)
    MODEL_PATH = "/content/AR_Vit/AR_Vit/Results/lr - 0.0005/model_final.pth"
    MODEL_NAME = config['model']['model_name']

    val_dataset = HMDBVideoDataset(
        root_dir=DATA_PATH,
        categories=CATEGORIES,
        num_frames=NUM_FRAMES,
        sampling_rate=SAMPLING_RATE,
        frame_size=FRAME_SIZE,
        split='val',
        val_split=VAL_SPLIT
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    if "vit" in MODEL_NAME.lower():
        model = get_vit_model(num_classes=len(CATEGORIES))
    else:
        model = get_timesformer_model(num_classes=len(CATEGORIES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    evaluate(model, val_loader, extractor, device, CATEGORIES)
