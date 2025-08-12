# Evaluation script
# Add your evaluation logic here

import torch
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, val_loader, device, categories):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Top-1 and Top-5 accuracy
    top1 = (np.array(all_preds) == np.array(all_labels)).mean()
    top5 = top_k_accuracy_score(all_labels, logits.cpu().numpy(), k=5)
    print(f"Top-1 Accuracy: {top1:.3f}, Top-5 Accuracy: {top5:.3f}")
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,10))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
