"""
Utility functions for training, evaluation, and visualization
"""

import torch
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, filename):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)
    else:
        print(f"No checkpoint found at: {filename}")
        return 0, 0.0

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {save_path}")
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=True):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    
    return plt.gcf()

def generate_classification_report(y_true, y_pred, class_names, save_path=None):
    """Generate detailed classification report"""
    
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Pretty print report
    print("\nDetailed Classification Report:")
    print("=" * 70)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)
    
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:<20} {metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} "
                  f"{int(metrics['support']):<10}")
    
    print("-" * 70)
    print(f"{'Accuracy':<20} {'':<10} {'':<10} {report['accuracy']:<10.3f} "
          f"{int(report['macro avg']['support']):<10}")
    print(f"{'Macro Avg':<20} {report['macro avg']['precision']:<10.3f} "
          f"{report['macro avg']['recall']:<10.3f} {report['macro avg']['f1-score']:<10.3f} "
          f"{int(report['macro avg']['support']):<10}")
    print(f"{'Weighted Avg':<20} {report['weighted avg']['precision']:<10.3f} "
          f"{report['weighted avg']['recall']:<10.3f} {report['weighted avg']['f1-score']:<10.3f} "
          f"{int(report['weighted avg']['support']):<10}")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nClassification report saved: {save_path}")
    
    return report

def plot_experiment_comparison(experiment_results, metric='best_accuracy', save_path=None):
    """Plot comparison of experiment results"""
    
    experiment_names = list(experiment_results.keys())
    values = [experiment_results[name][metric] for name in experiment_names]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(experiment_names)), values, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Experiments')
    plt.ylabel(f'{metric.replace("_", " ").title()} (%)')
    plt.title(f'Experiment Comparison - {metric.replace("_", " ").title()}')
    plt.xticks(range(len(experiment_names)), experiment_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Experiment comparison saved: {save_path}")
    
    return plt.gcf()

def get_class_names():
    """Get HMDB action class names in order"""
    return [
        'brush_hair', 'cartwheel', 'catch', 'chew', 'climb', 
        'climb_stairs', 'draw_sword', 'eat', 'fencing', 'flic_flac',
        'golf', 'handstand', 'kiss', 'pick', 'pour',
        'pullup', 'pushup', 'ride_bike', 'shoot_bow', 'shoot_gun',
        'situp', 'smile', 'smoke', 'throw', 'wave'
    ]

def calculate_model_complexity(model):
    """Calculate model parameters and FLOPs"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Convert to millions
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    
    print(f"Model Complexity:")
    print(f"  Total parameters: {total_params_m:.2f}M")
    print(f"  Trainable parameters: {trainable_params_m:.2f}M")
    print(f"  Non-trainable parameters: {(total_params - trainable_params)/1e6:.2f}M")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_m': total_params_m,
        'trainable_params_m': trainable_params_m
    }

def save_model_for_inference(model, config, save_path):
    """Save model with config for inference"""
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'class_names': get_class_names()
    }, save_path)
    
    print(f"Model saved for inference: {save_path}")

def create_experiment_summary(results_file):
    """Create a summary of all experiments"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\nEXPERIMENT SUMMARY")
    print("=" * 80)
    
    # Find best overall experiment
    best_exp = max(results.items(), key=lambda x: x[1]['best_accuracy'])
    
    print(f"Best performing experiment: {best_exp[0]}")
    print(f"Best accuracy: {best_exp[1]['best_accuracy']:.2f}%")
    print(f"Configuration: {best_exp[1]['config']}")
    
    # Group experiments by type
    experiment_types = {
        'optimizer': [k for k in results.keys() if k.startswith('optimizer_')],
        'batch_size': [k for k in results.keys() if k.startswith('batch_size_')],
        'learning_rate': [k for k in results.keys() if k.startswith('lr_')],
        'model': [k for k in results.keys() if k.startswith('model_')]
    }
    
    for exp_type, exp_names in experiment_types.items():
        if exp_names:
            print(f"\n{exp_type.upper()} EXPERIMENTS:")
            print("-" * 40)
            for name in exp_names:
                acc = results[name]['best_accuracy']
                print(f"  {name}: {acc:.2f}%")
    
    return results

def visualize_training_progress(experiment_results, save_dir=None):
    """Create visualizations for all experiments"""
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for exp_name, result in experiment_results.items():
        # Training curves
        fig = plot_training_curves(
            result['train_losses'],
            result['val_losses'],
            result['train_accuracies'],
            result['val_accuracies'],
            save_path=os.path.join(save_dir, f"{exp_name}_curves.png") if save_dir else None
        )
        plt.close(fig)
        
        # Confusion matrix
        if 'predictions' in result and 'targets' in result:
            fig = plot_confusion_matrix(
                result['targets'],
                result['predictions'],
                get_class_names(),
                save_path=os.path.join(save_dir, f"{exp_name}_confusion.png") if save_dir else None
            )
            plt.close(fig)
    
    # Overall comparison
    fig = plot_experiment_comparison(
        experiment_results,
        save_path=os.path.join(save_dir, "experiment_comparison.png") if save_dir else None
    )
    plt.close(fig)
    
    print(f"Visualizations saved to: {save_dir}")

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test class names
    class_names = get_class_names()
    print(f"Number of classes: {len(class_names)}")
    print(f"First 5 classes: {class_names[:5]}")
    
    print("Utility functions ready!")
