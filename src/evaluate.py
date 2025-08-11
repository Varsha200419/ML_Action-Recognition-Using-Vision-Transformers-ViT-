"""
Evaluation and Analysis Module - Coursework Implementation
Comprehensive evaluation of trained models with detailed metrics
"""

import torch
import torch.nn as nn
import yaml
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dataloader import create_data_loaders
from models.model_vit_separate import create_vit_model
from utils import (
    accuracy, plot_confusion_matrix, generate_classification_report,
    get_class_names, calculate_model_complexity, AverageMeter
)

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize evaluator"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Class names
        self.class_names = get_class_names()
        
        print("Model Evaluator initialized")
        print(f"Device: {self.device}")
        print(f"Number of classes: {len(self.class_names)}")
    
    def load_trained_model(self, model_path, model_type=None):
        """Load a trained model from checkpoint"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model type from checkpoint or parameter
        if model_type is None:
            if 'config' in checkpoint and 'model_type' in checkpoint['config']:
                model_type = checkpoint['config']['model_type']
            else:
                model_type = self.config['model']['type']
        
        # Create model
        model = create_vit_model(
            model_type=model_type,
            num_classes=self.config['data']['num_classes'],
            num_frames=self.config['data']['num_frames']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded: {model_path}")
        print(f"Model type: {model_type}")
        
        return model, checkpoint
    
    def evaluate_model(self, model, data_loader, detailed=True):
        """Comprehensive model evaluation"""
        
        model.eval()
        
        # Metrics tracking
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # Detailed predictions
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Per-class metrics
        class_correct = np.zeros(len(self.class_names))
        class_total = np.zeros(len(self.class_names))
        
        criterion = nn.CrossEntropyLoss()
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for videos, targets in tqdm(data_loader, desc="Evaluation"):
                # Move to device
                videos = videos.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = model(videos)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                
                # Update metrics
                batch_size = videos.size(0)
                losses.update(loss.item(), batch_size)
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
                
                # Store detailed results
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Per-class accuracy
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_total[label] += 1
                    if predicted[i] == targets[i]:
                        class_correct[label] += 1
        
        # Calculate per-class accuracies
        per_class_acc = {}
        for i, class_name in enumerate(self.class_names):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                per_class_acc[class_name] = acc
            else:
                per_class_acc[class_name] = 0.0
        
        # Evaluation results
        eval_results = {
            'overall_accuracy': top1.avg,
            'top5_accuracy': top5.avg,
            'average_loss': losses.avg,
            'per_class_accuracy': per_class_acc,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        if detailed:
            self._print_detailed_results(eval_results)
        
        return eval_results
    
    def _print_detailed_results(self, results):
        """Print detailed evaluation results"""
        
        print("\n" + "="*80)
        print("DETAILED EVALUATION RESULTS")
        print("="*80)
        
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
        print(f"Average Loss: {results['average_loss']:.4f}")
        
        print(f"\nPER-CLASS ACCURACY:")
        print("-" * 40)
        
        # Sort classes by accuracy
        sorted_classes = sorted(
            results['per_class_accuracy'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for class_name, accuracy in sorted_classes:
            print(f"{class_name:<20}: {accuracy:>6.2f}%")
        
        # Find best and worst performing classes
        best_class = max(results['per_class_accuracy'].items(), key=lambda x: x[1])
        worst_class = min(results['per_class_accuracy'].items(), key=lambda x: x[1])
        
        print(f"\nBest performing class: {best_class[0]} ({best_class[1]:.2f}%)")
        print(f"Worst performing class: {worst_class[0]} ({worst_class[1]:.2f}%)")
        print("="*80)
    
    def compare_models(self, model_paths, model_types=None):
        """Compare multiple trained models"""
        
        print("\nMODEL COMPARISON")
        print("="*80)
        
        # Create test data loader
        _, test_loader = create_data_loaders(
            self.config['data']['dataset_path'],
            batch_size=self.config['training']['batch_size']
        )
        
        comparison_results = {}
        
        for i, model_path in enumerate(model_paths):
            model_name = os.path.basename(model_path).replace('.pth', '')
            
            # Determine model type
            if model_types and i < len(model_types):
                model_type = model_types[i]
            else:
                model_type = None
            
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Load and evaluate model
                model, checkpoint = self.load_trained_model(model_path, model_type)
                results = self.evaluate_model(model, test_loader, detailed=False)
                
                # Add model complexity
                complexity = calculate_model_complexity(model)
                results['model_complexity'] = complexity
                
                comparison_results[model_name] = results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        # Print comparison summary
        self._print_model_comparison(comparison_results)
        
        return comparison_results
    
    def _print_model_comparison(self, comparison_results):
        """Print model comparison summary"""
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Top-5':<12} {'Loss':<12} {'Params (M)':<12}")
        print("-" * 80)
        
        # Sort by accuracy
        sorted_models = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['overall_accuracy'],
            reverse=True
        )
        
        for model_name, results in sorted_models:
            accuracy = results['overall_accuracy']
            top5_acc = results['top5_accuracy']
            loss = results['average_loss']
            params = results['model_complexity']['total_params_m']
            
            print(f"{model_name:<25} {accuracy:<12.2f} {top5_acc:<12.2f} "
                  f"{loss:<12.4f} {params:<12.2f}")
        
        print("\nBest model: " + sorted_models[0][0])
    
    def analyze_predictions(self, model_path, save_dir=None):
        """Detailed prediction analysis with visualizations"""
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Load model and evaluate
        model, checkpoint = self.load_trained_model(model_path)
        _, test_loader = create_data_loaders(
            self.config['data']['dataset_path'],
            batch_size=self.config['training']['batch_size']
        )
        
        results = self.evaluate_model(model, test_loader)
        
        # Generate confusion matrix
        if save_dir:
            cm_path = os.path.join(save_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                results['targets'],
                results['predictions'],
                self.class_names,
                save_path=cm_path
            )
        
        # Generate classification report
        if save_dir:
            report_path = os.path.join(save_dir, "classification_report.json")
            generate_classification_report(
                results['targets'],
                results['predictions'],
                self.class_names,
                save_path=report_path
            )
        
        # Save detailed results
        if save_dir:
            results_path = os.path.join(save_dir, "evaluation_results.json")
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = results.copy()
            for key in ['predictions', 'targets']:
                serializable_results[key] = [int(x) for x in serializable_results[key]]
            
            serializable_results['probabilities'] = [
                [float(p) for p in prob] for prob in results['probabilities']
            ]
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"Analysis results saved to: {save_dir}")
        
        return results
    
    def error_analysis(self, model_path, save_dir=None):
        """Analyze model errors and failure cases"""
        
        print("\nERROR ANALYSIS")
        print("="*50)
        
        # Load model and get predictions
        model, checkpoint = self.load_trained_model(model_path)
        _, test_loader = create_data_loaders(
            self.config['data']['dataset_path'],
            batch_size=1  # Process one sample at a time for detailed analysis
        )
        
        results = self.evaluate_model(model, test_loader, detailed=False)
        
        # Find misclassified samples
        misclassified = []
        for i, (pred, target) in enumerate(zip(results['predictions'], results['targets'])):
            if pred != target:
                confidence = max(results['probabilities'][i])
                misclassified.append({
                    'sample_idx': i,
                    'predicted_class': self.class_names[pred],
                    'true_class': self.class_names[target],
                    'confidence': confidence,
                    'probabilities': results['probabilities'][i]
                })
        
        # Sort by confidence (most confident wrong predictions first)
        misclassified.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Total misclassified samples: {len(misclassified)}")
        print(f"Error rate: {len(misclassified)/len(results['predictions'])*100:.2f}%")
        
        # Show most confident wrong predictions
        print(f"\nMost confident wrong predictions:")
        print("-" * 70)
        print(f"{'True Class':<15} {'Predicted':<15} {'Confidence':<12} {'Sample #'}")
        print("-" * 70)
        
        for error in misclassified[:10]:  # Show top 10
            print(f"{error['true_class']:<15} {error['predicted_class']:<15} "
                  f"{error['confidence']:<12.3f} {error['sample_idx']}")
        
        # Analyze confusion patterns
        confusion_patterns = {}
        for error in misclassified:
            pattern = f"{error['true_class']} -> {error['predicted_class']}"
            confusion_patterns[pattern] = confusion_patterns.get(pattern, 0) + 1
        
        # Show most common confusion patterns
        print(f"\nMost common confusion patterns:")
        print("-" * 50)
        sorted_patterns = sorted(confusion_patterns.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_patterns[:10]:
            print(f"{pattern}: {count} times")
        
        if save_dir:
            error_analysis_path = os.path.join(save_dir, "error_analysis.json")
            with open(error_analysis_path, 'w') as f:
                json.dump({
                    'misclassified_samples': misclassified,
                    'confusion_patterns': confusion_patterns,
                    'error_rate': len(misclassified)/len(results['predictions'])*100
                }, f, indent=2)
            print(f"\nError analysis saved to: {error_analysis_path}")
        
        return misclassified, confusion_patterns

def main():
    """Main evaluation function"""
    
    evaluator = ModelEvaluator("config.yaml")
    
    # Check for trained models
    models_dir = "outputs/models"
    if not os.path.exists(models_dir):
        print("No trained models found. Please run training first.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("No model files found in outputs/models/")
        return
    
    print(f"Found {len(model_files)} trained models:")
    for i, model_file in enumerate(model_files):
        print(f"  {i+1}. {model_file}")
    
    # Evaluate all models
    model_paths = [os.path.join(models_dir, f) for f in model_files]
    comparison_results = evaluator.compare_models(model_paths)
    
    # Detailed analysis of best model
    if comparison_results:
        best_model = max(comparison_results.items(), key=lambda x: x[1]['overall_accuracy'])
        best_model_path = os.path.join(models_dir, f"{best_model[0]}.pth")
        
        print(f"\nPerforming detailed analysis of best model: {best_model[0]}")
        
        analysis_dir = "outputs/analysis"
        evaluator.analyze_predictions(best_model_path, analysis_dir)
        evaluator.error_analysis(best_model_path, analysis_dir)

if __name__ == "__main__":
    main()
