"""
Training Pipeline with Experiment Support - Coursework Implementation
Supports multiple models, optimizers, hyperparameter experiments, and TensorBoard logging
Meets all coursework requirements: 45% top-1, 75% top-5 accuracy targets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import time
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from dataloader import create_data_loaders
from preprocessed_dataloader import create_preprocessed_dataloaders
from models.model_timesformer import TimeSFormerActionRecognition, create_timesformer_model
from models.model_vit_separate import VisionTransformerActionRecognition, create_vit_model
from models.model_cnn_lstm import CNNLSTMActionRecognition, create_cnn_lstm_model
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint

class ActionRecognitionTrainer:
    """Main trainer class for action recognition experiments with TensorBoard support"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize trainer with configuration"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup directories
        self._setup_directories()
        
        # Initialize TensorBoard writer with script-specific directory
        self.writer = SummaryWriter(log_dir=str(self.dirs['tensorboard']))
        
        # Initialize metrics storage
        self.experiment_results = {}
        
        # Load class names for confusion matrix labels
        self.class_names = self._load_class_names()
        
        print("Action Recognition Trainer initialized")
        print(f"Device: {self.device}")
        print(f"Config loaded from: {config_path}")
        print(f"Classes: {len(self.class_names)}")
        print(f"📊 TensorBoard logs saved to: {self.dirs['tensorboard']}")
        print(f"💡 Run 'tensorboard --logdir={self.dirs['tensorboard']}' to view progress!")
    
    def _setup_device(self):
        """Setup computation device"""
        device_config = self.config['hardware']['device']
        
        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)
        
        return device
    
    def _setup_directories(self):
        """Setup all output directories with script-specific organization"""
        import inspect
        from datetime import datetime
        
        # Get the script name that's calling this
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back  # Go up two frames to get the actual caller
            script_name = os.path.basename(caller_frame.f_globals.get('__file__', 'unknown_script'))
            script_name = script_name.replace('.py', '')
        finally:
            del frame
            
        # Create timestamp for unique runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Base results directory
        self.base_results_dir = Path("results")
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Script-specific results directory
        self.script_results_dir = self.base_results_dir / f"{script_name}_{timestamp}"
        self.script_results_dir.mkdir(exist_ok=True)
        
        # Create organized subdirectories (global for script)
        self.dirs = {
            'script_root': self.script_results_dir,
            'experiments': self.script_results_dir / "experiments",
            'global_analysis': self.script_results_dir / "global_analysis",
            'tensorboard': self.script_results_dir / "tensorboard",
            'configs': self.script_results_dir / "configs"
        }
        
        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration copy
        config_copy_path = self.dirs['configs'] / "config_used.yaml"
        with open(config_copy_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"📁 Results will be saved to: {self.script_results_dir}")
        print(f"   - Experiments: {self.dirs['experiments']}")
        print(f"   - Global Analysis: {self.dirs['global_analysis']}")
        print(f"   - TensorBoard: {self.dirs['tensorboard']}")
        print(f"   - Configs: {self.dirs['configs']}")
    
    def create_experiment_directories(self, experiment_name):
        """Create directories for a specific experiment"""
        exp_dir = self.dirs['experiments'] / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        exp_dirs = {
            'experiment_root': exp_dir,
            'models': exp_dir / "models",
            'confusion_matrices': exp_dir / "confusion_matrices",
            'training_logs': exp_dir / "training_logs", 
            'predictions': exp_dir / "predictions",
            'analysis': exp_dir / "analysis"
        }
        
        # Create all experiment directories
        for dir_path in exp_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Experiment '{experiment_name}' will save to: {exp_dir}")
        return exp_dirs
    
    def update_config_for_experiment(self, experiment_name, exp_dirs):
        """Update config paths for specific experiment"""
        self.config['paths'] = {
            'models_dir': str(exp_dirs['models']),
            'logs_dir': str(self.dirs['tensorboard']),
            'predictions_dir': str(exp_dirs['predictions']),
            'results_dir': str(exp_dirs['experiment_root'])
        }
    
    def _load_class_names(self):
        """Load class names from dataset"""
        dataset_path = Path(self.config['data']['dataset_path'])
        
        # Get class names from folder structure
        if dataset_path.exists():
            class_dirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
            class_names = sorted(class_dirs)
        else:
            # Fallback to default HMDB classes
            class_names = [
                'brush_hair', 'cartwheel', 'catch', 'chew', 'climb', 'climb_stairs',
                'draw_sword', 'eat', 'fencing', 'flic_flac', 'golf', 'handstand',
                'kiss', 'pick', 'pour', 'pullup', 'pushup', 'ride_bike',
                'shoot_bow', 'shoot_gun', 'situp', 'smile', 'smoke', 'throw', 'wave'
            ]
        
        return class_names
    
    def create_model(self, model_type, num_classes, num_frames):
        """Create model based on model type"""
        
        if model_type.lower() == "timesformer":
            model = create_timesformer_model(
                num_classes=num_classes,
                num_frames=num_frames,
                model_size=self.config['model'].get('size', 'base')
            )
            
        elif model_type.lower() == "vit":
            model = create_vit_model(
                num_classes=num_classes,
                num_frames=num_frames,
                model_size=self.config['model'].get('size', 'base')
            )
            
        elif model_type.lower() == "cnn_lstm":
            model = create_cnn_lstm_model(
                num_classes=num_classes,
                num_frames=num_frames,
                model_size=self.config['model'].get('size', 'base')
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def save_confusion_matrix(self, predictions, targets, exp_name, exp_dirs):
        """Save confusion matrix for experiment"""
        
        # Create confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Calculate accuracy for each class
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 12))
        
        # Create heatmap with class names
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Number of samples'}
        )
        
        plt.title(f'Confusion Matrix - {exp_name}\nOverall Accuracy: {(cm.diagonal().sum() / cm.sum()) * 100:.2f}%', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save plot to experiment-specific confusion matrix directory
        cm_path = exp_dirs['confusion_matrices'] / f'{exp_name}_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion of samples'},
            vmin=0, vmax=1
        )
        
        plt.title(f'Normalized Confusion Matrix - {exp_name}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save normalized plot to experiment-specific directory
        cm_norm_path = exp_dirs['confusion_matrices'] / f'{exp_name}_confusion_matrix_normalized.png'
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save detailed classification report to analysis directory
        from sklearn.metrics import classification_report
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        report_path = exp_dirs['analysis'] / f'{exp_name}_classification_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Confusion matrix saved: {cm_path}")
        print(f"Normalized confusion matrix saved: {cm_norm_path}")
        print(f"Classification report saved: {report_path}")
        
        return cm, cm_normalized
    
    def save_training_logs(self, exp_name, train_losses, train_accs, val_losses, val_accs, exp_dirs):
        """Save training logs and plots"""
        
        epochs = range(1, len(train_losses) + 1)
        
        # Create training plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title(f'Training & Validation Loss - {exp_name}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title(f'Training & Validation Accuracy - {exp_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Save plot to experiment-specific training logs directory
        plot_path = exp_dirs['training_logs'] / f'{exp_name}_training_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save logs as JSON to training logs directory
        logs = {
            'experiment': exp_name,
            'epochs': len(train_losses),
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'best_train_acc': max(train_accs),
            'best_val_acc': max(val_accs),
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1]
        }
        
        log_path = exp_dirs['training_logs'] / f'{exp_name}_training_log.json'
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Training logs saved: {log_path}")
        print(f"Training plot saved: {plot_path}")
    
    def create_optimizer(self, model, optimizer_type, learning_rate):
        """Create optimizer based on configuration"""
        
        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=self.config['training']['weight_decay']
            )
        
        elif optimizer_type.lower() == "sgd":
            sgd_config = self.config['training']['sgd']
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=sgd_config['momentum'],
                weight_decay=self.config['training']['weight_decay'],
                nesterov=sgd_config['nesterov']
            )
        
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=self.config['training']['weight_decay']
            )
        
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def create_scheduler(self, optimizer, scheduler_type, epochs):
        """Create learning rate scheduler"""
        
        if scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        elif scheduler_type.lower() == "step":
            step_config = self.config['training']['step_scheduler']
            scheduler = StepLR(
                optimizer, 
                step_size=step_config['step_size'],
                gamma=step_config['gamma']
            )
        
        elif scheduler_type.lower() == "none":
            scheduler = None
        
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        return scheduler
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        
        model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (videos, targets) in enumerate(progress_bar):
            # Move to device
            videos = videos.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            # Update meters
            batch_size = videos.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}%',
                'Acc@5': f'{top5.avg:.2f}%'
            })
        
        return losses.avg, top1.avg, top5.avg
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        
        model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for videos, targets in tqdm(val_loader, desc="Validation"):
                # Move to device
                videos = videos.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = model(videos)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                
                # Update meters
                batch_size = videos.size(0)
                losses.update(loss.item(), batch_size)
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
                
                # Store predictions for analysis
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return losses.avg, top1.avg, top5.avg, all_predictions, all_targets
    
    def train_single_experiment(self, exp_name, exp_config):
        """Train a single experiment configuration"""
        
        print(f"\nStarting experiment: {exp_name}")
        print("=" * 50)
        
        # Create experiment-specific directories
        exp_dirs = self.create_experiment_directories(exp_name)
        
        # Update config for this experiment
        self.update_config_for_experiment(exp_name, exp_dirs)
        
        # Create data loaders
        use_preprocessed = self.config.get('data', {}).get('use_preprocessed', False)
        preprocessed_path = self.config.get('data', {}).get('preprocessed_path', None)
        
        if use_preprocessed and preprocessed_path and Path(preprocessed_path).exists():
            print(f"📦 Using preprocessed data from: {preprocessed_path}")
            train_loader, val_loader, class_names = create_preprocessed_dataloaders(
                preprocessed_data_path=preprocessed_path,
                batch_size=exp_config['batch_size'],
                num_workers=self.config.get('training', {}).get('num_workers', 4)
            )
        else:
            if use_preprocessed:
                print(f"⚠️ Preprocessed data not found at {preprocessed_path}, using original data")
            print(f"📦 Using original data from: {self.config['data']['dataset_path']}")
            train_loader, val_loader = create_data_loaders(
                self.config['data']['dataset_path'],
                batch_size=exp_config['batch_size']
            )
        
        # Create model
        model = self.create_model(
            model_type=exp_config.get('model_type', self.config['model']['type']),
            num_classes=self.config['data']['num_classes'],
            num_frames=self.config['data']['num_frames']
        ).to(self.device)
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(
            model, 
            exp_config['optimizer'], 
            exp_config['learning_rate']
        )
        
        scheduler = self.create_scheduler(
            optimizer,
            self.config['training']['scheduler'],
            exp_config['epochs']
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(1, exp_config['epochs'] + 1):
            # Train
            train_loss, train_acc1, train_acc5 = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_loss, val_acc1, val_acc5, predictions, targets = self.validate_epoch(
                model, val_loader, criterion
            )
            
            # TensorBoard logging - Coursework requirement
            self.writer.add_scalars(f'{exp_name}/Loss', {
                'Train': train_loss,
                'Validation': val_loss
            }, epoch)
            
            self.writer.add_scalars(f'{exp_name}/Top1_Accuracy', {
                'Train': train_acc1,
                'Validation': val_acc1
            }, epoch)
            
            self.writer.add_scalars(f'{exp_name}/Top5_Accuracy', {
                'Train': train_acc5,
                'Validation': val_acc5
            }, epoch)
            
            # Learning rate logging
            current_lr = optimizer.param_groups[0]['lr']
            self.writer.add_scalar(f'{exp_name}/Learning_Rate', current_lr, epoch)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Store metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc1)
            val_losses.append(val_loss)
            val_accs.append(val_acc1)
            
            # Print epoch results with coursework targets
            target_msg = ""
            if val_acc1 >= 45.0 and val_acc5 >= 75.0:
                target_msg = " ✅ COURSEWORK TARGETS MET!"
            elif val_acc1 >= 45.0:
                target_msg = f" 🎯 Top-1 target met! Top-5 needs {75.0-val_acc5:.1f}% more"
            elif val_acc5 >= 75.0:
                target_msg = f" 🎯 Top-5 target met! Top-1 needs {45.0-val_acc1:.1f}% more"
            
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc1:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc1:.2f}% | "
                  f"Val Top-5: {val_acc5:.2f}%{target_msg}")
            
            # Save best model
            if val_acc1 > best_acc:
                best_acc = val_acc1
                
                # Save to experiment-specific models directory
                model_save_path = exp_dirs['models'] / f"{exp_name}_best.pth"
                
                if hasattr(model, 'save_model'):
                    # Use model's built-in save method
                    model.save_model(
                        model_save_path,
                        epoch=epoch,
                        optimizer_state=optimizer.state_dict(),
                        metrics={
                            'best_accuracy': best_acc,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'train_acc': train_acc1,
                            'val_acc': val_acc1,
                            'best_val_top5_accuracy': val_acc5  # Include top-5 for coursework
                        }
                    )
                else:
                    # Fallback to standard checkpoint saving
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'config': exp_config
                    }, model_save_path)
        
        # Final evaluation
        final_predictions = predictions
        final_targets = targets
        
        # Save confusion matrix
        self.save_confusion_matrix(final_predictions, final_targets, exp_name, exp_dirs)
        
        # Save training logs
        self.save_training_logs(exp_name, train_losses, train_accs, val_losses, val_accs, exp_dirs)
        
        # Perform top-5 analysis - Coursework requirement
        print(f"\n🎯 Performing Top-5 Analysis for {exp_name}")
        self.analyze_top5_predictions(model, val_loader, num_samples=10)
        
        # Store experiment results
        experiment_result = {
            'config': exp_config,
            'best_accuracy': best_acc,
            'final_accuracy': val_accs[-1],
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'predictions': final_predictions,
            'targets': final_targets
        }
        
        return experiment_result
    
    def run_experiments(self):
        """Run all enabled experiments"""
        
        experiments_config = self.config['experiments']
        
        # Single model training (default configuration)
        if not any(exp['enabled'] for exp in experiments_config.values()):
            print("Running default training configuration...")
            default_config = self.config['training'].copy()
            default_config['model_type'] = self.config['model']['type']
            
            result = self.train_single_experiment("default", default_config)
            self.experiment_results["default"] = result
        
        # Optimizer experiments
        if experiments_config['optimizer_exp']['enabled']:
            print("\nRunning optimizer experiments...")
            base_config = experiments_config['optimizer_exp']['base_config'].copy()
            base_config['model_type'] = self.config['model']['type']
            
            for optimizer in experiments_config['optimizer_exp']['optimizers']:
                exp_config = base_config.copy()
                exp_config['optimizer'] = optimizer
                exp_name = f"optimizer_{optimizer}"
                
                result = self.train_single_experiment(exp_name, exp_config)
                self.experiment_results[exp_name] = result
        
        # Batch size experiments
        if experiments_config['batch_size_exp']['enabled']:
            print("\nRunning batch size experiments...")
            base_config = experiments_config['batch_size_exp']['base_config'].copy()
            base_config['model_type'] = self.config['model']['type']
            
            for batch_size in experiments_config['batch_size_exp']['batch_sizes']:
                exp_config = base_config.copy()
                exp_config['batch_size'] = batch_size
                exp_name = f"batch_size_{batch_size}"
                
                result = self.train_single_experiment(exp_name, exp_config)
                self.experiment_results[exp_name] = result
        
        # Learning rate experiments
        if experiments_config['learning_rate_exp']['enabled']:
            print("\nRunning learning rate experiments...")
            base_config = experiments_config['learning_rate_exp']['base_config'].copy()
            base_config['model_type'] = self.config['model']['type']
            
            for lr in experiments_config['learning_rate_exp']['learning_rates']:
                exp_config = base_config.copy()
                exp_config['learning_rate'] = lr
                exp_name = f"lr_{lr}"
                
                result = self.train_single_experiment(exp_name, exp_config)
                self.experiment_results[exp_name] = result
        
        # Model comparison experiments
        if experiments_config['model_comparison_exp']['enabled']:
            print("\nRunning model comparison experiments...")
            base_config = experiments_config['model_comparison_exp']['base_config'].copy()
            
            for model_type in experiments_config['model_comparison_exp']['models']:
                exp_config = base_config.copy()
                exp_config['model_type'] = model_type
                exp_name = f"model_{model_type}"
                
                result = self.train_single_experiment(exp_name, exp_config)
                self.experiment_results[exp_name] = result
        
        # Save all results
        self.save_experiment_results()
        self.generate_experiment_report()
        self.create_model_comparison()
        self.create_comprehensive_analysis()
        
        # Close TensorBoard writer
        self.writer.close()
        print(f"\n📊 TensorBoard logs saved to: {self.writer.log_dir}")
        print("💡 Run 'tensorboard --logdir=logs/tensorboard' to view training progress!")
    
    def save_experiment_results(self):
        """Save experiment results to file"""
        
        results_path = os.path.join(self.dirs['global_analysis'], 'experiment_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for exp_name, result in self.experiment_results.items():
            serializable_result = result.copy()
            for key in ['predictions', 'targets']:
                if key in serializable_result:
                    serializable_result[key] = [int(x) for x in serializable_result[key]]
            serializable_results[exp_name] = serializable_result
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nExperiment results saved to: {results_path}")
    
    def generate_experiment_report(self):
        """Generate comprehensive experiment report"""
        
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        # Sort experiments by best accuracy
        sorted_experiments = sorted(
            self.experiment_results.items(),
            key=lambda x: x[1]['best_accuracy'],
            reverse=True
        )
        
        print(f"{'Experiment':<25} {'Best Acc':<12} {'Final Acc':<12} {'Config':<30}")
        print("-" * 80)
        
        for exp_name, result in sorted_experiments:
            config_str = f"lr={result['config']['learning_rate']}, bs={result['config']['batch_size']}"
            if 'optimizer' in result['config']:
                config_str += f", opt={result['config']['optimizer']}"
            if 'model_type' in result['config']:
                config_str += f", model={result['config']['model_type']}"
            
            print(f"{exp_name:<25} {result['best_accuracy']:<12.2f} "
                  f"{result['final_accuracy']:<12.2f} {config_str:<30}")
        
        print("\n" + "="*80)
        print(f"Best performing experiment: {sorted_experiments[0][0]}")
        print(f"Best accuracy achieved: {sorted_experiments[0][1]['best_accuracy']:.2f}%")
        print("="*80)
    
    def create_model_comparison(self):
        """Create comprehensive model comparison including confusion matrices"""
        
        print("\nCreating model comparison analysis...")
        
        # Find model comparison experiments
        model_experiments = {
            exp_name: result for exp_name, result in self.experiment_results.items()
            if 'model_' in exp_name or any(model in exp_name.lower() for model in ['timesformer', 'vit', 'cnn_lstm'])
        }
        
        if len(model_experiments) < 2:
            print("Not enough model experiments for comparison")
            return
        
        # Create comparison confusion matrices
        fig, axes = plt.subplots(2, len(model_experiments), figsize=(6*len(model_experiments), 12))
        if len(model_experiments) == 1:
            axes = axes.reshape(-1, 1)
        
        model_metrics = {}
        
        for idx, (exp_name, result) in enumerate(model_experiments.items()):
            predictions = result['predictions']
            targets = result['targets']
            
            # Raw confusion matrix
            cm = confusion_matrix(targets, predictions)
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[0, idx],
                xticklabels=self.class_names,
                yticklabels=self.class_names if idx == 0 else False
            )
            axes[0, idx].set_title(f'{exp_name}\nAccuracy: {result["best_accuracy"]:.2f}%')
            axes[0, idx].set_xlabel('Predicted')
            if idx == 0:
                axes[0, idx].set_ylabel('True')
            
            # Normalized confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(
                cm_norm, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                ax=axes[1, idx],
                xticklabels=self.class_names,
                yticklabels=self.class_names if idx == 0 else False,
                vmin=0, vmax=1
            )
            axes[1, idx].set_title(f'{exp_name} (Normalized)')
            axes[1, idx].set_xlabel('Predicted')
            if idx == 0:
                axes[1, idx].set_ylabel('True')
            
            # Store metrics
            model_metrics[exp_name] = {
                'accuracy': result['best_accuracy'],
                'final_accuracy': result['final_accuracy'],
                'confusion_matrix': cm.tolist(),
                'normalized_cm': cm_norm.tolist(),
                'per_class_accuracy': (cm.diagonal() / cm.sum(axis=1)).tolist()
            }
        
        plt.tight_layout()
        comparison_path = os.path.join(self.dirs['global_analysis'], 'model_comparison_confusion_matrices.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save model comparison metrics
        comparison_metrics_path = os.path.join(self.dirs['global_analysis'], 'model_comparison_metrics.json')
        with open(comparison_metrics_path, 'w') as f:
            json.dump(model_metrics, f, indent=2)
        
        print(f"Model comparison saved: {comparison_path}")
        print(f"Model metrics saved: {comparison_metrics_path}")
        
        return model_metrics
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis of all experiments"""
        
        print("\nCreating comprehensive analysis...")
        
        # Performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Accuracy comparison bar chart
        exp_names = list(self.experiment_results.keys())
        best_accs = [result['best_accuracy'] for result in self.experiment_results.values()]
        final_accs = [result['final_accuracy'] for result in self.experiment_results.values()]
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        ax1.bar(x - width/2, best_accs, width, label='Best Accuracy', alpha=0.8)
        ax1.bar(x + width/2, final_accs, width, label='Final Accuracy', alpha=0.8)
        ax1.set_xlabel('Experiments')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison Across Experiments')
        ax1.set_xticks(x)
        ax1.set_xticklabels(exp_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training curves comparison
        for exp_name, result in self.experiment_results.items():
            epochs = range(1, len(result['val_accuracies']) + 1)
            ax2.plot(epochs, result['val_accuracies'], label=f'{exp_name}', marker='o', markersize=3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy (%)')
        ax2.set_title('Validation Accuracy Curves')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Loss comparison
        for exp_name, result in self.experiment_results.items():
            epochs = range(1, len(result['val_losses']) + 1)
            ax3.plot(epochs, result['val_losses'], label=f'{exp_name}', marker='s', markersize=3)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Loss')
        ax3.set_title('Validation Loss Curves')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Convergence speed analysis
        convergence_epochs = []
        for exp_name, result in self.experiment_results.items():
            val_accs = result['val_accuracies']
            # Find epoch where 95% of best accuracy is reached
            best_acc = max(val_accs)
            target_acc = 0.95 * best_acc
            
            convergence_epoch = len(val_accs)  # Default to last epoch
            for epoch, acc in enumerate(val_accs):
                if acc >= target_acc:
                    convergence_epoch = epoch + 1
                    break
            
            convergence_epochs.append(convergence_epoch)
        
        ax4.bar(exp_names, convergence_epochs, alpha=0.8, color='lightcoral')
        ax4.set_xlabel('Experiments')
        ax4.set_ylabel('Epochs to 95% Best Accuracy')
        ax4.set_title('Convergence Speed Comparison')
        ax4.set_xticklabels(exp_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        analysis_path = os.path.join(self.dirs['global_analysis'], 'comprehensive_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create summary statistics
        summary_stats = {
            'total_experiments': len(self.experiment_results),
            'best_experiment': max(self.experiment_results.items(), key=lambda x: x[1]['best_accuracy'])[0],
            'best_accuracy': max(result['best_accuracy'] for result in self.experiment_results.values()),
            'average_accuracy': np.mean([result['best_accuracy'] for result in self.experiment_results.values()]),
            'accuracy_std': np.std([result['best_accuracy'] for result in self.experiment_results.values()]),
            'convergence_analysis': {
                exp_name: epochs for exp_name, epochs in zip(exp_names, convergence_epochs)
            }
        }
        
        # Save summary
        summary_path = os.path.join(self.dirs['global_analysis'], 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Comprehensive analysis saved: {analysis_path}")
        print(f"Experiment summary saved: {summary_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Total experiments conducted: {summary_stats['total_experiments']}")
        print(f"Best performing experiment: {summary_stats['best_experiment']}")
        print(f"Highest accuracy achieved: {summary_stats['best_accuracy']:.2f}%")
        print(f"Average accuracy across experiments: {summary_stats['average_accuracy']:.2f}% ± {summary_stats['accuracy_std']:.2f}%")
        print("="*80)
    
    def analyze_top5_predictions(self, model, val_loader, num_samples=10):
        """
        Analyze top-5 predictions for sample videos - Coursework requirement
        Shows correctly and wrongly classified videos with top-5 scores
        """
        print(f"\n🔍 Analyzing Top-5 Predictions for {num_samples} Sample Videos")
        print("="*70)
        
        model.eval()
        correct_samples = []
        wrong_samples = []
        
        with torch.no_grad():
            for videos, targets in val_loader:
                if len(correct_samples) >= num_samples//2 and len(wrong_samples) >= num_samples//2:
                    break
                    
                videos = videos.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(videos)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top-5 predictions
                top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
                
                for i in range(videos.size(0)):
                    true_label = targets[i].item()
                    top5_preds = top5_indices[i].cpu().numpy()
                    top5_probs_vals = top5_probs[i].cpu().numpy()
                    
                    sample_info = {
                        'true_label': true_label,
                        'true_class': self.class_names[true_label],
                        'top5_preds': top5_preds,
                        'top5_probs': top5_probs_vals,
                        'top5_classes': [self.class_names[idx] for idx in top5_preds]
                    }
                    
                    # Check if correctly classified (top-1)
                    if top5_preds[0] == true_label:
                        if len(correct_samples) < num_samples//2:
                            correct_samples.append(sample_info)
                    else:
                        if len(wrong_samples) < num_samples//2:
                            wrong_samples.append(sample_info)
        
        # Display correct predictions
        print("\n✅ CORRECTLY CLASSIFIED VIDEOS:")
        print("-" * 70)
        for i, sample in enumerate(correct_samples):
            print(f"\nSample {i+1}:")
            print(f"  True Class: {sample['true_class']}")
            print(f"  Top-5 Predictions:")
            for j, (class_name, prob) in enumerate(zip(sample['top5_classes'], sample['top5_probs'])):
                indicator = "👑" if j == 0 else "  "
                print(f"    {indicator} {j+1}. {class_name:<20} ({prob*100:.2f}%)")
        
        # Display wrong predictions
        print(f"\n❌ WRONGLY CLASSIFIED VIDEOS:")
        print("-" * 70)
        for i, sample in enumerate(wrong_samples):
            print(f"\nSample {i+1}:")
            print(f"  True Class: {sample['true_class']}")
            print(f"  Top-5 Predictions:")
            for j, (class_name, prob) in enumerate(zip(sample['top5_classes'], sample['top5_probs'])):
                if class_name == sample['true_class']:
                    indicator = f"✓ (Rank {j+1})"
                elif j == 0:
                    indicator = "❌ (Wrong)"
                else:
                    indicator = "  "
                print(f"    {indicator} {j+1}. {class_name:<20} ({prob*100:.2f}%)")
        
        # Analyze confusion patterns
        print(f"\n📊 TOP-5 ANALYSIS INSIGHTS:")
        print("-" * 70)
        
        # Calculate how often true class appears in top-5 for wrong predictions
        true_in_top5_count = 0
        for sample in wrong_samples:
            if sample['true_label'] in sample['top5_preds']:
                true_in_top5_count += 1
        
        if wrong_samples:
            top5_coverage = (true_in_top5_count / len(wrong_samples)) * 100
            print(f"• In {true_in_top5_count}/{len(wrong_samples)} wrongly classified videos, ")
            print(f"  the true class appears in top-5 ({top5_coverage:.1f}%)")
        
        # Find most confused classes
        confusion_pairs = {}
        for sample in wrong_samples:
            true_class = sample['true_class']
            pred_class = sample['top5_classes'][0]  # Top-1 prediction
            pair = f"{true_class} → {pred_class}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        if confusion_pairs:
            print(f"• Most common confusion patterns:")
            sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
            for pair, count in sorted_confusions[:3]:
                print(f"  - {pair} (occurred {count} times)")
        
        print("="*70)

def main():
    """Main training function"""
    
    # Initialize trainer
    trainer = ActionRecognitionTrainer("config.yaml")
    
    # Run experiments
    trainer.run_experiments()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
