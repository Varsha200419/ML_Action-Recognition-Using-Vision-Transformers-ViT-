"""
Main Entry Point for Action Recognition Project
Coordinates all components: training, evaluation, and demo
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add src and models to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from extract_data import DatasetExtractor
from train import ActionRecognitionTrainer
from evaluate import main as evaluate_models

def setup_project(config_path="config.yaml"):
    """Setup project directories and check requirements"""
    
    print("Setting up Action Recognition Project...")
    
    # Load config to get dataset path
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        dataset_path = Path(config['data']['dataset_path'])
    except:
        print(f"Warning: Could not load config from {config_path}, using default path")
        dataset_path = Path("data/raw/HMDB_simp")
    
    # Check if dataset is extracted
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Run: python src/extract_data.py")
        return False
    
    # Create output directories
    output_dirs = ['outputs/models', 'outputs/logs', 'outputs/predictions', 'outputs/results']
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("Project setup complete!")
    return True

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Action Recognition with TimeSFormer, ViT, and CNN-LSTM")
    parser.add_argument('--mode', type=str, default='demo', 
                       choices=['extract', 'train', 'evaluate', 'demo', 'all'],
                       help='Mode to run: extract data, train models, evaluate, or demo')
    
    parser.add_argument('--model', type=str, default='timesformer',
                       choices=['timesformer', 'vit', 'cnn_lstm', 'all'],
                       help='Model type to use for training')
    
    parser.add_argument('--experiment', type=str, default='default',
                       choices=['default', 'optimizer', 'batch_size', 'learning_rate', 'model_comparison', 'all'],
                       help='Experiment type to run')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ACTION RECOGNITION PROJECT - COURSEWORK IMPLEMENTATION")
    print("=" * 80)
    print("Models: TimeSFormer, Vision Transformer, CNN-LSTM")
    print("Dataset: HMDB-51 (simplified)")
    print("Features: Data augmentation, Experiment framework, Streamlit demo")
    print("=" * 80)
    
    if args.mode == 'extract':
        print("\n📦 EXTRACTING DATASET...")
        extractor = DatasetExtractor()
        extractor.extract_and_analyze()
    
    elif args.mode == 'train':
        print(f"\n🚀 TRAINING MODELS...")
        if not setup_project():
            return
        
        # Use enhanced training system
        trainer = ActionRecognitionTrainer("config.yaml")
        trainer.run_experiments()
    
    elif args.mode == 'evaluate':
        print(f"\n📊 EVALUATING MODELS...")
        if not setup_project():
            return
        evaluate_models()
    
    elif args.mode == 'demo':
        print(f"\n🎬 LAUNCHING STREAMLIT DEMO...")
        if not setup_project():
            return
        
        # Check if models exist
        models_dir = Path("outputs/models")
        if not any(models_dir.glob("*.pth")):
            print("No trained models found. Please train models first.")
            print("Run: python main.py --mode train")
            return
        
        # Launch Streamlit app
        os.system("streamlit run streamlit_app/app.py")
    
    elif args.mode == 'all':
        print(f"\n🔄 RUNNING COMPLETE PIPELINE...")
        
        # Load config to get dataset path
        try:
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            dataset_path = Path(config['data']['dataset_path'])
        except:
            dataset_path = Path("data/raw/HMDB_simp")
        
        # Step 1: Extract data (if needed)
        if not dataset_path.exists():
            print("\n📦 Step 1: Extracting dataset...")
            extractor = DatasetExtractor()
            extractor.extract_and_analyze()
        else:
            print("\n✅ Step 1: Dataset already extracted")
        
        if not setup_project():
            return
        
        # Step 2: Train models
        print("\n🚀 Step 2: Training models...")
        trainer = ActionRecognitionTrainer("config.yaml")
        trainer.run_experiments()
        
        # Step 3: Evaluate models
        print("\n📊 Step 3: Evaluating models...")
        evaluate_models()
        
        # Step 4: Launch demo
        print("\n🎬 Step 4: Launching demo...")
        print("Starting Streamlit app...")
        os.system("streamlit run streamlit_app/app.py")
    
    else:
        parser.print_help()

def quick_start():
    """Quick start guide for users"""
    
    print("QUICK START GUIDE")
    print("=" * 50)
    print("1. Extract dataset:")
    print("   python main.py --mode extract")
    print()
    print("2. Train models:")
    print("   python main.py --mode train")
    print()
    print("3. Evaluate models:")
    print("   python main.py --mode evaluate")
    print()
    print("4. Launch demo:")
    print("   python main.py --mode demo")
    print()
    print("5. Run everything:")
    print("   python main.py --mode all")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        quick_start()
    else:
        main()
