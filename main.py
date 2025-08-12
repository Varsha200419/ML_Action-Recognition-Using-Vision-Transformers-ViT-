# main.py
# Entry point for HMDB Video Action Recognition experiments

import argparse
import os
import sys
import subprocess
from datetime import datetime

def print_banner():
    print("=" * 70)
    print("🎬 HMDB Video Action Recognition - Experiment Runner")
    print("=" * 70)

def run_script(script_path, args=None):
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    print(f"\n📋 Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc == 0

def main():
    print_banner()
    parser = argparse.ArgumentParser(description='HMDB Video Action Recognition Experiment Runner')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'timesformer'], help='Model type to train')
    parser.add_argument('--skip-train', action='store_true', help='Skip training step')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation step')
    args = parser.parse_args()

    # Step 1: Training
    if not args.skip_train:
        print(f"\n=== Training {args.model.upper()} model ===")
        train_args = ['--model', args.model]
        success = run_script(os.path.join('src', 'train.py'), train_args)
        if not success:
            print("❌ Training failed. Exiting.")
            return
    else:
        print("⏭️  Training step skipped.")

    # Step 2: Evaluation
    if not args.skip_eval:
        print(f"\n=== Evaluating {args.model.upper()} model ===")
        eval_args = ['--model', args.model]
        success = run_script(os.path.join('src', 'evaluate.py'), eval_args)
        if not success:
            print("❌ Evaluation failed.")
    else:
        print("⏭️  Evaluation step skipped.")

    print("\n✅ Experiment completed.")

if __name__ == "__main__":
    main()
