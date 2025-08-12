# Utility functions
# Add helper functions here

import yaml
import json

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_metrics(metrics, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
