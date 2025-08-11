# timesformer_model.py
# TimeSFormer video classification model

from transformers import AutoModelForVideoClassification, AutoConfig
import torch.nn as nn

def get_timesformer_model(num_classes=25):
    """Load TimeSFormer pretrained model for video classification."""
    # Load the config and modify the number of labels
    config = AutoConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
    config.num_labels = num_classes
    
    # Load model with new config, ignoring mismatched classifier weights
    model = AutoModelForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        config=config,
        ignore_mismatched_sizes=True
    )
    return model
