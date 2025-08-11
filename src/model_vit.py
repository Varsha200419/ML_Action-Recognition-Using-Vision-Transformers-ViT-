# vit_base_model.py
# Vision Transformer (ViT) image classification model adapted for video classification

from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
import torch.nn as nn

class VideoViTModel(nn.Module):
    """ViT model adapted for video classification by processing the middle frame."""
    
    def __init__(self, num_classes=24):
        super().__init__()
        # Load pre-trained ViT model for image classification
        self.vit_model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            ignore_mismatched_sizes=True
        )
        # Replace the classifier layer with the correct number of classes
        self.vit_model.classifier = nn.Linear(
            self.vit_model.classifier.in_features, 
            num_classes
        )
        # Update the num_labels attribute to match our classes
        self.vit_model.num_labels = num_classes
        self.vit_model.config.num_labels = num_classes
        self.num_classes = num_classes
        
    def forward(self, pixel_values, labels=None, **kwargs):
        """
        Forward pass for video input.
        Args:
            pixel_values: Tensor of shape [batch_size, num_frames, channels, height, width]
                         or [batch_size, channels, height, width]
            labels: Optional labels for training
        Returns:
            Model outputs with logits and loss (if labels provided)
        """
        # Debug: Print input shape
        print(f"ViT input shape: {pixel_values.shape}, dims: {pixel_values.dim()}")
        
        # Handle both video (5D) and image (4D) inputs
        if pixel_values.dim() == 5:
            # Video input: [batch_size, num_frames, channels, height, width]
            batch_size, num_frames, channels, height, width = pixel_values.shape
            print(f"   Processing 5D video input: {batch_size}x{num_frames}x{channels}x{height}x{width}")
            
            # Take the middle frame for classification
            middle_frame_idx = num_frames // 2
            selected_frames = pixel_values[:, middle_frame_idx]  # [batch_size, channels, height, width]
            print(f"   Selected middle frame {middle_frame_idx}, output shape: {selected_frames.shape}")
        elif pixel_values.dim() == 4:
            # Image input: [batch_size, channels, height, width]
            print(f"   Processing 4D image input: {pixel_values.shape}")
            selected_frames = pixel_values
        else:
            raise ValueError(f"Expected 4D or 5D input, got {pixel_values.dim()}D with shape {pixel_values.shape}")
        
        # Process through ViT model
        print(f"   Passing to ViT: {selected_frames.shape}")
        outputs = self.vit_model(pixel_values=selected_frames, labels=labels, **kwargs)
        return outputs

def get_vit_model(num_classes=24):
    """Load ViT model adapted for video classification."""
    try:
        model = VideoViTModel(num_classes=num_classes)
        print(f"Video-adapted ViT model loaded successfully with {num_classes} classes")
        return model
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        raise
