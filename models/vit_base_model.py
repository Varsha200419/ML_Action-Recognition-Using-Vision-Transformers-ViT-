from transformers import AutoModelForImageClassification, AutoConfig

def get_vit_model(num_classes=25):
    config = AutoConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    config.num_labels = num_classes
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        config=config,
        ignore_mismatched_sizes=True
    )
    return model
        self.vit_model.classifier = nn.Linear(
            self.vit_model.classifier.in_features, 
            num_classes
        )
        self.vit_model.num_labels = num_classes
        self.vit_model.config.num_labels = num_classes
        self.num_classes = num_classes

    def forward(self, pixel_values, labels=None, **kwargs):
        # If input is video, select middle frame
        if pixel_values.dim() == 5:
            middle = pixel_values.shape[1] // 2
            pixel_values = pixel_values[:, middle]
        return self.vit_model(pixel_values=pixel_values, labels=labels)

def get_vit_model(num_classes=25):  # Updated default to 25
    try:
        model = VideoViTModel(num_classes=num_classes)
        print(f"Video-adapted ViT model loaded successfully with {num_classes} classes")
        return model
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        raise
