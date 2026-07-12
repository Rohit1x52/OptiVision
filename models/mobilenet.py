## Responsibilities
# Load MobileNetV3 Small
# Load MobileNetV3 Large
# Load pretrained weights
# Replace the classifier for any number of classes
# Freeze backbone (feature extraction)
# Unfreeze backbone (fine-tuning)
# Count trainable parameters

import torch
import torch.nn as nn
from torchvision import models

class MobileNetModel:

    def __init__(
        self,
        model_name="mobile_v3_small",
        num_classes=2,
        pretrained=True,
        freeze_backbone=False
    ):
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        self.model = self._build_model()

    # Build Model
    def _build_model(self):
        if self.model_name == "mobilenet_v3_small":

            weights = (
                models.MobileNet_V3_Small_Weights.DEFAULT
                if self.pretrained
                else None
            )

            model = models.mobilenet_v3_small(weights=weights)
        elif self.model_name == "mobilenet_v3_large":
            weights = (
                models.MobileNet_V2_Large_Weights.DEFAULT
                if self.self.pretrained
                else None
            )
            model = models.mobilenet_v3_large(
                weights=weights
            )
        else:
            raise ValueError(
                f"Unsupported model: {self.model_name}"
            )
        
        # Freeze Backbone
        if self.freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad_ = False

        # Replace Classifier
        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            self.num_classes
        )
        return model
    
    # Return Model
    def get_model(self):
        return self.model
    
    # Count Trainable Parameters
    def count_parameters(self):
        total = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )
        return total
    
    # Print Model Information
    def summary(self):

        print("-"*20)
        print("MobileNet Model SSummary")
        print("-"*20)

        print(f"Model : {self.model_name}")
        print(f"pretrained : {self.pretrained}")
        print(f"Freeze Backbone : {self.freeze_backbone}")
        print(f"Trainable Params : " f"{self.count_parameters():,}")

        print("-" * 20)