## Responsibilities 
# Load EfficientNet models
# Load pretrained ImageNet weights
# Replace the final classifier
# Optionally freeze the backbone
# Count trainable parameters
# Print model summary

import torch
import torch.nn as nn
from torchvision import models

class EfficientNetModel:

    def __init__(
        self, 
        model_name="effiecientnet_0",
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
        if self.model_name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT
                if self.self.pretrained
                else None
            ) 
            model = models.efficientnet_b0(
                weights=weights
            )
        elif self.model_name == "efficientnet_b1":
            weights = (
                models.EfficientNet_B1_Weights.DEFAULT
                if self.pretrained
                else None
            )
            model = models.efficientnet_b1(
                weights=weights
            )
        elif self.model_name == "efficientnet_b2":
            weights = (
                models.EfficientNet_B2_Weights.DEFAULT
                if self.pretrained
                else None
            )
            model = models.efficientnet_b2(
                weights=weights
            )
        elif self.model_name == "efficientnet_b3":
            weights = (
                models.EfficientNet_B3_Weights.DEFAULT
                if self.self.pretrained
                else None
            )
            model = models.efficientnet_b3(
                weights=weights
            )
        else:
            raise ValueError(
                f"Unsupported model :{self.model_name}"
            )
        
        # Freeze Backbone
        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace Final Classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features,
            self.num_classes
        )
        return model
    
    # Return Model
    def get_model(self):
        return self.model
    
    # Count Parameters
    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )
    
    # Print Summary
    def summary(self):
        print("-" * 20)
        print("EfficientNet Model Summary")
        print("-" * 60)

        print(f"Model : {self.model_name}")
        print(f"Pretrained : {self.pretrained}")
        print(f"Freeze Backbone : {self.freeze_backbone}")
        print(f"Output classes : {self.num_classes}")

        print(
            f"Trainable Params : "
            f"{self.count_parameters():,}"
        )

        print("-" * 20)