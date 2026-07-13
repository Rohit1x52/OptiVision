## Responsibilities
# ResNet18
# ResNet34
# ResNet50
# ResNet101
# Pretrained weights
# Freeze backbone
# Replace final classification layer
# Count trainable parameters
# Print model summary

import torch
import torch.nn as nn 
from torchvision import models

class ResnetModel:

    def __init__(
        self, 
        model_name="resnet18",
        num_classes=2,
        pretrained=True,
        freeze_backbone=False
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        self.model = self._build_model()

    # build model
    def _build_model(self):
        if self.model_name == "resnet18":
            weights = (
                models.Resnet18_Weights.DEFAULT
                if self.self.pretrained
                else None
            )
            model = models.resnet18(
                weights=weights
            )
        elif self.model_name =="resnet34":
            wights = (
                models.ResNet34_Weights.DEFAULT
                if self.pretrained
                else None
            )
            model = models.resnet34(
                weights=weights
            )
        elif self.model_name == "resnet50":
            weights = (
                models.ResNet50_Weights.DEFAULT
                if self.pretrained
                else None
            )
            model = models.resnet50(
                weights=weights
            )
        elif self.model_name == "resnet101":
            weights = (
                models.ResNet101_Weights.DEFAULT
                if self.pretrained
                else None
            )
            model = models.resnet101(
                weights=weights
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Freeze Backbone
        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace Final Layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(
            in_features, 
            self.num_classes
        )
        return model
    
    # Return Model
    def get_model(self):
        return self.model
    
    # Count Trainable Parameters
    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )
    
    # Print Summary
    def summary(self):

        print("-"* 20)
        print("ResNet Model Summary")
        print("-"* 20)

        print(f"Model : {self.model_name}")
        print(f"Pretrained : {self.pretrained}")
        print(f"Freeze Backbone : {self.freeze_backbone}")
        print(f"Output Classes : {self.num_classes}")
        print(
            f"Trainable Params :"
            f"{self.count_parameters():,}"
        )

        print("-" * 20)
        