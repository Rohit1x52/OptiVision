## Responsibilities 
# Loading DenseNet models
# Loading pretrained ImageNet weights
# Replacing the classifier
# Freezing the backbone (optional)
# Counting trainable parameters
# Printing model summary

import torch
import torch.nn as nn
from torchvision import models

class DenseNetModel:
    def __init__(
        self,
        model_name="densenet121",
        num_classes = 2,
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
        if self.model_name == "densenet121":
            weights = (
                models.DenseNet121_Weights.DEFAULT
                if self.self.pretrained
                else None
            )
            model = models.densenet121(
                weights=weights
            )
        elif self.model_name == "densenet161":
            weights = (
                models.DenseNet161_weights.DEFAULT
                if self.self.pretrained
                else None
            )
            model = models.densenet161(
                weights=weights
            )
        elif self.model_name == "densenet169":
            weights = (
                models.DenseNet169_Weights.DEFAULT
                if self.self.pretrained
                else None
            )
            model = models.densenet169(
                weights=weights
            )
        elif self.model_name == "densenet201":
            weights = (
                models.DenseNet201_Weights.DEFAULT
                if self.pretrained
                else None
            )
            model = models.densenet201(
                weights=weights
            )
        else:
            raise ValueError(
                f"Unsupported model : {self.model_name}"
            )
        
        # Freeze Backbone
        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace Classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(
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
    
    # Print Model Summary
    def summary(self):
        print("-" * 20)
        print("DenseNet Model Summary")
        print("-" * 20)

        print(f"Model              : {self.model_name}")
        print(f"Pretrained         : {self.pretrained}")
        print(f"Freeze Backbone    : {self.freeze_backbone}")
        print(f"Output Classes     : {self.num_classes}")

        print(
            f"Trainable Params   : "
            f"{self.count_parameters():,}"
        )

        print("-" * 20)


        