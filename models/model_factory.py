## Responsibilities
# Create requested model
# Hide model implementation
# Provide one common interface
# Validate model names
# Centralize supported models

from models.mobilenet import MobileNetModel
from models.resnet import ResnetModel
from models.efficientnet import EfficientNetModel
from models.densenet import DenseNetModel

class ModelFactory:

    @staticmethod
    def create_model(
       model_name,
       num_classes,
       pretrained=True,
       freeze_backbone=False 
    ):
        
        # MobileNet
        if model_name in [
            "mobilenet_v3_small",
            "mobilenet_v3_large"
        ]:
            
            builder = MobileNetModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
            return builder.get_model()
        
        # ResNet
        elif model_name in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101"
        ]:
            builder = ResnetModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
            return builder.get_model()
        
        # EfficientNet
        elif model_name in  [
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3"
        ]:
            builder = EfficientNetModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
            return builder.get_model()
        
        # DenseNet
        elif model_name in [
            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201"
        ]:
            builder = DenseNetModel(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
            return builder.get_model()
        
        else:
            raise ValueError(
                f"Unsupported model : {model_name}"
            )
        
    # Supported Models
    @staticmethod
    def list_models():
        return [
            "mobilenet_v3_small",
            "mobilenet_v3_large",

            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",

            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",

            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201"
        ]
    

    