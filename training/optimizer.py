## Responsibilities of optimizer.py
# Creating the optimizer
# Supporting multiple optimizers
# Setting learning rate
# Setting weight decay
# Setting momentum (for SGD)
# Returning the optimizer object

import torch.optim as optim

class OptimizerFactory:

    @staticmethod
    def create_optimizer(
        model,
        optimizer_name="Adam",
        learning_rate=1e-3,
        weight_decay=0.0,
        momentum=0.9
    ):
        optimizer_name = optimizer_name.lower()

        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )

        elif optimizer_name == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(
                model.optimizer(),
                lr=learning_rate,
                momentum=momentum
            )
        
        else:
            raise ValueError(
                f"Unsupported Optimizer : {optimizer_name}"
            )
        return optimizer
    
    # Supported Optimizers
    @staticmethod
    def list_optimizer():
        return [
            "SGD", "Adam", "AdamW", "RMSprop"
        ]
    
