## Responsibilities
# Trainer initialization
# Device setup
# Model initialization
# Loss function
# Optimizer
# Scheduler
# DataLoaders
# Training configuration

from typing import Optional
import torch
import torch.nn as nn
from training.optimizer import OptimizerFactory
from training.scheduler import SchedulerFactory

class Trainer:
    """
    Supports"
    -Any Pytorch Model
    -Any Optimizer
    -Any Scheduler
    -CPU/GPU
    """
    def __init__(
            self,
            model: nn.Module,
            train_loader,
            val_loader=None,
            test_loader=None,
            optimizer_name: str = "Adam",
            scheduler_name: str = "StepLR",
            criterion: Optional[nn.Module] = None,
            learning_rate: float = 1e-3,
            weight_decay: float = 0.0,
            momentum: float = 0.9,
            device: Optional[str] = None,
            num_epochs: int = 50,
    ):
        # Device
        self.device = self._setup_device(device)

        # Model
        self.model = model.to(self.device)

        # DataLoaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Training Configuration
        self.num_spochs = num_epochs

        # Loss Function
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.criterion = criterion

        # Optimizer
        self.optimizer = OptimizerFactory.create_optimizer(
            model=self.model,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        # Scheduler
        self.scheduler = SchedulerFactory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_name=scheduler_name
        )

        # Training Hostory
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_loss = float("inf")

        self.history = {
            "train_loss":[],
            "train_accuracy":[],
            "val_loss":[],
            "val_accuracy":[],
            "learning_rate":[],
        }

        print("-" * 30)
        print("Trainer Initialized Successfully")
        print("-"*30)
        print(f"Device : {self.device}")
        print(f"Epochs : {self.num_epochs}")
        print(f"Optimizer : {optimizer_name}")
        print(f"Scheduler : {scheduler_name}")
        print(f"Loss Function : {self.criterion.__class__.__name__}")

    ## Private Methods
    @staticmethod
    def _setup_device(device=None):
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        return torch.device("cpu")
    
    