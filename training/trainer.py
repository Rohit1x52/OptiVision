## Responsibilities
# Trainer initialization
# Device setup
# Model initialization
# Loss function
# Optimizer
# Scheduler
# DataLoaders
# Training configuration
# Forward Pass
# Loss computation
# Backpropagation
# Gradient reset
# Optimizer parameter updates
# Batch accuracy computation
# Single-epoch training (train_one_epoch)
# Model evaluation mode
# Validation loss computation
# Validation accuracy computation
# Precision computation
# Recall computation
# F1-score computation
# Validation metrics collection
# Validation history logging


from typing import Optional
import torch
import torch.nn as nn
from training.optimizer import OptimizerFactory
from training.scheduler import SchedulerFactory
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    ## Forward Pass
    def _forward_pass(self, images):
        outputs = self.model(images)
        return outputs

    ## Loss Computation
    def _compute_loss(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss

    ## Backward Pass
    def _backward_pass(self, loss):
        loss.backward()

    ## Optimizer Step
    def _optimizer_step(self):
        self.optimizer.step()

    ## Gradient reset
    def _zero_grad(self):
        self.optimizer.zero_grad()

    ## Accuracy Calculation
    @staticmethod
    def _calculate_accuracy(outputs, labels):
        predictions = outputs.argmax(dim=1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        return correct, total

    ## Core Training Loop
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} / {self.num_epochs}",
            leave=False
        )
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            ## Reset Gradients
            self._zero_grad()

            ## Forward Pass
            outputs = self._forward_pass(images)

            ## Loss
            loss = self._compute_loss(outputs, labels)

            ## Backpropagation
            self._backward_pass(loss)

            ## Update Parameters
            self._optimizer_step()

            ## Statistics
            batch_Size = labels.size(0)
            running_loss += loss.item() * batch_Size
            correct, total = self._calculate_accuracy(outputs, labels)
            total_correct += correct
            total_sample += total
            avg_loss = running_loss / total_samples
            avg_accuracy = 100 * total_correct / total_samples
            progress_bar.set_postfix({
                "Loss" : f"{avg_loss:.4f}",
                "Acc": f"{avg_accuracy:.2f}%"
            })

        epoch_loss = running_loss / total_samples
        epoch_accuracy = 100 * total_correct / total_samples
        self.history["train_loss"].append(epoch_loss)
        self.history["train_accuracy"].append(epoch_accuracy)
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.history["learning_rate"].append(current_lr)
        return epoch_loss, epoch_accuracy

    ## Validation Method
    def validate(self):
        if self.val_loader is None:
           raise ValueError("Validation DataLoader is not provided") 

        ## Evaluation Mode
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets = []

        ## Disable Gradient Computation
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc="Validation",
                leave=False
            )
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                ## Forward Pass
                outputs = self._forward_pass(images)

                ## Loss
                loss = self._compute_loss(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                ## Predictions
                preds = outputs.argmax(dim=1)
                predictions.extend(
                    preds.cpu().numpy()
                )
                targets.extend(
                    labels.cpu().numpy()
                )

        ## Epoch Loss
        epoch_loss = running_loss / len(self.val_loader.dataset)

        ## Metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0
        )
        recall = recall_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0
        )
        f1 = f1_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0
        )
        metrics = {
            "loss": epoch_loss,
            "accuracy": accuracy * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1 * 100
        }

        ## Save History
        self.history["val_loss"].append(epoch_loss)
        self.history["val_accuracy"].append(metrics["accuracy"])
        return metrics
