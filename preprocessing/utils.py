## Responsibilities
# Set random seeds for reproducibility
# Detect CPU/GPU
# Create directories
# Load and save images
# Save and load model checkpoints
# Count model parameters
# Print project information

import os 
import random
import cv2
import numpy as np
import torch

# Set Random Seed
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed Set : {seed}")

# Create Directory
def create_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"Directory Ready : {path}")

# Load Image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found : {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Save Image
def save_image(image, save_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)
    print(f"Image Saved : {save_path}")

# set device
def set_device():
    if torch.cude.is_available():
        device = torch.device("cuda")
        print("Using GPU----")
    else:
        device = torch.device("cpu")
        print("Using CPU----")

# Save Model Checkpoint
def save_checkpoints(model, optimizer, epoch, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint Saved : {path}")

# Load Model Checkpoint
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint Loaded : {path}")
    return model, optimizer, epoch

# Count Trainable Parameters
def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.patrameters()
        if p.requires_grad
    )
    print(f"Trainable Prameters : {total:,}")
    return total

# Project Banner
def print_project_info():
    print("-" * 20)
    print("OptiVision")
    print("End-to-End Computer Vision Optimization Platform")
    print("-" * 20)
