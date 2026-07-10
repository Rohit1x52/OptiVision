## This file is responsible for helping the developer visually inspect, understand, and verify the dataset 
## by displaying images, class distributions, dimensions, and other visual characteristics before model training.

## Responsibilities
# Display Random Images
# Display Images from Each Class
# Plot Class Distribution
# Show Image Grid
# Show Image Dimensions

import os
import random

import matplotlib.pyplot as plt
from PIL import Image

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp"
)

class datasetVisualizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classs = sorted(
            [
                folder
                for folder in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, folder))
            ]
        )

    # Show Random Images
    def show_random_images(self, num_images=6):
        image_paths = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(VALID_EXTENSIONS):
                    image_paths.append(os.path.join(root, file))
        
        random.shuffle(image_paths)
        image_paths = image_paths[:num_images]
        cols = 3
        rows = (num_images + cols - 1) // cols

        plt.figure(figsize=(12,8))

        for i, path in enumerate(image_paths):
            image = Image.open(path)

            label = os.path.basename(os.path.dirname(path))

            plt.subplot(rows, cols, i+1)
            plt.imshow(image)
            plt.title(label)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Show Samples From Every Class
    def show_class_samples(self, samples_per_class=3):
        rows = len(self.classes)
        cols = samples_per_class
        plt.figure(figsize=(4 * cols, 4 * rows))
        plot_index = 1

        for class_name in self.classes:
            folder = os.path.join(self.dataset_path, class_name)

            images = [
                img
                for img in os.listdir(folder)
                if img.lower().endswith(VALID_EXTENSIONS)
            ]
            random.shuffle(images)
            images = images[:samples_per_class]

            for img_name in images:
                img_path = os.path.join(
                    folder, img_name
                )
                image = Image.open(img_path)

                plt.subplot(rows, cols, plot_index)
                plt.imshow(class_name)
                plt.axis("off")
                plt_index += 1

        plt.tight_layout()
        plt.show()

    # Plot Class Distribution
    def plot_class_distribution(self):
        class_counts = []
        for class_name in self.classes:
            folder = os.path.join(self.dataset_path, class_name)
            count = len([
                img 
                for img in os.listdir(folder)
                if img.lower().endswith(VALID_EXTENSIONS)
            ])
            class_counts[class_name] = count
        plt.figure(figsize=(10, 6))
        plt.bar(
            class_counts.keys(),
            class_counts.values()
        )
        plt.xlabel("classes")
        plt.ylabel("Number of Images")
        plt.title("images per class")
        plt.xticks(rotation=45)
        plt.tight_lauout()
        plt.show()

    # Display Images In Grid
    def show_grid(self, rows=3, cols=3):
        total = rows * cols
        image_paths = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(VALID_EXTENSIONS):
                    image_paths.append(os.path.join(root, file))
        random.shuffle(image_paths)
        image_paths = image_paths[:total]
        plt.figure(figsize=(12, 12))
        for i, path in enumerate(image_paths):
            image = Image.open(path)
            label = os.path.basename(os.path.dirname(path))
            plt.subplot(rows, cols, i+1)
            plt.imshow(image)
            plt.title(label)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Display Image Dimensions
    def show_image_dimensions(self, num_images=10):
        print("\nImage Dimensions")
        print("-" * 20)

        image_paths = []

        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(VALID_EXTENSIONS):
                    image_paths.append(os.path.join(root, file))
        random.shuffle(image_paths)
        image_paths = image_paths[:num_images]

        for path in image_paths:
            image = Image.open(path)
            label = os.path.basename(os.path.dirname(path))
            print(
                f"{label:20s}"
                f"{image.width} x {image.height}"
            )


