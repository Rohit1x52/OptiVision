## This module is responsible for understanding the dataset before training. 
## In production ML, data analysis is a separate step from validation because validation checks whether the data is usable, while statistics help you understand the data.

import os
from PIL import Image

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp"
}

class DatasetStatistics:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.class_counts = {}

    # Count Images Per Class
    def count_images(self):
        self.class_counts = {}
        total_images = 0

        for class_name in sorted(os.listdir(self.dataset_path)):

            class_path = os.path.join(self.dataset_path, class_name)

            if not os.path.isdir(class_path):
                continue

            images = [
                img 
                for img in os.listdir(class_path)

                if img.lower().endswith(VALID_EXTENSIONS)
            ]

            self.class_counts[class_name] = len(images)
            total_images += len(images)

        return total_images
    

    # Print Class Distribution
    def class_distribution(self):

        if not self.class_counts:
            self.count_images()

        print("n\Class Distribution")

        for class_name, count in self.class_counts.items():
            print(f"{class_name:20s}: {count}")

    # Average Image Size
    def average_image_size(self):
        total_width = 0
        total_height = 0
        total_images = 0

        for root, _, files in os.walk(self.dataset_path):

            for file in files:
                if not file.lower().endswith(VALID_EXTENSIONS):
                    continue

                path = os.path.join(root, file)

                try:
                    image = Image.open(path)
                    total_width += image.width
                    total_height += image.height

                    total_images += 1

                except:
                    pass

        avg_width = total_width / total_images
        avg_height = total_height / total_images

        return avg_height, avg_width
    
    # Complete Dataset Summary
    def dataset_summary(self):
        total_images = self.count_images()
        avg_width, avg_height = self.average_image_size()

        largest_class = max(
            self.class_counts,
            key=self.class_counts.get
        )

        smallest_class = min(
            self.class_counts,
            key=self.class_counts.get
        )

        print("\n" + "-" * 10)
        print("DATASET SUMMARY")
        print("-" * 50)

        print(f"Classes         : {len(self.class_counts)}")
        print(f"Images          : {total_images}")

        print(f"Average Width   : {avg_width:.2f}")
        print(f"Average Height  : {avg_height:.2f}")

        print(f"Largest Class   : {largest_class}")
        print(f"Smallest Class  : {smallest_class}")

        print("=" * 50)

        self.class_distribution()
