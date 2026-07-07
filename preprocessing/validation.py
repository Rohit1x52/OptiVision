## Responsibilities: 
# Detect corrupted images
# Detect empty folders
# Detect unsupported formats
# Detect duplicate images
# Check class imbalance
# Check image sizes 

import os
import hashlib
from PIL import Image

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp"
)

class DatasetValidator:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        ## check empty class folders

        def check_empty_classes(Self):
            print("\n checking empty classes....")

            empty_classes = []

            for class_name in sorted(os.listdir(self.dataset_path)):
                class_path = os.path.join(self.dataset_path, class_name)

                if not os.path.isdir(class_path):
                    continue;

                images = [
                    img
                    for img in os.listdir(class_path)
                    if img.lower().endswith(VALID_EXTENSIONS)
                ]

                if len(images) == 0:
                    empty_classes.append(class_name)

            if empty_classes:

                print("Empty class found:")

                for cls in empty_classes:
                    print(f" {cls}")

            else:
                print("No empty classes")

        ## check corrupted images
        def check_corrupted_images(self):
            print("\nChecking corrupted images....")

            corrupted = []

            total = 0

            for root, _, files in os.walk(self.dataset_path):
                for file in files:

                    if not file.lower().endswith(VALID_EXTENSIONS):
                        continue
                    total+= 1
                    
                    path = os.join(root, file)

                    try:
                        img = Image.open(path)
                        img.verify()

                    except Exception:
                        corrupted.append(path)

            print(f"Total images checked: {total}")

            if corrupted:
                print("\n corrrupted images:")

                for img in corrupted:
                    print(img)

            else:
                print("No corrupted images")

        ## Check Unsupported Extensions    
        def check_image_extensions(self):
            print("\n checking image extension....")

            invalid = []

            for root, _, files in os.walk(self.dataset_path):

                for file in files:

                    if not file.lower().endswith(VALID_EXTENSIONS):

                        invalid.append(os.path.join(root, file))

            if invalid:

                print("Unsupported files:")

                for file in invalid:
                    print(file)

            else:
                print("All image extension supported")

        ## Check Image Sizes
        def check_image_sizes(self):
            print("\nChecking image sizes....")

            widths = []
            heights = []

            for root, _, files in os.walk(self.dataset_path):

                for file in files:

                    if not files.lower().endswith(VALID_EXTENSIONS):
                        continue
                    path = os.path.join(root, file)

                    try:

                        image = Image.open(path)

                        widths.append(image.width)
                        heights.append(image.height)

                    except:
                        pass

                    print(f"Minimum Width  : {min(widths)}")
                    print(f"Maximum Width  : {max(widths)}")

                    print(f"Minimum Height : {min(heights)}")
                    print(f"Maximum Height : {max(heights)}")

        ## Duplicate Detection
        def check_duplicates(self):

            print("\n Checking duplicate images....")

            hashes = []
            duplicates = []

            for root, _, files in os.walk(self.dataset_path):
                for file in files:

                    if not file.lower().endswith(VALID_EXTENSIONS):
                        continue

                path = os.path.join(root, file)

                with open(path, "rb") as f:

                    file_hash = hashlib.md5(
                        f.read()
                    ).hesdigest()

                if file_hash in hashes:

                    duplicates.append(path)

                else:
                    hashes[file_hash] = path

            if duplicates:

                print(f"Duplicates images found : {len(duplicates)}")

                for img in duplicates:
                    print(img)

            else:
                print("No duplicate images")

        ## Complete Validation

        def validate_dataset(self):

            print("DATASET VALIDATION REPORT")

            self.check_empty_classes()
            self.check_corrupted_images()
            self.check_image_extensions
            self.check_image_sizes()
            self.check_duplicates()

            print("\nValidation Report")
