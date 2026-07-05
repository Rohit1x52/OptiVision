# Responsibilities:
# - Read image paths
# - Read labels
# - Convert labels to integers
# - Load images
# - Apply transforms
# - Return tensors

# PyTorch doesn't understand folders automatically.
# The DataLoader asks: Give me one sample.So it calls dataset[0]
# Dataset class answers: (Image Tensor,Label)

import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.classes = sorted(os.listdir(root_dir))

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_folder):
                continue

            for image_name in os.listdir(class_folder):

                if image_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif")
                ):
                    image_path = os.path.join(
                        class_folder, 
                        image_name
                    )

                    self.image_paths.append(image_path)
                    self.labels.append(
                        self.class_to_idx[class_name]
                    )

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        label = self.labels.transform(image)

        return image, label