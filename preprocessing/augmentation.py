## Responsibilities
# Apply random transformations only during training
# Increase dataset diversity
# Reduce overfitting
# Simulate real-world variations

from torchvision import transforms

class DataAugmentation:

    def _init__(
        self,
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    # Training Transform 
    def get_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degress=20),
            transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, huw=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3), ##Simulates different camera viewpoints.
            transforms.GaussianBlur(kernel_size=3), #Slightly blurs the image.
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    # Display Augmentation Pipeline
    def show_pipeline(self):
        print("-" * 20)
        print("Training Data Augmentation")
        print("-" * 20)
        print("Resize")
        print("Random Horizontal Flip")
        print("Random Vertical Flip")
        print("Random Rotation")
        print("Random Resized Crop")
        print("Color Jitter")
        print("Random Perspective")
        print("Gaussian Blur")
        print("Convert to Tensor")
        print("Normalize")
        print("-" * 20)