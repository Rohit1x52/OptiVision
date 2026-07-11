## Responsibilities
# Resize images
# Center crop (optional)
# Normalize pixel values
# Provide separate pipelines for train, validation, and test

from torchvision import transforms

class ImagePreprocessor:

    def __init__(
            self,
            image_size=224,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

    # Train Transform
    def get_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    # Validate transform
    def get_validation_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    # Test Transform
    def get_test_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    # Display Current Configuration
    def show_configuration(self):
        print("-" * 20)
        print("Image preprocessing configuration")
        print("-" * 20)
        print(f"Image size : {self.image_size}")
        print(f"Mean : {self.mean}")
        print(f"std : {self.std}")
        print("-" * 20)
