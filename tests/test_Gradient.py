import unittest
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from Adversarial_Observation.Attacks import fgsm_attack  # Import your fgsm_attack function

class TestGradientMap(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights='DEFAULT').to(self.device)  # Updated for torchvision 0.13+
        self.epsilon = 0.1
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Define the transform pipeline for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        # Load CIFAR10 dataset directly without DataLoader
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=None)
        # Use the first 4 images for testing
        self.images = self.testset.data[:4]  # Get first 4 images
        self.labels = self.testset.targets[:4]  # Corresponding labels
        
        # Convert images to PIL Images and apply transformations
        self.images_tensor = []
        for img in self.images:
            img_pil = Image.fromarray(img)  # Convert to PIL Image
            img_transformed = self.transform(img_pil).unsqueeze(0)  # Apply transform and add batch dimension
            self.images_tensor.append(img_transformed)

        self.images_tensor = torch.cat(self.images_tensor).to(self.device)  # Combine into a single tensor

    def test_Gradient_map_generation(self):
        gradient_maps = fgsm_attack(self.images_tensor, self.model, (1, 3, 224, 224), .02)

        self.assertEqual(len(gradient_maps), len(self.images_tensor), 
                         "Number of gradient maps should match number of images.")

        for map in gradient_maps:
            self.assertEqual(map.shape, (3, 224, 224),  # Check for (channels, height, width)
                             "Gradient map should have shape (3, 224, 224).")

    def test_plot_Gradient_maps(self):
        os.makedirs('./tests/images/test_Gradient_maps', exist_ok=True)
        gradient_maps_equal = fgsm_attack(self.images_tensor, self.model, (1, 3, 224, 224), 0)
        gradient_maps_diff = fgsm_attack(self.images_tensor, self.model, (1, 3, 224, 224), .2)

        self.assertEqual(len(gradient_maps_equal), len(gradient_maps_diff),
                        "Number of gradient maps should match number of images.")

        for i in range(len(gradient_maps_equal)):
            equal_np = gradient_maps_equal[i].cpu().detach().numpy().transpose(1, 2, 0)  # Change shape to (224, 224, 3)
            diff_np = gradient_maps_diff[i].cpu().detach().numpy().transpose(1, 2, 0)    # Change shape to (224, 224, 3)

            original_np = self.images_tensor[i].cpu().detach().numpy()
            original_np = np.transpose(original_np, (1, 2, 0))  # Ensure original is (224, 224, 3)

            # Now check the shapes match
            self.assertTrue(np.allclose(original_np, equal_np, atol=1e-5), "Original image should be the same as equal_np.")
            self.assertFalse(np.array_equal(original_np, diff_np), "Original image should differ from diff_np.")

if __name__ == '__main__':
    unittest.main()
