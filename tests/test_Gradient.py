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
        self.model = models.resnet18(weights='DEFAULT').to(self.device)
        self.epsilon_values = [0, 0.01, 0.05, 0.1, 0.2]
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
        # Use the first 100 images for testing to ensure a larger sample size
        self.images = self.testset.data[:100]
        self.labels = self.testset.targets[:100]
        
        # Convert images to PIL Images and apply transformations
        self.images_tensor = []
        for img in self.images:
            img_pil = Image.fromarray(img)
            img_transformed = self.transform(img_pil).unsqueeze(0)
            self.images_tensor.append(img_transformed)

        self.images_tensor = torch.cat(self.images_tensor).to(self.device)

    def test_gradient_map_generation(self):
        gradient_maps = fgsm_attack(self.images_tensor, self.model, (1, 3, 224, 224), 0.02)

        self.assertEqual(len(gradient_maps), len(self.images_tensor), 
                         "Number of gradient maps should match number of images.")

        for map in gradient_maps:
            self.assertEqual(map.shape, (3, 224, 224),  # Check for (channels, height, width)
                             "Gradient map should have shape (3, 224, 224).")

    def test_plot_gradient_maps(self):
        os.makedirs('./tests/images/test_Gradient_maps', exist_ok=True)
        gradient_maps_equal = fgsm_attack(self.images_tensor, self.model, (1, 3, 224, 224), 0)
        gradient_maps_diff = fgsm_attack(self.images_tensor, self.model, (1, 3, 224, 224), 0.2)

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

    def test_accuracy_decrease_with_epsilon(self):
        original_labels = torch.tensor(self.labels).to(self.device)
        
        accuracies = []
        for epsilon in self.epsilon_values:
            # Get adversarial examples
            adv_images = fgsm_attack(self.images_tensor, self.model, (1, 3, 224, 224), epsilon)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs, 1)
                
            # Calculate accuracy
            accuracy = (predicted == original_labels).float().mean().item()
            accuracies.append(accuracy)

        # Assert that accuracy decreases with increasing epsilon
        for i in range(1, len(accuracies)):
            self.assertGreaterEqual(accuracies[i - 1], accuracies[i], 
                                    f"Accuracy should decrease as epsilon increases. "
                                    f"Expected: {accuracies[i - 1]} >= {accuracies[i]} (epsilon: {self.epsilon_values[i]})")

if __name__ == '__main__':
    unittest.main()
