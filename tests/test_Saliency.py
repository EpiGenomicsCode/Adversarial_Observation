import unittest
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from Adversarial_Observation.Attacks import fgsm_attack as saliency_map  # Ensure this is correct

class TestSaliencyMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = models.resnet18(weights='DEFAULT').to(cls.device)  # Updated for deprecation
        cls.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        cls.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=cls.transform)

    def test_saliency_map_generation(self):
        # Load a batch of images directly
        images, _ = zip(*[self.testset[i] for i in range(4)])  # Get first 4 images
        images = torch.stack(images).to(self.device)  # Convert to tensor and move to device
        saliency_maps = saliency_map(images, self.model, (1, 3, 224, 224), 0)

        self.assertEqual(len(saliency_maps), len(images))

        for map_ in saliency_maps:
            self.assertEqual(map_.shape, (3, 224, 224))  # Check for 3 channels

    def test_plot_saliency_maps(self):
        os.makedirs('./tests/images/test_saliency_maps', exist_ok=True)
        # Load a batch of images directly
        images, _ = zip(*[self.testset[i] for i in range(4)])  # Get first 4 images
        images = torch.stack(images).to(self.device)  # Convert to tensor and move to device
        saliency_maps = saliency_map(images, self.model, (1, 3, 224, 224))

        for i, saliency_map_ in enumerate(saliency_maps):
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(images[i].cpu().permute(1, 2, 0).numpy())  # Original image
            axs[0].axis('off')

            # Take the mean of the channels for plotting, or choose a specific channel
            saliency_map_display = saliency_map_.cpu().mean(dim=0).numpy()  # Average over channels
            axs[1].imshow(saliency_map_display, cmap='hot')
            axs[1].axis('off')
            plt.savefig(f'./tests/images/test_saliency_maps/{i}_saliency_map.png')

if __name__ == '__main__':
    unittest.main()
