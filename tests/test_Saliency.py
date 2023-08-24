import unittest
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Adversarial_Observation.Attacks import gradient_map as  saliency_map

class TestSaliencyMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = models.resnet18(pretrained=True).to(cls.device)
        cls.epsilon = 0.1
        cls.loss_fn = nn.CrossEntropyLoss()
        cls.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        cls.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=cls.transform)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=4,
                                                     shuffle=False, num_workers=2)
        cls.classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def test_saliency_map_generation(self):
        # Obtain the first batch of images
        dataiter = iter(self.testloader)
        images, _ = next(dataiter)

        # Generate saliency maps for the first batch of images
        saliency_maps = saliency_map(images, self.model, (1, 3, 224, 224))

        # Verify the shape of saliency maps
        self.assertEqual(len(saliency_maps), len(images),
                         f"Expected {len(images)} saliency maps, but got {len(saliency_maps)}")

        # Verify the shape and non-zero property of each saliency map
        for map_ in saliency_maps:
            # Verify shape
            self.assertEqual(map_.shape, (224, 224),
                             f"Expected (224, 224) saliency map, but got {map_.shape}")
            # Verify non-zero property
            self.assertTrue(np.any(map_)), "Expected non-zero saliency map"

    def test_plot_saliency_maps(self):
        os.makedirs('./tests/images/test_saliency_maps', exist_ok=True)
        # Obtain the first batch of images
        dataiter = iter(self.testloader)
        images, _ = next(dataiter)

        # Generate saliency maps for the first batch of images
        saliency_maps = saliency_map(images, self.model, (1, 3, 224, 224))

        # Plot the saliency maps for the first batch of images
        for i, saliency_map_ in enumerate(saliency_maps):
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(images[i].permute(1, 2, 0))
            axs[0].axis('off')
            axs[1].imshow(saliency_map_, cmap='hot')
            axs[1].axis('off')

            # Verify the successful save of the plot
            plt.savefig(f'./tests/images/test_saliency_maps/{i}_saliency_map.png')

if __name__ == '__main__':
    unittest.main()
