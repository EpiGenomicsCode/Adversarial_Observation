import unittest
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from Adversarial_Observation.Attacks import gradient_map 
import os
import numpy as np
import matplotlib.pyplot as plt

class TestGradientMap(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True).to(self.device)
        self.epsilon = 0.1
        self.loss_fn = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4,
                                                      shuffle=False, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def test_Gradient_map_generation(self):
        # Get the first batch of images
        dataiter = iter(self.testloader)
        images, _ = next(dataiter)

        # Generate the Gradient map for the first batch of images
        gradient_maps = gradient_map(images, self.model, (1, 3, 224, 224))

        # Assert the shape of Gradient maps
        self.assertEqual(len(gradient_maps), len(images)), f"Expected {len(images)} Gradient maps, got {len(gradient_maps)} instead"

        # Assert the shape of each Gradient map
        for map in gradient_maps:
            # assert shape
            self.assertEqual(map.shape, (224, 224)), f"Expected (224, 224) Gradient map, got {map.shape} instead"
            # assert they are not zero
            self.assertTrue(np.any(map)), f"Expected non-zero Gradient map, got {map} instead"
            #
            


    def test_plot_Gradient_maps(self):
        os.makedirs('./tests/images/test_Gradient_maps', exist_ok=True)
        # Get the first batch of images
        dataiter = iter(self.testloader)
        images, _ = next(dataiter)

        # Generate the Gradient map for the first batch of images
        gradient_maps = gradient_map(images, self.model, (1, 3, 224, 224))

        # Plot the Gradient maps for the first batch of images
        for i in range(len(gradient_maps)):
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(images[i].permute(1, 2, 0))
            axs[0].axis('off')
            axs[1].imshow(gradient_maps[i], cmap='hot')
            axs[1].axis('off')

            # Assert the plot is saved successfully
            plt.savefig(f'./tests/images/test_Gradient_maps/{i}_Gradient_map.png')

if __name__ == '__main__':
    unittest.main()
