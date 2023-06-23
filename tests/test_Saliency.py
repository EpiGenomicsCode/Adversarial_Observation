import unittest
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from Adversarial_Observation.Attacks import saliency_map
import os
import numpy as np
import matplotlib.pyplot as plt

class TestSaliencyMap(unittest.TestCase):
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

    def test_saliency_map_generation(self):
        # Get the first batch of images
        dataiter = iter(self.testloader)
        images, _ = next(dataiter)

        # Generate the saliency map for the first batch of images
        saliency_maps = saliency_map(images, self.model, (1, 3, 224, 224))

        # Assert the shape of saliency maps
        self.assertEqual(len(saliency_maps), len(images)), f"Expected {len(images)} saliency maps, got {len(saliency_maps)} instead"

        # Assert the shape of each saliency map
        for map in saliency_maps:
            # assert shape
            self.assertEqual(map.shape, (224, 224)), f"Expected (224, 224) saliency map, got {map.shape} instead"
            # assert they are not zero
            self.assertTrue(np.any(map)), f"Expected non-zero saliency map, got {map} instead"
            #
            


    def test_plot_saliency_maps(self):
        os.makedirs('./tests/images/test_saliency_maps', exist_ok=True)
        # Get the first batch of images
        dataiter = iter(self.testloader)
        images, _ = next(dataiter)

        # Generate the saliency map for the first batch of images
        saliency_maps = saliency_map(images, self.model, (1, 3, 224, 224))

        # Plot the saliency maps for the first batch of images
        for i in range(len(saliency_maps)):
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(images[i].permute(1, 2, 0))
            axs[0].axis('off')
            axs[1].imshow(saliency_maps[i], cmap='hot')
            axs[1].axis('off')

            # Assert the plot is saved successfully
            plt.savefig(f'./tests/images/test_saliency_maps/{i}_saliency_map.png')

if __name__ == '__main__':
    unittest.main()
