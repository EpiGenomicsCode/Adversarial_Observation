import unittest
import torch
import torch.nn as nn
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
        cls.model = models.resnet18(pretrained=True).to(cls.device)
        cls.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        cls.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=cls.transform)
        cls.testloader = torch.utils.data.DataLoader(cls.testset, batch_size=4,
                                                     shuffle=False, num_workers=2)

    def test_saliency_map_generation(self):
        dataiter = iter(self.testloader)
        images, _ = next(dataiter)
        saliency_maps = saliency_map(images, self.model, (1, 3, 224, 224),0)

        self.assertEqual(len(saliency_maps), len(images))

        for map_ in saliency_maps:
            self.assertEqual(map_.shape, (224, 224))
            self.assertTrue(np.any(map_))

    def test_plot_saliency_maps(self):
        os.makedirs('./tests/images/test_saliency_maps', exist_ok=True)
        dataiter = iter(self.testloader)
        images, _ = next(dataiter)
        saliency_maps = saliency_map(images, self.model, (1, 3, 224, 224))

        for i, saliency_map_ in enumerate(saliency_maps):
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(images[i].permute(1, 2, 0))
            axs[0].axis('off')
            axs[1].imshow(saliency_map_, cmap='hot')
            axs[1].axis('off')
            plt.savefig(f'./tests/images/test_saliency_maps/{i}_saliency_map.png')

if __name__ == '__main__':
    unittest.main()
