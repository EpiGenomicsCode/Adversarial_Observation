from Adversarial_Observation.utils import *
from Adversarial_Observation.Attacks import *
import unittest
import torch

class TestFGSMAttack(unittest.TestCase):

    def test_fgsm_attack_shape(self):
        # Build a CNN
        model = buildCNN(10)
        model.eval()

        # generate a random input 1 image with 1 channel of size 28x28
        input_data = torch.rand(size=(1, 1, 28,28))
        
        # Get the activation map
        perturbed = fgsm_attack(input_data, 0, 0.1, model)
        
        # Assert they have the same shape
        assert perturbed.shape == input_data.shape


class TestActivationMap(unittest.TestCase):

    def test_activation_map_shape(self):
        # Build a CNN
        model = buildCNN(10)
        model.eval()

        # generate a random input 1 image with 1 channel of size 28x28
        input_data = torch.rand(size=(1, 1, 28,28))
        
        # Get the activation map
        activation = activation_map(input_data, model)
        
        # Assert they have the same shape
        assert activation.shape == input_data.shape

    # check assertion that input data has batch dimension of 1
    def test_activation_map_batch(self):
        # Build a CNN
        model = buildCNN(10)
        model.eval()

        # generate a random input 1 image with 1 channel of size 28x28
        input_data = torch.rand(size=(2, 1, 28,28))
        
        # this should raise an assertion error
        with self.assertRaises(AssertionError):
            activation = activation_map(input_data, model)
