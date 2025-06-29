# tests/test_adversarial_attacks.py

import torch
import pytest
from Adversarial_Observation.utils import fgsm_attack, load_MNIST_model
from torch import nn

# Helper function to generate random image data and a simple model
def get_sample_data():
    model = load_MNIST_model()
    # Fake image (1, 1, 28, 28) for MNIST (batch_size=1, channel=1, height=28, width=28)
    input_data = torch.rand((1, 1, 28, 28))  # Random image
    return input_data, model

def test_fgsm_attack():
    input_data, model = get_sample_data()
    epsilon = 0.1  # Perturbation size
    device = torch.device('cpu')  # Using CPU for simplicity

    # Apply FGSM attack
    adversarial_data = fgsm_attack(input_data, model, epsilon, device)

    # Check if adversarial data has been perturbed (should not be equal to original data)
    assert not torch.allclose(input_data, adversarial_data, atol=1e-5), "FGSM attack failed to perturb the input"
    
def test_success_rate():
    # Testing the success rate calculation
    original_preds = torch.tensor([0, 1, 2])  # Some dummy predictions
    adversarial_preds = torch.tensor([1, 0, 2])  # Adversarial predictions

    # Calculate success rate (should be 2/3 as two predictions are different)
    from Adversarial_Observation.utils import compute_success_rate
    success_rate = compute_success_rate(original_preds, adversarial_preds)
    
    assert success_rate == 2/3, f"Expected success rate 2/3, but got {success_rate}"

