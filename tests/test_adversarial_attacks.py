import torch
import pytest
import numpy as np
from Adversarial_Observation.Attacks import fgsm_attack, gradient_ascent, gradient_map
from Adversarial_Observation.utils import load_MNIST_model, compute_success_rate

@pytest.fixture
def sample_data():
    model = load_MNIST_model()
    # Ensure the model is moved to the GPU if the system has one
    # This matches the automatic device assignment in Attacks.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Fake image (1, 1, 28, 28) for MNIST (batch_size, channel, height, width)
    input_data = torch.rand((1, 1, 28, 28)) 
    return input_data, model

def test_fgsm_attack(sample_data):
    """Test standard FGSM attack modifies the image."""
    input_data, model = sample_data
    epsilon = 0.1 
    
    adversarial_data = fgsm_attack(input_data, model, (1, 1, 28, 28), epsilon)

    # Move back to CPU for assertion if needed
    assert not torch.allclose(input_data.cpu(), adversarial_data.cpu(), atol=1e-5), "FGSM failed to perturb the input"
    
def test_success_rate():
    """Testing the success rate logic."""
    original_preds = torch.tensor([0, 1, 2, 3])
    adversarial_preds = torch.tensor([1, 0, 2, 3])

    # 2 out of 4 changed
    success_rate = compute_success_rate(original_preds, adversarial_preds)
    assert success_rate == 0.5

def test_gradient_ascent(sample_data):
    """Test the gradient ascent method returns correct shapes and types."""
    input_data, model = sample_data
    
    adv_img = gradient_ascent(
        input_data, 
        model, 
        (1, 1, 28, 28), 
        target_class=3, 
        num_iterations=2, 
        step_size=0.1
    )
    
    assert isinstance(adv_img, np.ndarray)
    assert adv_img.shape == (1, 1, 28, 28)

def test_gradient_map(sample_data):
    """Test that the gradient mapping extraction correctly averages channels."""
    input_data, model = sample_data
    
    g_map = gradient_map(input_data.numpy(), model, (1, 1, 28, 28))
    
    assert isinstance(g_map, np.ndarray)
    assert g_map.shape == (1, 28, 28)  # Averaged over the 1 channel