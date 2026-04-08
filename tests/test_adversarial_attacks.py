import torch
import pytest
import numpy as np
from Adversarial_Observation.Attacks import fgsm_attack, gradient_ascent, gradient_map
from Adversarial_Observation.utils import load_MNIST_model, compute_success_rate

@pytest.fixture
def sample_data():
    model = load_MNIST_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_data = torch.rand((1, 1, 28, 28)) 
    return input_data, model

def test_fgsm_attack(sample_data):
    """Test standard FGSM attack modifies the image."""
    input_data, model = sample_data
    epsilon = 0.1 
    
    adversarial_data = fgsm_attack(input_data, model, (1, 1, 28, 28), epsilon)
    
    # Convert the returned list of numpy arrays back to a tensor for comparison
    adversarial_tensor = torch.tensor(np.array(adversarial_data))

    assert not torch.allclose(input_data.cpu(), adversarial_tensor.cpu(), atol=1e-5), "FGSM failed to perturb the input"
    
def test_success_rate():
    """Testing the success rate logic."""
    original_preds = torch.tensor([0, 1, 2, 3])
    adversarial_preds = torch.tensor([1, 0, 2, 3])

    success_rate = compute_success_rate(original_preds, adversarial_preds)
    assert success_rate == 0.5

def test_gradient_ascent(sample_data):
    """Test the gradient ascent method returns correct shapes and types."""
    input_data, model = sample_data
    
    adv_img = gradient_ascent(
        input_data, 
        model, 
        (1, 1, 28, 28), 
        target_neuron=3, # Changed from target_class to target_neuron
        num_iterations=2, 
        step_size=0.1
    )
    
    assert isinstance(adv_img, list)
    assert isinstance(adv_img[0], np.ndarray)
    assert adv_img[0].shape == (1, 28, 28) # Updated from (1, 1, 28, 28)

def test_gradient_map(sample_data):
    """Test that the gradient mapping extraction correctly averages channels."""
    input_data, model = sample_data
    
    g_map = gradient_map(input_data.numpy(), model, (1, 1, 28, 28))
    
    assert isinstance(g_map, list)
    assert isinstance(g_map[0], np.ndarray)
    assert g_map[0].shape == (28, 28) # Updated from (1, 28, 28)