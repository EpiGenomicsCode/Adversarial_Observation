import pytest
import torch
import torch.nn as nn
import numpy as np
from Adversarial_Observation.BirdParticle import BirdParticle

@pytest.fixture
def simple_pytorch_model():
    """Create a simple PyTorch model to match source logic."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10)
    )

@pytest.fixture
def test_data():
    """Generate PyTorch tensors for testing."""
    input_data = torch.rand((1, 1, 28, 28))
    target_class = 3
    return input_data, target_class

@pytest.fixture
def bird_particle(simple_pytorch_model, test_data):
    input_data, target_class = test_data
    # Removed 'epsilon' as it is not in BirdParticle.__init__
    return BirdParticle(
        model=simple_pytorch_model,
        input_data=input_data,
        target_class=target_class,
        clip_value_position=0.2
    )

def test_bird_particle_initialization(bird_particle):
    """Test initialization against PyTorch source."""
    assert bird_particle.target_class == 3
    assert bird_particle.clip_value_position == 0.2
    assert torch.allclose(bird_particle.position, bird_particle.original_data)
    assert bird_particle.best_score == -np.inf

def test_velocity_update(bird_particle):
    """Test velocity update using PyTorch tensors."""
    initial_velocity = bird_particle.velocity.clone()
    global_best = torch.randn_like(bird_particle.position)
    
    bird_particle.update_velocity(global_best)
    
    assert not torch.allclose(initial_velocity, bird_particle.velocity)

def test_position_update(bird_particle):
    """Test position update and clamping."""
    # Manually set a velocity to force movement
    bird_particle.velocity = torch.ones_like(bird_particle.position) * 0.5
    bird_particle.update_position()
    
    # Check clamping (0.0 to 1.0) and clip_value_position
    assert torch.all(bird_particle.position <= 1.0)
    assert torch.all(bird_particle.position >= 0.0)