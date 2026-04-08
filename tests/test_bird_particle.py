import pytest
import torch
import numpy as np
from Adversarial_Observation.BirdParticle import BirdParticle

@pytest.fixture
def test_data():
    """Generate PyTorch tensors for testing."""
    return torch.rand((1, 1, 28, 28))

@pytest.fixture
def bird_particle(test_data):
    # Initialize with the actual parameters expected by BirdParticle.__init__
    return BirdParticle(
        position=test_data,
        minclamp=0.0,
        maxclamp=0.2
    )

def test_bird_particle_initialization(bird_particle, test_data):
    """Test initialization against PyTorch source."""
    assert bird_particle.minclamp == 0.0
    assert bird_particle.maxclamp == 0.2
    assert torch.allclose(bird_particle.position_i, test_data)
    assert bird_particle.cost_best_i == -1

def test_velocity_update(bird_particle):
    """Test velocity update using PyTorch tensors."""
    initial_velocity = bird_particle.velocity_i.clone()
    global_best = torch.randn_like(bird_particle.position_i)
    
    bird_particle.update_velocity(global_best)
    
    assert not torch.allclose(initial_velocity, bird_particle.velocity_i)

def test_position_update(bird_particle):
    """Test position update and clamping."""
    # Manually set a velocity to force movement
    bird_particle.velocity_i = torch.ones_like(bird_particle.position_i) * 0.5
    bird_particle.update_position()
    
    # Check clamping using maxclamp (0.2) and minclamp (0.0)
    assert torch.all(bird_particle.position_i <= 0.2)
    assert torch.all(bird_particle.position_i >= 0.0)