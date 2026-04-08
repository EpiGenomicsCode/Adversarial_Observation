import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from Adversarial_Observation.Swarm import PSO

@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3),
        nn.Flatten(),
        nn.Linear(4 * 26 * 26, 10)
    )

def dummy_cost_func(model, position):
    # A dummy cost function to satisfy the PSO requirements during testing
    return torch.sum(position).item()

@pytest.fixture
def test_data():
    # 5 particles, dimensions: (1, 28, 28)
    input_images = torch.rand((5, 1, 28, 28))
    return input_images

@pytest.fixture
def particle_swarm(simple_model, test_data):
    swarm = PSO(
        starting_positions=test_data,
        cost_func=dummy_cost_func,
        model=simple_model,
        minclamp=0.0,
        maxclamp=1.0
    )
    return swarm

def test_particle_swarm_initialization(particle_swarm):
    """Test if swarm initializes components correctly."""
    assert particle_swarm.epoch == 0
    assert len(particle_swarm.swarm) == 5
    assert particle_swarm.cos_best_g > -float('inf')

def test_pso_optimization(particle_swarm):
    """Test the full optimization loop and score improvement."""
    initial_score = particle_swarm.cos_best_g.clone()
    particle_swarm.run(epochs=2)
    
    # The score should have updated (or at least maintained) after evaluation
    assert particle_swarm.cos_best_g >= initial_score

def test_logging_creation(particle_swarm, tmp_path):
    """Verify history files are created in the temporary directory."""
    particle_swarm.run(epochs=2)
    save_path = str(tmp_path / 'history.csv')
    
    particle_swarm.save_history(save_path)
    assert os.path.exists(save_path)

def test_getters(particle_swarm):
    """Expand testing to ensure data extraction methods return expected types and shapes."""
    best_pos = particle_swarm.getBest()
    # PSO getBest returns a torch.Tensor, not an ndarray
    assert isinstance(best_pos, torch.Tensor)

    points = particle_swarm.getPoints()
    # PSO getPoints returns a vertically stacked torch.Tensor
    assert isinstance(points, torch.Tensor)
    assert points.shape[0] == 5