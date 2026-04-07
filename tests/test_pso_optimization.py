import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from Adversarial_Observation.Swarm import ParticleSwarm

@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3),
        nn.Flatten(),
        nn.Linear(4 * 26 * 26, 10)
    )

@pytest.fixture
def test_data():
    input_images = torch.rand((5, 1, 28, 28))
    starting_class = 0
    target_class = 9
    return input_images, starting_class, target_class

@pytest.fixture
def particle_swarm(simple_model, test_data, tmp_path):
    input_images, starting_class, target_class = test_data
    
    # Use pytest's built-in tmp_path fixture to avoid all permission errors
    save_dir = str(tmp_path / 'test_results')
    
    swarm = ParticleSwarm(
        model=simple_model,
        input_set=input_images,
        starting_class=starting_class,
        target_class=target_class,
        num_iterations=2,
        save_dir=save_dir,
        enable_logging=True
    )
    return swarm

def test_particle_swarm_initialization(particle_swarm):
    """Test if swarm initializes components correctly."""
    assert particle_swarm.num_iterations == 2
    assert particle_swarm.start_class == 0
    assert len(particle_swarm.particles) == 5
    assert particle_swarm.global_best_score == -float('inf')

def test_pso_optimization(particle_swarm):
    """Test the full optimization loop and score improvement."""
    initial_score = particle_swarm.global_best_score
    particle_swarm.optimize()
    
    # The score should have updated after evaluation
    assert particle_swarm.global_best_score > initial_score or particle_swarm.global_best_score > -float('inf')

def test_logging_creation(particle_swarm):
    """Verify log files are created in the temporary directory."""
    particle_swarm.optimize()
    log_path = os.path.join(particle_swarm.save_dir, 'iteration_log.log')
    assert os.path.exists(log_path)

def test_getters(particle_swarm):
    """Expand testing to ensure data extraction methods return expected types and shapes."""
    best_pos = particle_swarm.getBest()
    assert isinstance(best_pos, np.ndarray)

    points = particle_swarm.getPoints()
    assert isinstance(points, list)
    assert len(points) == 5
    assert isinstance(points[0], np.ndarray)