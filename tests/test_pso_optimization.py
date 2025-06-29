import pytest
import tensorflow as tf
import numpy as np
import os
from unittest.mock import MagicMock
from Adversarial_Observation.Swarm import ParticleSwarm  # Assuming ParticleSwarm is in the ParticleSwarm module

@pytest.fixture
def simple_model():
    """A simple mock model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='softmax')  # Output 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@pytest.fixture
def test_data():
    """Generate some mock data for testing."""
    # Create 5 random images (28x28) and random target class labels (between 0 and 9)
    input_images = np.random.rand(5, 28, 28).astype(np.float32)
    target_class = np.random.randint(0, 10)  # Target class for attack
    return input_images, target_class

@pytest.fixture
def particle_swarm(simple_model, test_data):
    input_images, target_class = test_data
    # Initialize the ParticleSwarm object
    swarm = ParticleSwarm(
        model=simple_model,
        input_set=input_images,
        target_class=target_class,
        num_iterations=2,  # Just a couple of iterations for testing
        save_dir='./test_results'  # Use a temporary directory for testing
    )
    return swarm

def test_particle_swarm_initialization(particle_swarm):
    """Test the initialization of ParticleSwarm class."""
    swarm = particle_swarm
    assert swarm.num_iterations == 2
    assert swarm.epsilon == 0.8
    assert len(swarm.particles) == 5  # Number of images in the test data
    assert swarm.save_dir == './test_results'

def test_pso_logging(particle_swarm, caplog):
    """Test logging during Particle Swarm Optimization."""
    swarm = particle_swarm
    with caplog.at_level('INFO'):
        swarm.log_progress(0)  # Log progress for the first iteration
    
    # Check if the logger captured the expected output
    assert "Iteration 1/2" in caplog.text
    assert "Particle" in caplog.text
    assert "Original Pred" in caplog.text

def test_pso_optimization(particle_swarm, caplog):
    """Test the optimization process."""
    swarm = particle_swarm
    
    # Capture the logs during optimization
    with caplog.at_level('INFO'):
        swarm.optimize()  # Run the optimization process
    
    # Check if the logs contain iteration details
    assert "Iteration 1/2" in caplog.text
    assert "Iteration 2/2" in caplog.text
    assert "Perturbed Pred" in caplog.text

def test_perturbed_images_saved(particle_swarm):
    """Test if perturbed images are saved correctly."""
    swarm = particle_swarm
    
    # Run one iteration of the optimization to generate and save images
    swarm.save_images(0)
    
    # Check if images are saved in the correct directory
    iteration_dir = os.path.join(swarm.save_dir, "iteration_1")
    assert os.path.exists(iteration_dir)
    
    # Check if files are saved in the directory
    files = os.listdir(iteration_dir)
    assert len(files) == 5  # One image per particle
    assert all(f.endswith('.png') for f in files)

def test_particle_update(particle_swarm):
    """Test the particle position and velocity updates."""
    swarm = particle_swarm
    initial_position = swarm.particles[0].position.numpy()
    
    # Simulate the evaluation and update of the first particle
    swarm.particles[0].evaluate()
    swarm.particles[0].update_velocity(swarm.global_best_position)
    swarm.particles[0].update_position()
    
    # Ensure the position has changed after update (it should not be the same)
    updated_position = swarm.particles[0].position.numpy()
    assert not np.allclose(initial_position, updated_position), "Particle position did not update"

@pytest.mark.parametrize("iterations", [5, 10])  # Parametrize for different iteration counts
def test_varying_iterations(simple_model, test_data, iterations):
    """Test the ParticleSwarm with different iteration counts."""
    input_images, target_class = test_data
    swarm = ParticleSwarm(
        model=simple_model,
        input_set=input_images,
        target_class=target_class,
        num_iterations=iterations,
        save_dir='./test_results'
    )
    
    # Run optimization for the specified number of iterations
    swarm.optimize()
    
    # Check if the optimization completed the specified number of iterations
    assert f"Iteration {iterations}" in open(os.path.join(swarm.save_dir, "iteration_log.log")).read()

