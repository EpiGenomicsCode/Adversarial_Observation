import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import MagicMock
from Adversarial_Observation.BirdParticle import BirdParticle  # Assuming BirdParticle is in the BirdParticle module

@pytest.fixture
def simple_model():
    """Create a simple mock model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='softmax')  # Output 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@pytest.fixture
def test_data():
    """Generate mock input data for testing."""
    input_data = np.random.rand(28, 28).astype(np.float32)  # Single image of size 28x28
    target_class = np.random.randint(0, 10)  # Random target class (0-9)
    return tf.convert_to_tensor(input_data), target_class

@pytest.fixture
def bird_particle(simple_model, test_data):
    input_data, target_class = test_data
    return BirdParticle(
        model=simple_model,
        input_data=input_data,
        target_class=target_class,
        epsilon=0.8
    )

def test_bird_particle_initialization(bird_particle):
    """Test the initialization of the BirdParticle."""
    particle = bird_particle
    assert particle.target_class >= 0 and particle.target_class < 10
    assert particle.epsilon == 0.8
    assert np.allclose(particle.position.numpy(), particle.original_data.numpy())
    assert np.allclose(particle.best_position.numpy(), particle.original_data.numpy())
    assert particle.velocity is not None
    assert particle.best_score == -np.inf

def test_velocity_update(bird_particle):
    """Test the velocity update of the BirdParticle."""
    particle = bird_particle
    initial_velocity = particle.velocity.numpy()
    
    # Create a dummy global best position for the test
    global_best_position = tf.random.normal(shape=particle.position.shape)
    
    # Perform velocity update
    particle.update_velocity(global_best_position)
    
    updated_velocity = particle.velocity.numpy()
    
    # Check that velocity has been updated (it should not be the same)
    assert not np.allclose(initial_velocity, updated_velocity), "Velocity did not update"

def test_position_update(bird_particle):
    """Test the position update of the BirdParticle."""
    particle = bird_particle
    initial_position = particle.position.numpy()

    particle.velocity = tf.random.normal(shape=particle.position.shape) * 0.05  # Assign a small random velocity

    # Perform position update
    particle.update_position()

    updated_position = particle.position.numpy()

    # Check if position is still within bounds [0, 1]
    assert np.all(updated_position >= 0) and np.all(updated_position <= 1), "Position out of bounds"
    
    # Ensure that the position has changed
    assert not np.allclose(initial_position, updated_position), "Position did not change"

def test_velocity_clamp(bird_particle):
    """Test if the velocity is correctly clamped."""
    particle = bird_particle
    particle.velocity = tf.random.normal(shape=particle.velocity.shape) * 10  # Assign high velocity
    
    # Apply velocity update
    particle.update_velocity(particle.best_position)

    # Ensure velocity is within the clamp range
    assert np.all(np.abs(particle.velocity.numpy()) <= particle.velocity_clamp), "Velocity exceeded clamp range"

@pytest.mark.parametrize("initial_velocity", [None, np.zeros((28, 28))])
def test_velocity_initialization(bird_particle, initial_velocity):
    """Test if the velocity is initialized correctly."""
    particle = bird_particle
    if initial_velocity is not None:
        particle.velocity = tf.convert_to_tensor(initial_velocity)
    
    # Check if the velocity is correctly initialized
    assert np.allclose(particle.velocity.numpy(), initial_velocity if initial_velocity is not None else np.zeros((28, 28)))

