import tensorflow as tf
import numpy as np

class BirdParticle:
    """
    Represents a particle in the Particle Swarm Optimization (PSO) algorithm for adversarial attacks.
    """

    def __init__(self, model: tf.keras.Model, input_data: tf.Tensor, target_class: int, epsilon: float,
                 velocity: tf.Tensor = None, inertia_weight: float = 0.5, 
                 cognitive_weight: float = 1.0, social_weight: float = 1.0, momentum: float = 0.9,
                 velocity_clamp: float = 0.1):
        """
        Initialize a particle in the PSO algorithm.
        """
        self.model = model
        self.target_class = target_class
        self.epsilon = epsilon
        self.best_position = tf.identity(input_data)  # Store initial position as best
        self.best_score = -np.inf
        self.position = input_data  # Directly use input_data
        self.velocity = velocity if velocity is not None else tf.zeros_like(input_data)
        
        # PSO algorithm parameters
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.momentum = momentum
        self.velocity_clamp = velocity_clamp
        
        # To store position history over epochs
        self.history = [tf.identity(input_data)]  # Initialize history with the starting position

    def fitness(self) -> float:
        """
        Compute the fitness score for the particle, which is the softmax probability of the target class.
        """
        output = self.model(self.position)  # Add batch dimension and pass through the model
        probabilities = tf.nn.softmax(output, axis=1)  # Get probabilities for each class
        target_prob = probabilities[:, self.target_class]  # Target class probability
        return target_prob.numpy().item()  # Return the target class probability as fitness score

    def update_velocity(self, global_best_position: tf.Tensor) -> None:
        """
        Update the velocity of the particle based on the PSO update rule.
        """
        inertia = self.inertia_weight * self.velocity
        cognitive = self.cognitive_weight * tf.random.uniform(self.position.shape) * (self.best_position - self.position)
        social = self.social_weight * tf.random.uniform(self.position.shape) * (global_best_position - self.position)

        self.velocity = inertia + cognitive + social  # Update velocity based on PSO formula

        # Apply momentum and velocity clamping
        self.velocity *= self.momentum  # Apply momentum in-place
        self.velocity = tf.clip_by_value(self.velocity, -self.velocity_clamp, self.velocity_clamp)  # Apply velocity clamp

    def update_position(self) -> None:
        """
        Update the position of the particle based on the updated velocity.
        Ensures that the position stays within the valid input range [0, 1].
        """
        self.position = tf.clip_by_value(self.position + self.velocity, 0, 1)  # Ensure position stays within bounds
        
        # Record the new position at this epoch
        self.history.append(tf.identity(self.position))  # Add current position to history

    def evaluate(self) -> None:
        """
        Evaluate the fitness of the current particle and update its personal best.
        """
        score = self.fitness()  # Get the current fitness score
        if score > self.best_score:  # If score is better than the personal best, update the best position
            self.best_score = score
            self.best_position = self.position  # No need to clone the position here (avoid extra memory)
