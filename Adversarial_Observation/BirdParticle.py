import tensorflow as tf
import numpy as np

class BirdParticle:
    """
    Represents a particle in the Particle Swarm Optimization (PSO) algorithm for adversarial attacks.
    
    The BirdParticle class encapsulates the state of each particle, including its position, velocity,
    fitness evaluation, and the updates to its velocity and position based on the PSO algorithm.
    """

    def __init__(self, model: tf.keras.Model, input_data: tf.Tensor, target_class: int, num_iterations: int = 20,
                 velocity: tf.Tensor = None, inertia_weight: float = 0.5, 
                 cognitive_weight: float = 1.0, social_weight: float = 1.0, 
                 momentum: float = 0.9, clip_value_position: float = 1.0):
        """
        Initialize a particle in the PSO algorithm.
        
        Args:
            model (tf.keras.Model): The model to attack.
            input_data (tf.Tensor): The input data (image) to attack.
            target_class (int): The target class for misclassification.
            velocity (tf.Tensor, optional): The initial velocity for the particle's movement. Defaults to zero velocity if not provided.
            inertia_weight (float): The inertia weight for the velocity update. Default is 0.5.
            cognitive_weight (float): The cognitive weight for the velocity update. Default is 1.0.
            social_weight (float): The social weight for the velocity update. Default is 1.0.
            momentum (float): The momentum for the velocity update. Default is 0.9.
        """
        self.model = model
        self.num_iterations = num_iterations
        self.original_data = tf.identity(input_data)  # Clone the input data
        self.target_class = target_class
        self.best_position = tf.identity(input_data)  # Clone the input data
        self.best_score = -np.inf
        self.position = tf.identity(input_data)  # Clone the input data
        self.velocity = velocity if velocity is not None else tf.zeros_like(input_data)
        self.history = [self.position]
        self.clip_value_position = clip_value_position
        # Class attributes
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.momentum = momentum

    def fitness(self) -> float:
        """
        Compute the fitness score for the particle, which is the softmax probability of the target class.
        
        Higher fitness scores correspond to better success in the attack (misclassifying the image into the target class).
        
        Returns:
            float: Fitness score for this particle (higher is better).
        """
        output = self.model(self.position)  # Add batch dimension and pass through the model
        probabilities = tf.nn.softmax(output, axis=1)  # Get probabilities for each class
        target_prob = probabilities[:, self.target_class]  # Target class probability
        
        return target_prob.numpy().item()  # Return the target class probability as fitness score

    def update_velocity(self, global_best_position: tf.Tensor) -> None:
        """
        Update the velocity of the particle based on the PSO update rule.
        
        Args:
            global_best_position (tf.Tensor): The global best position found by the swarm.
        """
        inertia = self.inertia_weight * self.velocity
        cognitive = self.cognitive_weight * tf.random.uniform(self.position.shape) * (self.best_position - self.position)
        social = self.social_weight * tf.random.uniform(self.position.shape) * (global_best_position - self.position)

        # Apply momentum to velocity update:
        self.velocity = self.momentum * self.velocity + inertia + cognitive + social  # Apply momentum

    def update_position(self) -> None:
        """
        Update the position of the particle based on the updated velocity.
        
        The position is updated directly without any bounds checking.
        """
        self.position = self.position + self.velocity  # Update position directly without clipping
        self.position = tf.clip_by_value(self.position + self.velocity, 0, 1) # Ensure position stays within bounds
        self.history.append(tf.identity(self.position))  # Store the position history
        self.position = tf.clip_by_value(self.position, -self.clip_value_position, self.clip_value_position)
        
    def evaluate(self) -> None:
        """
        Evaluate the fitness of the current particle and update its personal best.
        
        The fitness score is calculated using the target class probability. If the current fitness score
        is better than the personal best, update the best position and score.
        """
        score = self.fitness()  # Get the current fitness score based on the perturbation
        if score > self.best_score:  # If score is better than the personal best, update the best position
            self.best_score = score
            self.best_position = tf.identity(self.position)  # Clone the current position

