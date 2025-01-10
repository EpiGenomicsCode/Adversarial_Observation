import torch
import numpy as np
from torch import nn


class BirdParticle:
    """
    Represents a particle in the Particle Swarm Optimization (PSO) algorithm for adversarial attacks.
    
    The BirdParticle class encapsulates the state of each particle, including its position, velocity,
    fitness evaluation, and the updates to its velocity and position based on the PSO algorithm.
    """

    def __init__(self, model: nn.Module, input_data: torch.Tensor, target_class: int, epsilon: float,
                 velocity: torch.Tensor = None, inertia_weight: float = 0.5, 
                 cognitive_weight: float = 1.0, social_weight: float = 1.0, momentum: float = 0.9,
                 velocity_clamp: float = 0.1):
        """
        Initialize a particle in the PSO algorithm.
        
        Args:
            model (nn.Module): The model to attack.
            input_data (torch.Tensor): The input data (image) to attack.
            target_class (int): The target class for misclassification.
            epsilon (float): The perturbation bound (maximum amount the image can be altered).
            velocity (torch.Tensor, optional): The initial velocity for the particle's movement. Defaults to zero velocity if not provided.
            inertia_weight (float): The inertia weight for the velocity update. Default is 0.5.
            cognitive_weight (float): The cognitive weight for the velocity update. Default is 1.0.
            social_weight (float): The social weight for the velocity update. Default is 1.0.
            momentum (float): The momentum for the velocity update. Default is 0.9.
            velocity_clamp (float): The velocity clamp for limiting the maximum velocity. Default is 0.1.
        """
        self.model = model
        self.original_data = input_data.clone().detach()
        self.target_class = target_class
        self.epsilon = epsilon
        self.best_position = input_data.clone().detach()
        self.best_score = -np.inf
        self.position = input_data.clone().detach()
        self.velocity = velocity if velocity is not None else torch.zeros_like(input_data)
        
        # Class attributes
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.momentum = momentum
        self.velocity_clamp = velocity_clamp

    def fitness(self) -> float:
        """
        Compute the fitness score for the particle, which is the softmax probability of the target class.
        
        Higher fitness scores correspond to better success in the attack (misclassifying the image into the target class).
        
        Returns:
            float: Fitness score for this particle (higher is better).
        """
        with torch.no_grad():
            output = self.model(self.position)
            probabilities = torch.softmax(output, dim=1)  # Get probabilities for each class
            target_prob = probabilities[:, self.target_class]  # Target class probability
            
        return target_prob.item()  # Return the target class probability as fitness score

    def update_velocity(self, global_best_position: torch.Tensor) -> None:
        """
        Update the velocity of the particle based on the PSO update rule.
        
        Args:
            global_best_position (torch.Tensor): The global best position found by the swarm.
        """
        inertia = self.inertia_weight * self.velocity
        cognitive = self.cognitive_weight * torch.rand_like(self.position) * (self.best_position - self.position)
        social = self.social_weight * torch.rand_like(self.position) * (global_best_position - self.position)

        self.velocity = inertia + cognitive + social  # Update velocity based on PSO formula

        # Apply momentum and velocity clamping
        self.velocity = self.velocity * self.momentum  # Apply momentum
        self.velocity = torch.clamp(self.velocity, -self.velocity_clamp, self.velocity_clamp)  # Apply velocity clamp

    def update_position(self) -> None:
        """
        Update the position of the particle based on the updated velocity.
        
        Ensures that the position stays within the valid input range [0, 1] (normalized pixel values).
        """
        self.position = torch.clamp(self.position + self.velocity, 0, 1)  # Ensure position stays within bounds

    def evaluate(self) -> None:
        """
        Evaluate the fitness of the current particle and update its personal best.
        
        The fitness score is calculated using the target class probability. If the current fitness score
        is better than the personal best, update the personal best position and score.
        """
        score = self.fitness()  # Get the current fitness score based on the perturbation
        if score > self.best_score:  # If score is better than the personal best, update the best position
            self.best_score = score
            self.best_position = self.position.clone().detach()
