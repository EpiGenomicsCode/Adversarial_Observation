import torch
import numpy as np
import random

class BirdParticle:
    def __init__(self, model, input_data, target_class, epsilon, velocity=None):
        """
        Initialize a particle for the bird in the particle swarm optimization (PSO) attack algorithm.
        
        Args:
            model (torch.nn.Module): The model to attack.
            input_data (torch.Tensor): The original input data to attack.
            target_class (int): The target class for misclassification.
            epsilon (float): The perturbation bound.
            velocity (torch.Tensor, optional): The initial velocity for the particle.
        """
        self.model = model
        self.original_data = input_data.clone().detach()
        self.target_class = target_class  # Store the target class
        self.epsilon = epsilon
        self.best_position = input_data.clone().detach()  # Best known position for the particle
        self.best_score = -np.inf  # Score based on the success rate of attack
        self.position = input_data.clone().detach()  # Current position of the particle
        self.velocity = velocity if velocity is not None else torch.zeros_like(input_data)  # Current velocity
        
    def fitness(self):
        """
        Compute the fitness score for this particle.
        
        Fitness is based on the model's output probability for the target class (higher is better).
        
        Returns:
            float: The fitness score of this particle.
        """
        # Evaluate the model's prediction for the current adversarial example
        with torch.no_grad():
            output = self.model(self.position)
            probabilities = torch.softmax(output, dim=1)  # Get class probabilities
            target_prob = probabilities[:, self.target_class]  # Get the probability for the target class
            
        return target_prob.item()  # Higher probability for target class is better
    
    def update_velocity(self, global_best_position, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0):
        """
        Update the velocity of the particle using the PSO velocity update equation.
        
        Args:
            global_best_position (torch.Tensor): The best known global position of the swarm.
            inertia_weight (float): The inertia weight for the particle.
            cognitive_weight (float): The cognitive weight, determining the influence of the particle's best position.
            social_weight (float): The social weight, determining the influence of the global best position.
        """
        inertia = inertia_weight * self.velocity
        cognitive = cognitive_weight * torch.rand_like(self.position) * (self.best_position - self.position)
        social = social_weight * torch.rand_like(self.position) * (global_best_position - self.position)

        self.velocity = inertia + cognitive + social
        
    def update_position(self):
        """
        Update the position of the particle based on the current velocity.
        """
        self.position = torch.clamp(self.position + self.velocity, 0, 1)  # Ensure the position is within [0, 1]
        
    def evaluate(self):
        """
        Evaluate the particle's position and update the best known position and score if applicable.
        """
        score = self.fitness()
        
        # If the current position is better than the best known position, update
        if score > self.best_score:
            self.best_score = score
            self.best_position = self.position.clone().detach()

