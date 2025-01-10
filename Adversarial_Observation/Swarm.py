import torch
import numpy as np
import random
from Adversarial_Observation.BirdParticle import BirdParticle
import time

class ParticleSwarm:
    def __init__(self, model, input_set, target_class, num_particles=10, num_iterations=20, epsilon=0.1):
        self.model = model
        self.input_set = input_set  # The batch of input images
        self.target_class = target_class  # The target class index
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.epsilon = epsilon  # Perturbation bound
        
        # Initialize particles
        self.particles = [BirdParticle(model, input_set[i:i+1], target_class, epsilon) for i in range(num_particles)]
        
        self.global_best_position = torch.zeros_like(self.input_set[0])  # Global best position
        self.global_best_score = -np.inf  # Initialize with a very low score (we're maximizing)

    def fitness_function(self, images):
        """
        Compute the fitness score for each particle based on softmax probability for the target class.
        """
        self.model.eval()
        with torch.no_grad():
            # Pass images through the model to get raw logits
            logits = self.model(images)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1)
            
            # The fitness score is the probability of the target class
            target_probs = probabilities[:, self.target_class]
            
        return target_probs  # Return the probability of the target class

    def optimize(self):
        """
        Run the Particle Swarm Optimization to maximize the softmax probability of the target class.
        """
        for iteration in range(self.num_iterations):
            start_time = time.time()  # Track the time taken for each iteration

            # Evaluate fitness scores
            for particle in self.particles:
                particle.evaluate()
            
            # Update the global best position
            for particle in self.particles:
                if particle.best_score > self.global_best_score:
                    self.global_best_score = particle.best_score
                    self.global_best_position = particle.best_position.clone().detach()

            # Update velocities and positions
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

            # Print progress
            print(f"Iteration {iteration + 1}/{self.num_iterations}, Global Best Score: {self.global_best_score:.4f}, "
                  f"Time: {time.time() - start_time:.2f}s")

        # Return the final perturbed images based on the global best particle
        final_perturbed_images = torch.stack([particle.best_position for particle in self.particles])
        return final_perturbed_images
