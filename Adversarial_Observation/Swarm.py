import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List
from Adversarial_Observation.BirdParticle import BirdParticle

class ParticleSwarm:
    """
    Represents the Particle Swarm Optimization (PSO) algorithm applied to adversarial attacks.
    """

    def __init__(self, model: tf.keras.Model, input_set: np.ndarray, target_class: int,
                 num_iterations: int = 20, epsilon: float = 0.8, save_dir: str = 'results',
                 inertia_weight: float = 0.5, cognitive_weight: float = .5, social_weight: float = .5,
                 momentum: float = 0.9, velocity_clamp: float = 0.1, clip_range: tuple = (0.0, 1.0)):
        """
        Initialize the Particle Swarm Optimization (PSO) for adversarial attacks.
        """
        self.model = model
        self.input_set = tf.convert_to_tensor(input_set, dtype=tf.float32)  # Convert NumPy array to TensorFlow tensor
        self.target_class = target_class
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.save_dir = save_dir
        self.clip_range = clip_range  # Clipping range for particle positions
        
        # Create particles using list comprehension
        self.particles: List[BirdParticle] = [
            BirdParticle(model, self.input_set[i:i + 1], target_class, epsilon,
                         inertia_weight=inertia_weight, cognitive_weight=cognitive_weight,
                         social_weight=social_weight, momentum=momentum, velocity_clamp=velocity_clamp) 
            for i in range(len(input_set))
        ]
        
        self.global_best_position = tf.zeros_like(self.input_set[0])  # Global best position
        self.global_best_score = -float('inf')  # Initialize with a very low score
        
        self.fitness_history: List[float] = []  # History of fitness scores to track progress
        self.setup_logging()  # Set up logging
        self.log_progress(-1)  # Log initial state (before optimization)

    def setup_logging(self):
        """
        Set up logging for each iteration.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        
        log_file = os.path.join(self.save_dir, f'iteration_log.log')
        self.logger = logging.getLogger()
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging.INFO)

        self.logger.info(f"PSO for Adversarial Attack initialized.\n{'='*60}")

    def log_progress(self, iteration: int):
        """
        Log detailed information for the current iteration.
        """
        self.logger.info(f"\n{'-'*60}")
        self.logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")
        self.logger.info(f"{'='*60}")
    
        header = f"{'Particle':<10}{'Original Pred':<15}{'Perturbed Pred':<18}{'Orig Target Prob':<20}" \
                 f"{'Pert Target Prob':<20}{'Personal Best':<20}{'Global Best':<20}"
        self.logger.info(header)
        self.logger.info(f"{'-'*60}")
    
        # Instead of logging each particle individually in the loop, we process them all in batch.
        batch_outputs = self.model(self.input_set)  # Perform forward pass for all particles in batch
        
        for i, particle in enumerate(self.particles):
            original_output = batch_outputs[i:i+1]
            perturbed_output = batch_outputs[i:i+1]  # Same as original, but can replace with the perturbed state
            
            original_pred = tf.argmax(original_output, axis=1).numpy().item()
            perturbed_pred = tf.argmax(perturbed_output, axis=1).numpy().item()
    
            original_probs = tf.nn.softmax(original_output, axis=1)
            perturbed_probs = tf.nn.softmax(perturbed_output, axis=1)
    
            original_prob_target = original_probs[0, self.target_class].numpy().item()
            perturbed_prob_target = perturbed_probs[0, self.target_class].numpy().item()
    
            # Log each particle's data in a formatted row
            self.logger.info(f"{i+1:<10}{original_pred:<15}{perturbed_pred:<18}{original_prob_target:<20.4f}"
                             f"{perturbed_prob_target:<20.4f}{particle.best_score:<20.4f}{self.global_best_score:<20.4f}")
    
        self.logger.info(f"{'='*60}")

    def save_images(self, iteration: int):
        """
        Save the perturbed images for the current iteration.
        
        Args:
            iteration (int): The current iteration number.
        """
        iteration_folder = os.path.join(self.save_dir, f"iteration_{iteration + 1}")
        os.makedirs(iteration_folder, exist_ok=True)
        
        # Batch saving images
        for i, particle in enumerate(self.particles):
            # Convert TensorFlow tensor to NumPy array
            position_numpy = particle.position.numpy()
            position_numpy = np.squeeze(position_numpy)  # Now shape is (28, 28)
            plt.imsave(os.path.join(iteration_folder, f"perturbed_image_{i + 1}.png"), position_numpy, cmap="gray", vmin=0, vmax=1)

    def optimize(self):
        """
        Run the Particle Swarm Optimization process to optimize the perturbations.
        """
        for iteration in range(self.num_iterations):
            
            # Update particles and velocities, evaluate them, and track global best
            for particle in self.particles:
                particle.evaluate()
                particle.update_velocity(self.global_best_position)  # No need to pass inertia_weight explicitly
                particle.update_position()

                # Clip particle position to stay within the specified range
                particle.position = tf.clip_by_value(particle.position, self.clip_range[0], self.clip_range[1])
            
            # Update the global best based on the personal best scores of particles
            best_particle = max(self.particles, key=lambda p: p.best_score)
            if best_particle.best_score > self.global_best_score:
                self.global_best_score = best_particle.best_score
                self.global_best_position = tf.identity(best_particle.best_position)
            
            # Save images and log progress at the end of each iteration
            self.save_images(iteration)
            self.log_progress(iteration)
