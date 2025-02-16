import os
import logging
from typing import List
from Adversarial_Observation.BirdParticle import BirdParticle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ParticleSwarm:
    """
    Represents the Particle Swarm Optimization (PSO) algorithm applied to adversarial attacks.
    
    The ParticleSwarm class manages the swarm of particles and optimizes the perturbations on the input data (image)
    to misclassify it into the target class.
    """

    def __init__(self, model: tf.keras.Model, input_set: np.ndarray, target_class: int,
                 num_iterations: int = 20, save_dir: str = 'results',
                 inertia_weight: float = 0.5, cognitive_weight: float = .5, social_weight: float = .5,
                 momentum: float = 0.9, clip_value_position: float = 1.0, device: str = 'cpu'):
        """
        Initialize the Particle Swarm Optimization (PSO) for adversarial attacks.
        
        Args:
            model (tf.keras.Model): The model to attack.
            input_set (np.ndarray): The batch of input images to attack as a NumPy array.
            target_class (int): The target class for misclassification.
            num_iterations (int): The number of optimization iterations.
            save_dir (str): The directory to save output images and logs.
            inertia_weight (float): The inertia weight for the velocity update.
            cognitive_weight (float): The cognitive weight for the velocity update.
            social_weight (float): The social weight for the velocity update.
            momentum (float): The momentum for the velocity update.
            clip_value_position (float): The maximum value for clipping the particle positions (default 1.0).
            device (str): The device for computation ('cpu' or 'gpu'). Default is 'cpu'.
        """
        self.model = model
        self.input_set = tf.convert_to_tensor(input_set, dtype=tf.float32)  # Convert NumPy array to TensorFlow tensor
        self.target_class = target_class  # The target class index
        self.num_iterations = num_iterations
        self.save_dir = save_dir  # Directory to save perturbed images
        self.device = device  # Device ('cpu' or 'gpu')
        
        self.particles: List[BirdParticle] = [
            BirdParticle(model, self.input_set[i:i + 1], target_class,
                         inertia_weight=inertia_weight, cognitive_weight=cognitive_weight,
                         social_weight=social_weight, momentum=momentum) 
            for i in range(len(input_set))
        ]
        
        self.global_best_position = tf.zeros_like(self.input_set[0])  # Global best position
        self.global_best_score = -float('inf')  # Initialize with a very low score
        
        self.fitness_history: List[float] = []  # History of fitness scores to track progress
        self.setup_logging()  # Set up logging
        self.log_progress(-1)  # Log initial state (before optimization)
        
        # Clip value for particle positions
        self.clip_value_position = clip_value_position  # Max value for clipping positions

    def setup_logging(self):
        """
        Set up logging for each iteration. Each iteration will have a separate log file.
        Also prints logs to the terminal.
        """
        iteration_dir = self.save_dir
        os.makedirs(iteration_dir, exist_ok=True)
        
        log_file = os.path.join(iteration_dir, f'iteration_log.log')
        self.logger = logging.getLogger()
        
        # Create a file handler to save logs to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        # Create a stream handler to output logs to the console (terminal)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))  # Keep it simple for console
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)  # Add the stream handler to print to terminal
        self.logger.setLevel(logging.INFO)

        # Log class initialization details
        self.logger.info(f"\n{'*' * 60}")
        self.logger.info(f"ParticleSwarm Optimization (PSO) for Adversarial Attack")
        self.logger.info(f"{'-' * 60}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Target Class: {self.target_class} (This is the class we want to misclassify the image into)")
        self.logger.info(f"Number of Iterations: {self.num_iterations} (Optimization steps)")
        self.logger.info(f"Save Directory: {self.save_dir}")
        self.logger.info(f"{'*' * 60}")

    def log_progress(self, iteration: int):
        """
        Log detailed information for the current iteration in a manually formatted table.
        
        Args:
            iteration (int): The current iteration of the optimization process.
        """
        # Log the header for the iteration
        self.logger.info(f"\n{'-'*60}")
        self.logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")
        self.logger.info(f"{'='*60}")
    
        # Table header
        header = f"{'Particle':<10}{'Original Pred':<15}{'Perturbed Pred':<18}{'Orig Target Prob':<20}" \
                 f"{'Pert Target Prob':<20}{'Personal Best':<20}{'Global Best':<20}"
        self.logger.info(header)
        self.logger.info(f"{'-'*60}")
    
        # Log particle information
        for i, particle in enumerate(self.particles):
            # Get original and perturbed outputs
            original_output = self.model(particle.original_data)  # Pass through the model
            perturbed_output = self.model(particle.position)  # Pass through the model
    
            # Get predicted classes
            original_pred = tf.argmax(original_output, axis=1).numpy().item()
            perturbed_pred = tf.argmax(perturbed_output, axis=1).numpy().item()
    
            # Get softmax probabilities
            original_probs = tf.nn.softmax(original_output, axis=1)
            perturbed_probs = tf.nn.softmax(perturbed_output, axis=1)
    
            # Get target class probabilities
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
        
        for i, particle in enumerate(self.particles):
            # Convert TensorFlow tensor to NumPy array
            position_numpy = particle.position.numpy()
            # Remove extra batch dimension (if it exists)
            position_numpy = np.squeeze(position_numpy)  # Now shape is (28, 28)
            plt.imsave(os.path.join(iteration_folder, f"perturbed_image_{i + 1}.png"), position_numpy, cmap="gray", vmin=0, vmax=1)

    def optimize(self):
        """
        Run the Particle Swarm Optimization process to optimize the perturbations.
        """
        with tf.device(f"/{self.device}:0"):  # Use the GPU/CPU based on the flag
            for iteration in range(self.num_iterations):
                
                # Update particles and velocities, evaluate them, and track global best
                for particle in self.particles:
                    particle.evaluate()
                    particle.update_velocity(self.global_best_position)  # No need to pass inertia_weight explicitly
                    particle.update_position()  # Apply the position update
                
                # Clip particle positions to ensure they stay within the range [0, clip_value_position]
                self.clip_position()
                
                # Update the global best based on the personal best scores of particles
                best_particle = max(self.particles, key=lambda p: p.best_score)
                if best_particle.best_score > self.global_best_score:
                    self.global_best_score = best_particle.best_score
                    self.global_best_position = tf.identity(best_particle.best_position)
                
                self.save_images(iteration)
                self.log_progress(iteration)

    def clip_position(self):
        """
        Clips the position of each particle to the specified `clip_value_position` range.
        """
        for particle in self.particles:
            particle.position = tf.clip_by_value(particle.position, 0.0, self.clip_value_position)  # Ensure position is within range [0, clip_value_position]
