import os
import logging
import torch
from torch import nn
from torchvision import utils
from typing import List
from Adversarial_Observation.BirdParticle import BirdParticle


class ParticleSwarm:
    """
    Represents the Particle Swarm Optimization (PSO) algorithm applied to adversarial attacks.
    
    The ParticleSwarm class manages the swarm of particles and optimizes the perturbations on the input data (image)
    to misclassify it into the target class.
    """

    def __init__(self, model: nn.Module, input_set: torch.Tensor, target_class: int,
                 num_iterations: int = 20, epsilon: float = 0.8, save_dir: str = 'results',
                 inertia_weight: float = 0.5, cognitive_weight: float = .5, social_weight: float = .5,
                 momentum: float = 0.9, velocity_clamp: float = 0.1):
        """
        Initialize the Particle Swarm Optimization (PSO) for adversarial attacks.
        
        Args:
            model (nn.Module): The model to attack.
            input_set (torch.Tensor): The batch of input images to attack.
            target_class (int): The target class for misclassification.
            num_iterations (int): The number of optimization iterations.
            epsilon (float): The perturbation bound.
            save_dir (str): The directory to save output images and logs.
            inertia_weight (float): The inertia weight for the velocity update.
            cognitive_weight (float): The cognitive weight for the velocity update.
            social_weight (float): The social weight for the velocity update.
            momentum (float): The momentum for the velocity update.
            velocity_clamp (float): The velocity clamp to limit the velocity.
        """
        self.model = model
        self.input_set = input_set  # The batch of input images
        self.target_class = target_class  # The target class index
        self.num_iterations = num_iterations
        self.epsilon = epsilon  # Perturbation bound
        self.save_dir = save_dir  # Directory to save perturbed images
        
        self.particles: List[BirdParticle] = [
            BirdParticle(model, input_set[i:i + 1], target_class, epsilon,
                         inertia_weight=inertia_weight, cognitive_weight=cognitive_weight,
                         social_weight=social_weight, momentum=momentum, velocity_clamp=velocity_clamp) 
            for i in range(len(input_set))
        ]
        
        self.global_best_position = torch.zeros_like(self.input_set[0])  # Global best position
        self.global_best_score = -float('inf')  # Initialize with a very low score
        
        self.fitness_history: List[float] = []  # History of fitness scores to track progress
        self.setup_logging()  # Set up logging

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
        self.logger.info(f"Epsilon (perturbation bound): {self.epsilon} (Maximum perturbation allowed)")
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
            with torch.no_grad():
                original_output = self.model(particle.original_data)
                perturbed_output = self.model(particle.position)

                original_pred = torch.argmax(original_output, dim=1).item()
                perturbed_pred = torch.argmax(perturbed_output, dim=1).item()

                original_probs = torch.softmax(original_output, dim=1)
                perturbed_probs = torch.softmax(perturbed_output, dim=1)

                original_prob_target = original_probs[0, self.target_class].item()
                perturbed_prob_target = perturbed_probs[0, self.target_class].item()

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
            utils.save_image(particle.position, os.path.join(iteration_folder, f"perturbed_image_{i + 1}.png"))

    def optimize(self):
        """
        Run the Particle Swarm Optimization process to optimize the perturbations.
        """
        for iteration in range(self.num_iterations):
            self.log_progress(iteration)
            
            # Update particles and velocities, evaluate them, and track global best
            for particle in self.particles:
                particle.evaluate()
                particle.update_velocity(self.global_best_position)  # No need to pass inertia_weight explicitly
                particle.update_position()
            
            # Update the global best based on the personal best scores of particles
            best_particle = max(self.particles, key=lambda p: p.best_score)
            if best_particle.best_score > self.global_best_score:
                self.global_best_score = best_particle.best_score
                self.global_best_position = best_particle.best_position.clone().detach()
            
            self.save_images(iteration)
