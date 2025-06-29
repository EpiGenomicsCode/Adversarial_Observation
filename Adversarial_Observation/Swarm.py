import os
import logging
from typing import List
from Adversarial_Observation.BirdParticle import BirdParticle
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ParticleSwarm:
    """
    Represents the Particle Swarm Optimization (PSO) algorithm applied to adversarial attacks.
    
    The ParticleSwarm class manages the swarm of particles and optimizes the perturbations on the input data (image)
    to misclassify it into the target class.
    """

    def __init__(self, model: tf.keras.Model, input_set: np.ndarray, starting_class: int, target_class: int,
                 num_iterations: int = 20, save_dir: str = 'results', inertia_weight: float = 0.5,
                 cognitive_weight: float = .5, social_weight: float = .5, momentum: float = 0.9,
                 clip_value_position: float = 0.2, enable_logging: bool = False, device: str = 'cpu'):
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
            clip_value_position (float): The velocity clamp to limit the velocity.
            device (str): The device for computation ('cpu' or 'gpu'). Default is 'cpu'.
        """
        self.model = model
        self.input_set = tf.convert_to_tensor(input_set, dtype=tf.float32)  # Convert NumPy array to TensorFlow tensor
        self.start_class = starting_class # The starting class index
        self.target_class = target_class  # The target class index
        self.num_iterations = num_iterations
        self.save_dir = save_dir  # Directory to save perturbed images
        self.enable_logging = enable_logging
        self.device = device  # Device ('cpu' or 'gpu')

        self.particles: List[BirdParticle] = [
            BirdParticle(model, self.input_set[i:i + 1], target_class,
                         inertia_weight=inertia_weight, cognitive_weight=cognitive_weight, social_weight=social_weight,
                         momentum=momentum, clip_value_position=clip_value_position) 
            for i in range(len(input_set))
        ]
        
        self.global_best_position = tf.zeros_like(self.input_set[0]) # Global best position
        self.global_best_score = -float('inf')  # Initialize with a very low score
        
        self.fitness_history: List[float] = []  # History of fitness scores to track progress

        # Make output folder
        iteration_dir = self.save_dir
        os.makedirs(iteration_dir, exist_ok=True)
        if self.enable_logging:
            self.setup_logging()
            self.log_progress(-1)

    def setup_logging(self):
        """
        Set up logging for each iteration. Each iteration will have a separate log file.
        Also prints logs to the terminal.
        """
        log_file = os.path.join(self.save_dir, f'iteration_log.log')
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
        if not self.enable_logging:
            return

        # Log the header for the iteration
        self.logger.info(f"\n{'-'*60}")
        self.logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")
        self.logger.info(f"{'='*60}")
    
        # Table header
        header = f"{'Particle':<10}{'Original Pred':<15}{'Perturbed Pred':<18}{'Orig Start Prob':<20}{'Pert Start Prob':<20}{'Orig Target Prob':<20}" \
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

            # Get starting class probabilities (how far away we've moved)
            original_prob_start = original_probs[0, self.start_class].numpy().item()
            perturbed_prob_start = perturbed_probs[0, self.start_class].numpy().item()

            # Get target class probabilities
            original_prob_target = original_probs[0, self.target_class].numpy().item()
            perturbed_prob_target = perturbed_probs[0, self.target_class].numpy().item()
    
            # Log each particle's data in a formatted row
            self.logger.info(f"{i+1:<10}{original_pred:<15}{perturbed_pred:<18}{original_prob_start:<20.4f}{perturbed_prob_start:<20.4f}"
                            f"{original_prob_target:<20.4f}{perturbed_prob_target:<20.4f}{particle.best_score:<20.4f}{self.global_best_score:<20.4f}")
    
        self.logger.info(f"{'='*60}")
    
    def optimize(self):
        """
        Run the Particle Swarm Optimization process to optimize the perturbations.
        """
        with tf.device(f"/{self.device}:0"):  # Use the GPU/CPU based on the flag
            for iteration in tqdm(range(self.num_iterations), desc="Running Swarm"):
                # Update particles and velocities, evaluate them, and track global best
                for particle in self.particles:
                    particle.evaluate()
                    particle.update_velocity(self.global_best_position)  # No need to pass inertia_weight explicitly
                    particle.update_position()
                
                # Update the global best based on the personal best scores of particles
                best_particle = max(self.particles, key=lambda p: p.best_score)
                if best_particle.best_score > self.global_best_score:
                    self.global_best_score = best_particle.best_score
                    self.global_best_position = tf.identity(best_particle.best_position)
                
                self.log_progress(iteration)

    def reduce_excess_perturbations(self, original_img: np.ndarray, target_label: int, model_shape: tuple = (1, 28, 28, 1)) -> np.ndarray:
        """
        Reduces excess perturbations in adversarial images while ensuring the misclassification remains.

        Args:
            original_img (np.ndarray): The original (clean) image.
            target_label (int): The label we want the adversarial image to produce.
            model_shape (tuple): The expected shape of the model input.

        Returns:
            list[np.ndarray]: A list of denoised adversarial images.
        """
        denoised_adv = []
        total_pixels = np.prod(original_img.shape)  # Total pixels in the image

        for adv_particle in tqdm(self.particles, desc="Processing Particles", unit="particle"):
            adv_img = np.copy(adv_particle.position).reshape(original_img.shape)  # Copy to avoid modifying the original

            if original_img.shape != adv_img.shape:
                raise ValueError("original_img and adv_img must have the same shape.")

            # Iterate over every pixel coordinate in the image with tqdm progress bar
            with tqdm(total=total_pixels, desc="Processing Pixels", unit="pixel", leave=False) as pbar:
                for idx in np.ndindex(original_img.shape):
                    if original_img[idx] == adv_img[idx]:  # Ignore unchanged pixels
                        pbar.update(1)
                        continue

                    # Store old adversarial value
                    old_val = adv_img[idx]

                    # Try restoring the pixel to original
                    adv_img[idx] = original_img[idx]

                    # Check if the label is still the target label
                    output = self.model(adv_img.reshape(model_shape))
                    softmax_output = tf.nn.softmax(tf.squeeze(output), axis=0).numpy()
                    current_label = np.argmax(softmax_output)

                    if current_label != target_label:
                        # If misclassification is lost, try halfway adjustment
                        adv_img[idx] = old_val
                        adv_img[idx] += (original_img[idx] - adv_img[idx]) * 0.5

                        # Recheck if the label is still the target
                        output = self.model(adv_img.reshape(model_shape))
                        softmax_output = tf.nn.softmax(tf.squeeze(output), axis=0).numpy()
                        current_label = np.argmax(softmax_output)

                        if current_label != target_label:
                            # If misclassification is still lost, revert back
                            adv_img[idx] = old_val

                    pbar.update(1)  # Update pixel progress

            denoised_adv.append(adv_img)

        return denoised_adv
