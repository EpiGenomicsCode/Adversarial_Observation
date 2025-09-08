import os
import logging
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Adversarial_Observation.BirdParticle import BirdParticle


class ParticleSwarm:
    def __init__(self, model: torch.nn.Module, input_set: np.ndarray, starting_class: int, target_class: int,
                 num_iterations: int = 20, save_dir: str = 'results', inertia_weight: float = 0.5,
                 cognitive_weight: float = 0.5, social_weight: float = 0.5, momentum: float = 0.9,
                 clip_value_position: float = 0.2, enable_logging: bool = False, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.device = torch.device(device)

        self.input_set = input_set.float().view(-1, 1, 28, 28).to(self.device)

        self.start_class = starting_class
        self.target_class = target_class
        self.num_iterations = num_iterations
        self.save_dir = save_dir
        self.enable_logging = enable_logging

        self.particles: List[BirdParticle] = [
            BirdParticle(model, self.input_set[i:i + 1], target_class,
                         inertia_weight=inertia_weight, cognitive_weight=cognitive_weight,
                         social_weight=social_weight, momentum=momentum,
                         clip_value_position=clip_value_position, device=self.device)
            for i in range(len(input_set))
        ]

        self.global_best_position = torch.zeros_like(self.input_set[0])
        self.global_best_score = -float('inf')
        self.fitness_history: List[float] = []

        os.makedirs(self.save_dir, exist_ok=True)
        if self.enable_logging:
            self.setup_logging()
            self.log_progress(-1)

    def setup_logging(self):
        log_file = os.path.join(self.save_dir, 'iteration_log.log')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info(f"\n{'*' * 60}")
        self.logger.info(f"ParticleSwarm Optimization (PSO) for Adversarial Attack")
        self.logger.info(f"{'-' * 60}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Target Class: {self.target_class}")
        self.logger.info(f"Number of Iterations: {self.num_iterations}")
        self.logger.info(f"Save Directory: {self.save_dir}")
        self.logger.info(f"{'*' * 60}")

    def log_progress(self, iteration: int):
        if not self.enable_logging:
            return

        self.logger.info(f"\n{'-'*60}")
        self.logger.info(f"Iteration {iteration + 1}/{self.num_iterations}")
        self.logger.info(f"{'='*60}")

        header = f"{'Particle':<10}{'Original Pred':<15}{'Perturbed Pred':<18}{'Orig Start Prob':<20}{'Pert Start Prob':<20}{'Orig Target Prob':<20}{'Pert Target Prob':<20}{'Personal Best':<20}{'Global Best':<20}"
        self.logger.info(header)
        self.logger.info(f"{'-'*60}")

        for i, particle in enumerate(self.particles):
            with torch.no_grad():
                original_output = self.model(particle.original_data)
                perturbed_output = self.model(particle.position)

                original_probs = torch.softmax(original_output, dim=1)
                perturbed_probs = torch.softmax(perturbed_output, dim=1)

                original_pred = original_output.argmax(dim=1).item()
                perturbed_pred = perturbed_output.argmax(dim=1).item()

                orig_start_prob = original_probs[0, self.start_class].item()
                pert_start_prob = perturbed_probs[0, self.start_class].item()
                orig_target_prob = original_probs[0, self.target_class].item()
                pert_target_prob = perturbed_probs[0, self.target_class].item()

            self.logger.info(f"{i+1:<10}{original_pred:<15}{perturbed_pred:<18}"
                             f"{orig_start_prob:<20.4f}{pert_start_prob:<20.4f}"
                             f"{orig_target_prob:<20.4f}{pert_target_prob:<20.4f}"
                             f"{particle.best_score:<20.4f}{self.global_best_score:<20.4f}")

        self.logger.info(f"{'='*60}")

    def optimize(self):
        for iteration in tqdm(range(self.num_iterations), desc="Running Swarm"):
            for particle in self.particles:
                particle.evaluate()
                particle.update_velocity(self.global_best_position)
                particle.update_position()

            best_particle = max(self.particles, key=lambda p: p.best_score)
            if best_particle.best_score > self.global_best_score:
                self.global_best_score = best_particle.best_score
                self.global_best_position = best_particle.best_position.clone()

            self.log_progress(iteration)

    def reduce_excess_perturbations(self, original_img: np.ndarray, target_label: int, model_shape: tuple = (1, 1, 28, 28)) -> List[np.ndarray]:
        denoised_adv = []
        total_pixels = np.prod(original_img.shape)

        for adv_particle in tqdm(self.particles, desc="Processing Particles"):
            adv_img = adv_particle.position.clone().detach().cpu().numpy().reshape(original_img.shape)
            orig = original_img.copy()

            with tqdm(total=total_pixels, desc="Processing Pixels", leave=False) as pbar:
                for idx in np.ndindex(original_img.shape):
                    if orig[idx] == adv_img[idx]:
                        pbar.update(1)
                        continue

                    old_val = adv_img[idx]
                    adv_img[idx] = orig[idx]

                    test_img = torch.from_numpy(adv_img.reshape(model_shape)).float().to(self.device)
                    with torch.no_grad():
                        output = self.model(test_img)
                        pred = output.softmax(dim=1).argmax(dim=1).item()

                    if pred != target_label:
                        adv_img[idx] = old_val + (orig[idx] - old_val) * 0.5
                        test_img = torch.from_numpy(adv_img.reshape(model_shape)).float().to(self.device)
                        with torch.no_grad():
                            output = self.model(test_img)
                            pred = output.softmax(dim=1).argmax(dim=1).item()

                        if pred != target_label:
                            adv_img[idx] = old_val

                    pbar.update(1)

            denoised_adv.append(adv_img)

        return denoised_adv

    def getBest(self) -> np.ndarray:
        return self.global_best_position.detach().cpu().numpy()

    def getPoints(self) -> List[np.ndarray]:
        return [particle.position.detach().cpu().numpy() for particle in self.particles]
