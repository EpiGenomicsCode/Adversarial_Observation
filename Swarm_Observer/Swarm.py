import torch
import numpy as np
from .BirdParticle import BirdParticle
import pandas as pd
import logging

class PSO:
    def __init__(self, starting_positions: torch.Tensor, cost_func: callable, model: torch.nn.Module,
                 w: float = 1.0, c1: float = 0.8, c2: float = 0.2):
        """
        Initializes the Adversarial Particle Swarm Optimization algorithm.

        Args:
            starting_positions (torch.Tensor): The starting positions of the swarm.
            cost_func (callable): The cost function to be maximized.
            model (torch.nn.Module): The model to be used in the cost function.
            w (float): The inertia weight.
            c1 (float): The cognitive weight.
            c2 (float): The social weight.
        """
        self.swarm = [BirdParticle(pos, w=w, c1=c1, c2=c2) for pos in starting_positions]
        self.cost_func = cost_func
        self.model = model
        self.pos_best_g = self.swarm[0].position_i
        self.cos_best_g = self.swarm[0].cost_i
        self.epoch = 0
        self.history = []

    def step(self) -> tuple:
        """Performs one iteration of the Adversarial Particle Swarm Optimization algorithm."""
        self.epoch += 1
        for p in self.swarm:
            p.evaluate(self.cost_func, self.model)
            p.update_velocity(self.pos_best_g)
            p.update_position()
            p.evaluate(self.cost_func, self.model)

        for particle in self.swarm:
            if particle.cost_i > self.cos_best_g:
                self.pos_best_g = particle.position_i
                self.cos_best_g = particle.cost_i
            particle.history.append(particle.position_i)
        
    def get_points(self):
        return torch.vstack([particle.position_i for particle in self.swarm])
    
    def get_best(self):
        return self.pos_best_g
    
    def run(self, epochs: int):
        """Runs the Adversarial Particle Swarm Optimization algorithm for the specified number of epochs."""
        for _ in range(epochs):
            self.step()

    def get_history(self) -> pd.DataFrame:
        """Returns the history of the swarm's positions for each epoch."""
        history = {f"epoch_{i}": [particle.history[i] for particle in self.swarm] for i in range(self.epoch)}
        return pd.DataFrame(history)

    def save_history(self, filename):
        """Saves the history of the swarm's positions for each epoch."""
        history = self.get_history()
        history.to_csv(filename, index=False)
