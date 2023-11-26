import torch
import numpy as np
from Swarm_Observer.BirdParticle import BirdParticle
import pandas as pd

class PSO:
    def __init__(self, starting_positions: torch.Tensor, cost_func: callable, model: torch.nn.Module,
                 w: float = 1.0, c1: float = 0.8, c2: float = 0.2):
        """
        Initializes the Adversarial Particle Swarm Optimization algorithm.

        Args:
            starting_positions (torch.Tensor): The starting positions of the swarm.
                This should be a tensor of shape (n, m) where n is the number of particles and m is the number of dimensions.

            cost_func (callable): The cost function to be maximized.
                This should be a function that takes in a PyTorch model and a tensor of positions and returns a tensor of shape (n, 1) where n is the number of particles.

            model (torch.nn.Module): The model to be used in the cost function.
            w (float): The inertia weight.
            c1 (float): The cognitive weight.
            c2 (float): The social weight.
        """
        self.swarm = []
        self.epoch = 0
        self.history = []
        
        for i in starting_positions:
            self.swarm.append(BirdParticle(i, w=w, c1=c1, c2=c2))
        self.cost_func = cost_func
        self.model = model
        self.pos_best_g = self.swarm[0].position_i
        self.cos_best_g = self.swarm[0].cost_i

    def step(self) -> tuple:
        """
        Performs one iteration of the Adversarial Particle Swarm Optimization algorithm.

        Args:
            None

        Returns:
            None

        """
        self.epoch += 1
        # Update velocities and positions.
        for p in self.swarm:
            p.evaluate(self.cost_func, self.model)
            p.update_velocity(pos_best_g=self.pos_best_g)
            p.update_position()
            p.evaluate(self.cost_func, self.model)

        # Update history and global best.
        for particle in self.swarm:
            if particle.cost_i > self.cos_best_g:
                self.pos_best_g = particle.position_i
                self.cos_best_g = particle.cost_i
            particle.history.append(particle.position_i) 
        
    def getPoints(self):
        return torch.vstack([particle.position_i for particle in self.swarm])
    
    def getBest(self):
        return self.pos_best_g
    
    def run(self, epochs: int):
        """
        Runs the Adversarial Particle Swarm Optimization algorithm for the specified number of epochs.

        Args:
            epochs (int): The number of epochs to run the algorithm for.

        Returns:
            None
        """
        for i in range(epochs):
            self.step()

    def get_history(self) -> pd.DataFrame:
        """
        Returns the history of the swarm's positions for each epoch.

        Returns:
            pd.DataFrame: A dataframe containing the swarm's positions at each epoch.
        """

        history = {}
        
        for i in range(0, self.epoch):
            history[f"epoch_{i}"] = [particle.history[i] for particle in self.swarm]
        
        return history
    
    def save_history(self, filename):
        """
        Saves the history of the swarm's positions for each epoch.

        Args:
            filename (str): The filename to save the history to.

        Returns:
            None
        """
        history = self.get_history()
        history.to_csv(filename, index=False)
        