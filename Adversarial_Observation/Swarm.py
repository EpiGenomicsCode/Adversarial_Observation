import sys
import torch
import numpy as np
from Adversarial_Observation.BirdParticle import BirdParticle
import pandas as pd

class PSO:
    def __init__(self, starting_positions: torch.Tensor, cost_func: callable, model: torch.nn.Module,
                 w: float = 1.0, c1: float = 0.8, c2: float = 0.2, minclamp: float = 0.0, maxclamp: float = 1.0):
        """
        Initializes the Adversarial Particle Swarm Optimization algorithm.

        Args:
            starting_positions (torch.Tensor): Tensor of shape (n, m) where n = number of particles, m = dimensions.
            cost_func (callable): Cost function to maximize; takes (model, position) and returns scalar cost.
            model (torch.nn.Module): Model used in the cost function.
            w (float): Inertia weight.
            c1 (float): Cognitive (self) weight.
            c2 (float): Social (global) weight.
            minclamp, maxclamp (float): Value clamps for position updates.
        """
        self.model = model
        self.cost_func = cost_func
        self.epoch = 0
        self.history = []
    
        # Initialize all particles
        self.swarm = [
            BirdParticle(pos, w=w, c1=c1, c2=c2, minclamp=minclamp, maxclamp=maxclamp)
            for pos in starting_positions
        ]

        # Evaluate all initial costs to find the true best particle
        with torch.no_grad():
            costs = []
            for particle in self.swarm:
                cost_val = self.cost_func(self.model, particle.position_i)
                # Convert float → tensor if needed
                if not torch.is_tensor(cost_val):
                    cost_val = torch.tensor(cost_val, dtype=torch.float32)
                particle.cost_i = cost_val
                costs.append(cost_val)
            costs = torch.stack(costs)

        # Determine index of best particle (maximize cost)
        best_idx = torch.argmax(costs)
        self.cos_best_g = costs[best_idx].clone()
        self.pos_best_g = self.swarm[best_idx].position_i.clone()

        print(f"[Init] Swarm initialized with {len(self.swarm)} particles.", file=sys.stderr, flush=True)
        print(f"[Init] Best initial cost: {self.cos_best_g.item():.4f}", file=sys.stderr, flush=True)
        
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
            # Convert tensors to numpy arrays and flatten them so Pandas can handle them nicely
            history[f"epoch_{i}"] = [particle.history[i].cpu().numpy().flatten().tolist() for particle in self.swarm]
        
        # Wrap the dictionary in a pandas DataFrame before returning
        return pd.DataFrame(history)
    
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
        
