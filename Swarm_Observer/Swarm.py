import torch
import numpy as np
from Swarm_Observer import BirdParticle
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
        self.pos_best_g = self.swarm[0].position_i.clone().detach()
        self.cos_best_g = self.swarm[0].cost_i

    def step(self) -> tuple:
        """
        Performs one iteration of the Adversarial Particle Swarm Optimization algorithm.

        Args:
            None

        Returns:
            None

        """
        # Evaluate fitness and update the best position and error for the group.
        for p in self.swarm:
            p.evaluate(self.cost_func, self.model)
            # Determine if current particle is the best (globally)
            # best has the highest confidence
            if p.cost_i >= self.cos_best_g:
                self.pos_best_g = p.position_i.clone().detach()
                self.cos_best_g = p.cost_i

        # Update velocities and positions.
        for p in self.swarm:
            p.update_velocity(pos_best_g=self.pos_best_g)
            p.update_position()

        # Update history.
        for particle in self.swarm:
            particle.history.append([self.epoch] + [i for i in particle.position_i.detach().numpy()]) 

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

    def get_swarm(self) -> pd.DataFrame:
        """
        Returns the swarm with each particle's position history.

        Args:
            None

        Returns:
            pd.DataFrame: A dataframe with each particle's position history.
        """
        df = pd.DataFrame(self.history,  columns=['Epoch']+['pos_'+str(i) for i in range(len(self.swarm[0].position_i))])

        # sort dataframe by epoch
        df = df.sort_values(by=['Epoch'])

        return df
    
    def save(self, filename: str):
        """
        Saves the each particle's position history to a csv file.

        Args:
            filename (str): The name of the file to save the swarm to.

        Returns:
            None
        """
        df = self.get_swarm()
        df.to_csv(filename, index=False)