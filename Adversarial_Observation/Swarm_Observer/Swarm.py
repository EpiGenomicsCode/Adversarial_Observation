import torch
import numpy as np
from Adversarial_Observation.Swarm_Observer import BirdParticle
import tqdm
import pandas as pd

class PSO:
    def __init__(self, starting_positions: torch.Tensor, cost_func: callable, model: torch.nn.Module,
                 w: float = 1.0, c1: float = 0.8, c2: float = 0.2):
        """
        Initializes the Adversarial Particle Swarm Optimization algorithm.
        :param starting_positions: The starting positions of the swarm.
        :param cost_func: The cost function to be maximized.
        :param model: The model to be used in the cost function.
        :param w: The inertia weight.
        :param c1: The cognitive weight.
        :param c2: The social weight.
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
        Runs one iteration of the algorithm.
        :return: A tuple containing the final positions of the swarm and the best positions of the swarm.
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

    def save(self, filename: str):

        df = pd.DataFrame(self.history,  columns=['Epoch']+['pos_'+str(i) for i in range(len(self.swarm[0].position_i))])
        # sort dataframe by epoch
        df = df.sort_values(by=['Epoch'])
        df.to_csv(filename, index=False)