import torch
import numpy as np

class BirdParticle:
    def __init__(self, position, w=1.0, c1=0.8, c2=0.2, name=None):
        """
        Initializes a particle.

        Args:
            position (torch.Tensor): The initial position of the particle.
            w (float): The inertia weight.
            c1 (float): The cognitive weight.
            c2 (float): The social weight.
        """
        self.position_i = position.clone().detach()
        self.velocity_i = torch.rand(position.shape).to(position.device)
        self.history = [self.position_i]
        self.pos_best_i = position.clone().detach()
        self.cost_best_i = -1
        self.cost_i = -1
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.name = name

    def evaluate(self, costFunc: callable, model: torch.nn.Module):
        """
        Evaluates the current fitness of the particle.

        Args:
            costFunc (callable): The cost function to be maximized.
            model (torch.nn.Module): The model to be used in the cost function.
        """
        self.cost_i = costFunc(model, self.position_i)
        if self.cost_i >= self.cost_best_i:
            self.pos_best_i = self.position_i
            self.cost_best_i = self.cost_i

    def update_velocity(self, pos_best_g: torch.Tensor):
        """
        Updates the particle velocity based on its own position and the global best position.

        Args:
            pos_best_g (torch.Tensor): The global best position.
        """
        r1 = torch.rand(1).to(pos_best_g.device)
        r2 = torch.rand(1).to(pos_best_g.device)
        vel_cognitive = self.c1 * r1 * (self.pos_best_i - self.position_i)
        vel_social = self.c2 * r2 * (pos_best_g - self.position_i)
        self.velocity_i = self.w * self.velocity_i + vel_cognitive + vel_social

    def update_position(self):
        """Updates the particle position based on its velocity."""
        self.position_i = self.position_i + self.velocity_i
        self.history.append(self.position_i)

    def get_history(self):
        """Returns the history of the particle's positions."""
        return self.history
