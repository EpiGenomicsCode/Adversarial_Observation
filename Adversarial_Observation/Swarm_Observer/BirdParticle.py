import torch
import numpy as np
import pdb

class BirdParticle:
    def __init__(self, position, w=1.0, c1=0.8, c2=0.2, name=None):
        """
        Initializes a particle.
        :param position: the initial position of the particle
        :param w: the inertia weight
        :param c1: the cognitive weight
        :param c2: the social weight
        """

        self.position_i = torch.tensor(position)
        self.velocity_i = torch.rand(self.position_i.shape) 
        # copy the current position to the best position

        self.history = [self.position_i]
        
        self.pos_best_i = self.position_i.detach().clone()   # best position individual
        self.cost_best_i = -1   # best error individual
        self.cost_i = -1   # error individual

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.name = name

    def evaluate(self, costFunc, model):
        """
        Evaluates the current fitness of the particle.

        Args:
        - costFunc (function): a function that calculates the fitness value of the particle
        - model: the PyTorch model to use for calculating the fitness value
        """
        self.cost_i = costFunc(model, self.position_i)

        # check to see if the current position is an individual best
        # best has the highest confidence
        if self.cost_i >= self.cost_best_i or self.cost_best_i == -1:
            self.pos_best_i = self.position_i.clone().detach()
            self.cost_best_i = self.cost_i


                    
    def update_velocity(self, pos_best_g):
        """
        Updates the particle velocity based on its own position and the global best position.

        Args:
        - pos_best_g (list): the global best position
        """
        r1 = np.random.random()
        r2 = np.random.random()

        vel_cognitive = self.c1 * r1 * (self.pos_best_i - self.position_i)
        vel_social = self.c2 * r2 * (pos_best_g - self.position_i)
        self.velocity_i = self.w * self.velocity_i + vel_cognitive + vel_social


    def update_position(self):
        """
        Updates the particle position based on its velocity.
-
        """
        # update position based on velocity
        self.position_i = self.position_i +  self.velocity_i
        

        # clamp position to be within the bounds
        self.position_i = torch.clamp(self.position_i, 0, 1)

        self.history.append(self.position_i)
