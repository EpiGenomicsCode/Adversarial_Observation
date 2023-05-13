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

        self.position_i = position.clone().detach().numpy()
        self.velocity_i = np.random.random(position.shape)  # velocity
        # copy the current position to the best position

        self.history = [self.position_i]
        
        self.pos_best_i = position.clone()   # best position individual
        self.cost_best_i = -1   # best error individual
        self.cost_i = -1   # error individual

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.name = name

    def evaluate(self, costFunc: callable, model: torch.nn.Module):
        """
        Evaluates the current fitness of the particle.

        Args:
            costFunc (callable): The cost function to be maximized.
                This should be a function that takes in a PyTorch model and a tensor of positions and returns a tensor of shape (n, 1) where n is the number of particles.

            model (torch.nn.Module): The model to be used in the cost function.
            
        """
        self.cost_i = costFunc(model, self.position_i)

        # check to see if the current position is an individual best
        # best has the highest confidence
        if self.cost_i >= self.cost_best_i or self.cost_best_i == -1:
            self.pos_best_i = self.position_i
            self.cost_best_i = self.cost_i


                    
    def update_velocity(self, pos_best_g: list):
        """
        Updates the particle velocity based on its own position and the global best position.

        Args:
            pos_best_g (list): the global best position
        """
        r1 = np.random.random()
        r2 = np.random.random()

        vel_cognitive = self.c1 * r1 * (self.pos_best_i - self.position_i)
        vel_social = self.c2 * r2 * (pos_best_g - self.position_i)
        self.velocity_i = self.w * self.velocity_i + vel_cognitive + vel_social


    def update_position(self):
        """
        Updates the particle position based on its velocity.-
        """
        # update position based on velocity
        self.position_i = self.position_i +  self.velocity_i

        # add current position to history
        self.history.append(self.position_i)

    def get_history(self):
        """
        Returns the history of the particle's positions.

        Args:
            None

        Returns:
            list: The history of the particle's positions.
        """
        return self.history