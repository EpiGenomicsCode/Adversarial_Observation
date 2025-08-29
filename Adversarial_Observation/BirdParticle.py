import torch
import torch.nn.functional as F
import numpy as np

class BirdParticle:
    """
    Represents a particle in the Particle Swarm Optimization (PSO) algorithm for adversarial attacks (PyTorch version).
    """

    def __init__(self, model: torch.nn.Module, input_data: torch.Tensor, target_class: int, num_iterations: int = 20,
                 velocity: torch.Tensor = None, inertia_weight: float = 0.5, 
                 cognitive_weight: float = 1.0, social_weight: float = 1.0, 
                 momentum: float = 0.9, clip_value_position: float = 1.0, device='cpu'):
        """
        Initialize a particle in the PSO algorithm.
        
        Args:
            model (torch.nn.Module): The model to attack.
            input_data (torch.Tensor): The input data (image) to attack.
            target_class (int): The target class for misclassification.
            velocity (torch.Tensor, optional): Initial velocity; defaults to zero.
            inertia_weight (float): Inertia weight for velocity update.
            cognitive_weight (float): Cognitive weight for velocity update.
            social_weight (float): Social weight for velocity update.
            momentum (float): Momentum for velocity update.
            clip_value_position (float): Max absolute value to clip position.
            device (str): Device to run on.
        """
        self.device = device
        self.model = model.to(device)
        self.num_iterations = num_iterations
        self.original_data = input_data.clone().detach().to(device)
        self.position = input_data.clone().detach().to(device)
        self.target_class = target_class
        self.velocity = velocity.clone().detach().to(device) if velocity is not None else torch.zeros_like(input_data).to(device)
        self.best_position = self.position.clone().detach()
        self.best_score = -np.inf
        self.history = [self.position.clone().detach()]
        self.clip_value_position = clip_value_position

        # PSO hyperparameters
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.momentum = momentum

    def fitness(self) -> float:
        """
        Compute the fitness score for the particle, which is the softmax probability of the target class.
        Returns:
            float: Target class softmax probability.
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.position 
            output = self.model(input_tensor.to(self.device))
            probabilities = F.softmax(output, dim=1)
            target_prob = probabilities[:, self.target_class]
            return target_prob.item()

    def update_velocity(self, global_best_position: torch.Tensor) -> None:
        """
        Update the particle's velocity using the PSO rule.
        
        Args:
            global_best_position (torch.Tensor): Global best position in the swarm.
        """
        r1 = torch.rand_like(self.position).to(self.device)
        r2 = torch.rand_like(self.position).to(self.device)
        
        inertia = self.inertia_weight * self.velocity
        cognitive = self.cognitive_weight * r1 * (self.best_position - self.position)
        social = self.social_weight * r2 * (global_best_position.to(self.device) - self.position)
        
        self.velocity = self.momentum * self.velocity + inertia + cognitive + social

    def update_position(self) -> None:
        """
        Update the particle's position based on the new velocity.
        """
        self.position = self.position + self.velocity
        self.position = torch.clamp(self.position, 0.0, 1.0)  # Keep values in [0, 1]
        self.position = torch.clamp(self.position, -self.clip_value_position, self.clip_value_position)
        self.history.append(self.position.clone().detach())

    def evaluate(self) -> None:
        """
        Evaluate current fitness and update personal best if needed.
        """
        score = self.fitness()
        if score > self.best_score:
            self.best_score = score
            self.best_position = self.position.clone().detach()
