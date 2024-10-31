from Swarm_Observer.Swarm import PSO
import unittest
import torch
import numpy as np

# Define the cost function for optimization
def costFunc(model, x):
    return 1 - sum(model(x))

class TestSwarm(unittest.TestCase):
    def test_pso_init(self):
        points = 10
        # Initialize random starting positions for the swarm
        starting_positions = torch.rand(size=(points, 2), dtype=torch.float32)

        # Define the function to minimize (for example, a simple quadratic function)
        func = lambda x: x**2

        # Initialize the PSO instance
        swarm = PSO(starting_positions, costFunc, func)

        # Run the optimization process
        swarm.run(10)

        # Check that the swarm has the correct number of particles
        self.assertEqual(len(swarm.swarm), points)
        
        # Check if the best global cost is close to 1
        self.assertTrue(np.isclose(swarm.cos_best_g.item(), 1, atol=0.1))

if __name__ == '__main__':
    unittest.main()
