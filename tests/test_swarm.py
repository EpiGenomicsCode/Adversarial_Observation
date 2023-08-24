from Swarm_Observer.Swarm import PSO
import unittest
import torch
import numpy as np

def costFunc(model, x):
    """
    Cost function to be maximized.

    """
    return 1 - sum(model(x))

class TestSwarm(unittest.TestCase):
    
    def test_pso_init(self):
        points = 10
        starting_positions = torch.rand(size=(points, 2)).to(torch.float32)

        func = lambda x: x**2
        swarm = PSO(starting_positions, costFunc, func)

        swarm.run(10)

        self.assertEqual(len(swarm.swarm), 10)
        print(swarm.pos_best_g)
        print(swarm.cos_best_g)
        # tolerance of .001 
        assert np.isclose(swarm.cos_best_g, 0, atol=.001)

if __name__ == '__main__':
    unittest.main()