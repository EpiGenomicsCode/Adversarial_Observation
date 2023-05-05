from Swarm_Observer.Swarm import PSO
import unittest
import torch

class TestSwarm(unittest.TestCase):

    def test_pso_init(self):
        starting_positions = torch.randn((10, 2))
        cost_func = lambda model, pos: model(pos).sum()
        model = torch.nn.Linear(2, 1)
        pso = PSO(starting_positions, cost_func, model)
        pso.run(10)
        assert isinstance(pso.swarm, list)
        assert len(pso.swarm) == 10
        assert isinstance(pso.history, list)
        assert isinstance(pso.pos_best_g, torch.Tensor)
        assert pso.pos_best_g.shape == starting_positions[0].shape
        assert isinstance(pso.cos_best_g, torch.Tensor)