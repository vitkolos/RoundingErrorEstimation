import math
import torch
import torch.nn as nn

import appmax.neurons
import appmax.optimization
import appmax.solving
from appmax.solving import SOLVER_DEFAULT
import appmax.trainable
from appmax.trainable import Bounds
import tests.test_neurons


def optimize_testing_procedure(net: nn.Module, sample: torch.Tensor, bounds: appmax.trainable.Bounds, mixing: float = 0.0, solver: str = SOLVER_DEFAULT):
    net_collectable = net.layers if isinstance(net, appmax.trainable.TrainableModel) else net
    constraints = appmax.neurons.Constraints()
    constraints.neuron_states = []
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(net_collectable, message, constraints)
    lp = appmax.optimization.prepare_lp(message, constraints, bounds)
    sample_found, err_found = appmax.solving.solve(lp, solver)
    sample_found = sample_found.reshape_as(sample)
    sample_comb = (1-mixing)*sample_found + mixing*sample

    # check feasibility
    appmax.optimization.check_feasibility(sample, lp)
    appmax.optimization.check_feasibility(sample_found, lp)
    appmax.optimization.check_feasibility(sample_comb, lp)

    # check found objective value
    err_computed = net(sample_found.unsqueeze(0)).item()
    assert math.isclose(err_found, err_computed, abs_tol=1e-6), 'Optimization returns different value than the network'

    # check if we get the same linear program for a point on the polytope
    constraints_point = appmax.neurons.Constraints()
    constraints_point.neuron_states = []
    message_point = appmax.neurons.Message(sample_comb)
    message_point = appmax.neurons.collect(net_collectable, message_point, constraints_point)
    lp_point = appmax.optimization.prepare_lp(message_point, constraints_point, bounds)

    for t1, t2 in zip(constraints.neuron_states, constraints_point.neuron_states):
        torch.testing.assert_close(t1, t2, msg='States of neurons differ')

    for attr in ['A_ub', 'b_ub', 'objective', 'bias']:
        torch.testing.assert_close(getattr(lp, attr), getattr(lp_point, attr), msg=f'Linear programs differ in {attr}')


def test_deeper_appmax():
    net = tests.test_neurons.DummyNetDeeper()
    sample = torch.tensor([2.0, 1.0])
    optimize_testing_procedure(net, sample, bounds=Bounds([(-0.5, 3.0)]*2))


class DummyOptNetMaxPool(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.MaxPool2d(2),
            )
        )


def test_max_pool_lp():
    net = DummyOptNetMaxPool()
    # 1 channel, 4×4 input shape
    sample = torch.testing.make_tensor((1, 4, 4), dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    optimize_testing_procedure(net, sample, bounds=Bounds([(-2.0, 3.0)]*16), mixing=0.01)


class DummyOptNetConv(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Conv2d(1, 2, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(8, 1)
            )
        )


def test_conv_lp():
    """relies on random initialization of parameters"""
    net = DummyOptNetConv()
    # 1 channel, 4×4 input shape
    sample = torch.testing.make_tensor((1, 4, 4), dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    optimize_testing_procedure(net, sample, bounds=Bounds([(-2.0, 3.0)]*16), mixing=0.01)
