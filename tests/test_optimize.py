import math
import torch
import torch.nn as nn

import appmax.neurons
import appmax.optimize
import appmax.trainable
import tests.test_neurons


def optimize_testing_procedure(net: appmax.trainable.TrainableModel, sample: torch.Tensor, bounds: appmax.trainable.Bounds, mixing: float = 0.0):
    constraints = appmax.neurons.Constraints()
    constraints.neuron_states = []
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(net.layers, message, constraints)
    lp = appmax.optimize.prepare_lp(message, constraints)
    sample_found, err_found = appmax.optimize.optimize(*lp, bounds, verbose=False)
    sample_found = sample_found.reshape_as(sample)
    sample_comb = (1-mixing)*sample_found + mixing*sample

    # check feasibility
    c, bias, A_ub, b_ub = lp
    appmax.optimize.check_feasibility(sample, A_ub, b_ub, bounds)
    appmax.optimize.check_feasibility(sample_found, A_ub, b_ub, bounds, abs_tol=1e-6)
    appmax.optimize.check_feasibility(sample_comb, A_ub, b_ub, bounds)

    # check found objective value
    err_computed = net(sample_found.unsqueeze(0)).item()
    assert math.isclose(err_found, err_computed, abs_tol=1e-6), 'Optimization returns different value than the network'

    # check if we get the same linear program for a point on the polytope
    constraints_point = appmax.neurons.Constraints()
    constraints_point.neuron_states = []
    message_point = appmax.neurons.Message(sample_comb)
    message_point = appmax.neurons.collect(net.layers, message_point, constraints_point)
    lp_point = appmax.optimize.prepare_lp(message_point, constraints_point)

    for t1, t2 in zip(constraints.neuron_states, constraints_point.neuron_states):
        torch.testing.assert_close(t1, t2, msg='States of neurons differ')

    for t1, t2 in zip(lp, lp_point):
        torch.testing.assert_close(t1, t2, msg='Linear programs differ')


def test_deeper_appmax():
    net = tests.test_neurons.DummyNetDeeper()
    sample = torch.tensor([2.0, 1.0])
    optimize_testing_procedure(net, sample, bounds=(-0.5, 3.0))


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
    optimize_testing_procedure(net, sample, bounds=(-2.0, 3.0), mixing=0.01)


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
    optimize_testing_procedure(net, sample, bounds=(-2.0, 3.0), mixing=0.01)
