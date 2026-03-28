import math
import torch
import torch.nn as nn

import appmax.neurons
import appmax.optimize
import appmax.trainable
import tests.test_neurons


def optimize_testing_procedure(net: appmax.trainable.TrainableModel, sample: torch.Tensor, bounds: appmax.trainable.Bounds):
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(net.layers, message, constraints)
    lp = appmax.optimize.prepare_lp(message, constraints)
    sample_found, err_found = appmax.optimize.optimize(*lp, bounds, verbose=False)

    # check feasibility
    c, bias, A_ub, b_ub = lp
    appmax.optimize.check_feasibility(sample, A_ub, b_ub, bounds)
    appmax.optimize.check_feasibility(sample_found, A_ub, b_ub, bounds)

    # check found objective value
    err_computed = net(sample_found.unsqueeze(0)).item()
    assert math.isclose(err_found, err_computed), 'Optimization returns different value than the network'

    # check if we get the same linear program for the extreme point on the polytope
    constraints_found = appmax.neurons.Constraints()
    message_found = appmax.neurons.Message(sample_found)
    message_found = appmax.neurons.collect(net.layers, message_found, constraints_found)
    lp_found = appmax.optimize.prepare_lp(message_found, constraints_found)

    for t1, t2 in zip(lp, lp_found):
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
    # 7 channels, 4×4 input shape
    sample = torch.testing.make_tensor((1, 4, 4), dtype=torch.float32, device='cpu', low=-1.0, high=2.0)
    optimize_testing_procedure(net, sample, bounds=(-2.0, 3.0))
