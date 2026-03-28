import math
import torch
import appmax.neurons
import appmax.optimize
import tests.test_neurons

def test_deeper_appmax():
    net = tests.test_neurons.DummyNetDeeper()
    sample = torch.tensor([2.0, 1.0])
    bounds = (-0.5, 3.0)
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(net.layers, message, constraints)
    c, bias, A_ub, b_ub = appmax.optimize.prepare_lp(message, constraints)
    appmax.optimize.check_feasibility(sample, A_ub, b_ub, bounds)
    sample_found, err_found = appmax.optimize.optimize(c, bias, A_ub, b_ub, bounds, verbose=False)
    err_computed = net(sample_found.unsqueeze(0)).item()
    assert math.isclose(err_found, err_computed)
