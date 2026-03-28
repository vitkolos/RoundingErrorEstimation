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
