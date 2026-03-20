import torch
import torch.nn as nn
import appmax.trainable
import appmax.neurons


def net_param(list):
    return nn.Parameter(torch.tensor(list), requires_grad=False)


class DummyNetShallow(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                fc1 := nn.Linear(2, 3),
                nn.ReLU(),
                fc2 := nn.Linear(3, 2),
            )
        )

        with torch.no_grad():
            fc1.weight = net_param([
                [1.0,  1.0],  # hidden neuron 1
                [-1.0, -1.0],  # hidden neuron 2
                [0.5, -0.5]   # hidden neuron 3
            ])
            fc1.bias = net_param([0.5, 0.0, 0.0])
            fc2.weight = net_param([
                [1.0, 1.0, 1.0],  # output neuron 1
                [1.0, 0.0, 0.0]  # output neuron 2
            ])
            fc2.bias = net_param([0.0, 0.0])


class DummyNetDeeper(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                fc1 := nn.Linear(2, 3),
                nn.ReLU(),
                fc2 := nn.Linear(3, 2),
                nn.ReLU(),
                fc3 := nn.Linear(2, 1, bias=False),
            )
        )

        with torch.no_grad():
            fc1.weight = net_param([
                [1.0,  1.0],  # H1.1
                [-1.0, -1.0],  # H1.2
                [0.5, -0.5]   # H1.3
            ])
            fc1.bias = net_param([0.5, 0, 0])
            fc2.weight = net_param([
                [1.0, 1.0, 1.0],  # H2.1
                [1.0, 0.0, 0.0]  # H2.2
            ])
            fc2.bias = net_param([0.25, 0.25])
            fc3.weight = net_param([
                [1.0, 2.0],  # O1.1
            ])


def test_shallow_neurons():
    net = DummyNetShallow()
    sample = torch.tensor([[2.0, 1.0]])
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(net.layers, message, constraints)
    output = sample @ message.s_weight + message.s_bias
    assert torch.equal(output, net(sample))
    assert torch.equal(output, message.sample)
    assert torch.equal(constraints.S_weight[0], torch.tensor([[-1., -1.]])), \
        'weights corresponding to the (saturated) hidden neuron 2 should be (-1, -1)'


def test_deeper_constraints():
    net = DummyNetDeeper()
    sample = torch.tensor([[2.0, 1.0]])
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(net.layers, message, constraints)
    output = sample @ message.s_weight + message.s_bias
    assert torch.equal(output, net(sample))
    assert torch.equal(output, message.sample)
    assert torch.equal(constraints.U_weight[1][0], torch.tensor([1.5, 0.5])), \
        'weights corresponding to the (unsaturated) neuron H2.1 should be (1.0 + 0.5, 1.0 - 0.5)'
    assert torch.equal(constraints.U_bias[1][0], torch.tensor(0.75)), \
        'bias corresponding to the (unsaturated) neuron H2.1 should be 0.5 + 0.25'
