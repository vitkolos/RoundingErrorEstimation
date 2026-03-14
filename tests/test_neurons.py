import torch
import torch.nn as nn
import appmax.trainable
import appmax.neurons


class SimpleNet(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                fc1 := nn.Linear(2, 3),
                nn.ReLU(),
                fc2 := nn.Linear(3, 2),
            )
        )

        with torch.no_grad():
            fc1.weight = nn.Parameter(torch.tensor([
                [1.0,  1.0],  # hidden neuron 1
                [-1.0, -1.0],  # hidden neuron 2
                [0.5, -0.5]   # hidden neuron 3
            ]))
            fc1.bias = nn.Parameter(torch.tensor([0.5, 0, 0]))
            fc2.weight = nn.Parameter(torch.tensor([
                [1.0, 1.0, 1.0],  # output neuron 1
                [1.0, 0.0, 0.0]  # output neuron 2
            ]))
            fc2.bias = nn.Parameter(torch.zeros(2))


def test_shortcut_weights_constraints():
    net = SimpleNet()
    sample = torch.tensor([[2.0, 1.0]])
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(net, message, constraints)
    output = sample @ message.s_weight + message.s_bias
    assert torch.equal(output, net(sample))
    assert torch.equal(output, message.sample)
    assert torch.equal(constraints.S_weight[0], torch.tensor([[-1., -1.]]))
