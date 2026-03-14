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


def test_shortcut_weights():
    net = SimpleNet()
    input_data = torch.tensor([[2.0, 1.0]])
    message = appmax.neurons.Message(input_data, torch.eye(input_data.shape[1]), torch.zeros_like(input_data))
    message = appmax.neurons.forward(net, message, [])
    output = input_data @ message.s_weights + message.s_bias
    assert torch.equal(output, net(input_data))
