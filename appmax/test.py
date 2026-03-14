import torch
import torch.nn as nn
import network_train
import neurons


class SimpleNet(network_train.TrainableModel):
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


def main():
    net = SimpleNet()
    input_data = torch.tensor([[2.0, 1.0]])
    print('neuron states')
    print(x := input_data)

    for layer in net.layers:
        print(x := layer(x))

    print('manual mult')
    print(x := input_data)

    for layer in net.layers:
        if type(layer) == nn.ReLU:
            print(x := layer(x))
            unsaturated = x > 0
        else:
            print(x := x @ layer.weight.t() + layer.bias)

    print('shortcut weights')
    print(W := torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    print(W := W @ net.layers[0].weight.t())
    print(W := W * unsaturated)
    print(W := W @ net.layers[2].weight.t())
    
    print('shortcut bias')
    print(b := torch.tensor([[0.0, 0.0]]))
    print(b := b @ net.layers[0].weight.t() + net.layers[0].bias)
    print(b := b * unsaturated)
    print(b := b @ net.layers[2].weight.t() + net.layers[2].bias)
    
    print('apply')
    print(input_data @ W + b)

    message = neurons.Message(input_data, torch.eye(input_data.shape[1]), torch.zeros_like(input_data))
    message = neurons.forward(net, message, [])
    print(message)
    print('apply')
    print(input_data @ message.s_weights + message.s_bias)


if __name__ == '__main__':
    main()
