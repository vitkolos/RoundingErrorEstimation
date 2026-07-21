import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import SmallDenseNet, SmallConvNet
from dataset import create_dataset


def n_relu(w):
    return -F.relu(-w)


def load_network():
    net = MODEL()
    net.load_state_dict(torch.load(NETWORK, map_location='cpu'))
    net.eval().double()
    return net


class Message:
    def __init__(self, shape):
        self.a = torch.full(shape, torch.inf).double()
        self.b = torch.full(shape, -torch.inf).double()
        self.b_old = torch.full(shape, -torch.inf).double()
        self.alpha = torch.zeros(shape).double()
        self.beta = torch.zeros(shape).double()
        self.input_processed = False

    def apply_special(self, module: nn.Module):
        self.a = module(self.a)
        self.b = module(self.b)
        self.alpha = -module(-self.alpha)
        self.beta = module(self.beta)
        return self

    def report(self, layer):
        assert torch.all(self.a <= self.b)
        alpha, beta = self.alpha.flatten(), self.beta.flatten()
        assert torch.all(alpha <= 0)
        assert torch.all(beta >= 0)

        diff = beta-alpha
        imin, imax = torch.argmin(diff), torch.argmax(diff)
        print(f'layer {layer}:', f'min [{alpha[imin].item():.4f}, {beta[imin].item():.4f}]',
              f'max [{alpha[imax].item():.4f}, {beta[imax].item():.4f}]')


def input_ab_message(class_=None):
    test_data = create_dataset(train=False, batch_size=BATCH_SIZE)
    message = Message(shape=(1, *test_data.dataset[0][0].shape))

    for xs, ys in tqdm.tqdm(test_data):
        if class_ is not None:
            xs = xs[ys == class_]

        if len(xs) > 0:
            xs = xs.double()
            a_batch, _ = torch.min(xs, dim=0)
            b_batch, _ = torch.max(xs, dim=0)
            message.a = torch.minimum(message.a, a_batch)
            message.b = torch.maximum(message.b, b_batch)

    return message


def shift_bias(layer: nn.Module, message: Message):
    """accounts for the fact that input values can be below zero (using this new bias we can act as if they are not)"""
    if not message.input_processed:
        bias = layer(message.a)
        message.b -= message.a
        message.a -= message.a
        message.input_processed = True
        assert torch.all(message.b >= 0)
    else:
        bias = layer.bias if layer.bias is not None else torch.zeros(layer.weight.shape[0]).double()

        if isinstance(layer, nn.Conv2d):
            bias = bias[..., None, None]

        # assert torch.all(message.b > 0)

    # assert torch.all(message.a == 0)
    return message, bias


def layer_ab(layer: nn.Module, message: Message, bias: torch.Tensor):
    w_neg = n_relu(layer.weight)
    w_pos = F.relu(layer.weight)

    match layer:
        case nn.Linear():
            a_new = bias + F.linear(message.b, w_neg) + F.linear(message.a, w_pos)
            b_new = bias + F.linear(message.a, w_neg) + F.linear(message.b, w_pos)
        case nn.Conv2d():
            def conv(x, w): return layer._conv_forward(x, w, None)
            a_new = bias + conv(message.b, w_neg) + conv(message.a, w_pos)
            b_new = bias + conv(message.a, w_neg) + conv(message.b, w_pos)

    message.b_old = message.b
    message.a, message.b = a_new, b_new
    return message


def tilde(x):
    return x.half().double()


def layer_alpha_beta(layer: nn.Module, message: Message, bias: torch.Tensor):
    weight_tilde = tilde(layer.weight)
    bias_tilde = tilde(bias)
    weight_delta = weight_tilde - layer.weight
    bias_delta = bias_tilde - bias
    weight_delta_neg = n_relu(weight_delta)
    weight_delta_pos = F.relu(weight_delta)
    weight_tilde_neg = n_relu(weight_tilde)
    weight_tilde_pos = F.relu(weight_tilde)

    match layer:
        case nn.Linear():
            alpha_new = bias_delta + F.linear(message.b_old, weight_delta_neg)
            beta_new = bias_delta + F.linear(message.b_old, weight_delta_pos)
            alpha_new += F.linear(message.alpha, weight_tilde_pos) + F.linear(message.beta, weight_tilde_neg)
            beta_new += F.linear(message.alpha, weight_tilde_neg) + F.linear(message.beta, weight_tilde_pos)
        case nn.Conv2d():
            def conv(x, w): return layer._conv_forward(x, w, None)
            alpha_new = bias_delta + conv(message.b_old, weight_delta_neg)
            beta_new = bias_delta + conv(message.b_old, weight_delta_pos)
            alpha_new += conv(message.alpha, weight_tilde_pos) + conv(message.beta, weight_tilde_neg)
            beta_new += conv(message.alpha, weight_tilde_neg) + conv(message.beta, weight_tilde_pos)

    message.alpha, message.beta = alpha_new, beta_new
    return message


def intervals(class_):
    with torch.no_grad():
        message = input_ab_message(class_)
        net = load_network()

        for i, layer in enumerate(net.network):
            match layer:
                case nn.Dropout():
                    pass
                case nn.Flatten():
                    message = message.apply_special(layer)
                case nn.ReLU() | nn.MaxPool2d():
                    message = message.apply_special(layer)
                    message.report(i)
                case nn.Linear() | nn.Conv2d():
                    message, bias = shift_bias(layer, message)
                    message = layer_ab(layer, message, bias)
                    message = layer_alpha_beta(layer, message, bias)

        message.report(i)


if __name__ == "__main__":
    BATCH_SIZE = 256
    NETWORK = "models/mnist_dense_net.pt"
    MODEL = SmallDenseNet
    # NETWORK = "models/mnist_conv_net.pt"
    # MODEL = SmallConvNet

    intervals(class_=0)
