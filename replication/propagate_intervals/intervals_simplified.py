import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import SmallDenseNet, SmallConvNet
from dataset import create_dataset


def load_network():
    net = MODEL()
    net.load_state_dict(torch.load(NETWORK, map_location='cpu'))
    net.eval().double()
    return net


class Message:
    def __init__(self, shape):
        self.a = torch.full(shape, torch.inf).double()
        self.b = torch.full(shape, -torch.inf).double()
        self.a_old = torch.full(shape, torch.inf).double()
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


def get_fun(layer: nn.Module):
    match layer:
        case nn.Linear():
            return F.linear
        case nn.Conv2d():
            def conv(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
                return layer._conv_forward(input, weight, bias)
            return conv
        case _:
            raise NotImplementedError


def n_relu(w):
    return -F.relu(-w)


def layer_ab(layer: nn.Module, message: Message):
    assert isinstance(layer.weight, torch.Tensor)
    assert layer.bias is None or isinstance(layer.bias, torch.Tensor)

    w_neg = n_relu(layer.weight)
    w_pos = F.relu(layer.weight)

    fun = get_fun(layer)
    a_new = fun(message.b, w_neg, layer.bias) + fun(message.a, w_pos)
    b_new = fun(message.a, w_neg, layer.bias) + fun(message.b, w_pos)

    message.a_old, message.b_old = message.a, message.b
    message.a, message.b = a_new, b_new
    return message


def tilde(x):
    return x.half().double()


def layer_alpha_beta(layer: nn.Module, m: Message):
    weight_tilde = tilde(layer.weight)
    bias_tilde = tilde(layer.bias)
    weight_delta = weight_tilde - layer.weight
    bias_delta = bias_tilde - layer.bias
    w_delta_neg = n_relu(weight_delta)
    w_delta_pos = F.relu(weight_delta)
    w_tilde_neg = n_relu(weight_tilde)
    w_tilde_pos = F.relu(weight_tilde)

    fun = get_fun(layer)
    alpha_new = fun(m.a_old, w_delta_pos, bias_delta) + fun(m.b_old, w_delta_neg)
    beta_new = fun(m.a_old, w_delta_neg, bias_delta) + fun(m.b_old, w_delta_pos)
    alpha_new += fun(m.alpha, w_tilde_pos) + fun(m.beta, w_tilde_neg)
    beta_new += fun(m.alpha, w_tilde_neg) + fun(m.beta, w_tilde_pos)

    m.alpha, m.beta = alpha_new, beta_new
    return m


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
                    message = layer_ab(layer, message)
                    message = layer_alpha_beta(layer, message)
                case _:
                    raise NotImplementedError

        message.report(i)


if __name__ == "__main__":
    BATCH_SIZE = 256
    NETWORK = "models/mnist_dense_net.pt"
    MODEL = SmallDenseNet
    # NETWORK = "models/mnist_conv_net.pt"
    # MODEL = SmallConvNet

    intervals(class_=0)
