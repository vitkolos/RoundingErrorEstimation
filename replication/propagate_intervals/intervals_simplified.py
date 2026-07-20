import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import SmallDenseNet
from dataset import create_dataset


def n_relu(w):
    return -F.relu(-w)


class Message:
    def __init__(self, shape):
        self.a = torch.full(shape, torch.inf).double()
        self.b = torch.full(shape, -torch.inf).double()
        self.b_old = torch.full(shape, -torch.inf).double()
        self.alpha = torch.zeros(shape).double()
        self.beta = torch.zeros(shape).double()
        self.raw_input = True

    def report(self, layer):
        assert torch.all(self.a <= self.b)
        alpha, beta = self.alpha.flatten(), self.beta.flatten()
        assert torch.all(alpha <= 0)
        assert torch.all(beta >= 0)

        diff = beta-alpha
        imin, imax = torch.argmin(diff), torch.argmax(diff)
        print(f'layer {layer}:', f'min [{alpha[imin].item():.4f}, {beta[imin].item():.4f}]',
              f'max [{alpha[imax].item():.4f}, {beta[imax].item():.4f}]')


def compute_input_ab(class_=None):
    test_data = create_dataset(train=False, batch_size=BATCH_SIZE)
    message = Message(INPUT_SHAPE)

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


def compute_layer_ab(message: Message, w, bias):
    w_neg = n_relu(w)
    w_pos = F.relu(w)

    a_new = bias + message.b @ w_neg.T + message.a @ w_pos.T
    b_new = bias + message.a @ w_neg.T + message.b @ w_pos.T

    message.b_old = message.b
    message.a, message.b = a_new, b_new
    return message


def shift_bias(message: Message, w, bias):
    if message.raw_input:
        bias = bias + message.a @ w.T
        message.b -= message.a
        message.a -= message.a
        message.raw_input = False
        assert torch.all(message.b >= 0)
    else:
        assert torch.all(message.b > 0)

    assert torch.all(message.a == 0)
    return message, bias


def load_network():
    net = MODEL()
    net.load_state_dict(torch.load(NETWORK, map_location='cpu'))
    net.eval().double()
    return net


def hat(x):
    return x.half().double()


def compute_layer_interval(message: Message, weight, bias):
    weight_hat = hat(weight)
    bias_hat = hat(bias)
    delta_weight = weight_hat - weight
    delta_bias = bias_hat - bias

    delta_neg = n_relu(delta_weight)
    delta_pos = F.relu(delta_weight)

    alpha_new = message.b_old @ delta_neg.T
    beta_new = message.b_old @ delta_pos.T

    weight_hat_neg = n_relu(weight_hat)
    weight_hat_pos = F.relu(weight_hat)

    alpha_new += message.alpha @ weight_hat_pos.T
    alpha_new += message.beta @ weight_hat_neg.T

    beta_new += message.alpha @ weight_hat_neg.T
    beta_new += message.beta @ weight_hat_pos.T

    message.alpha = alpha_new + delta_bias
    message.beta = beta_new + delta_bias
    return message


def compute_pool_alpha_beta(alpha, beta, shape, max_):
    alpha_new = alpha.reshape(*shape)
    beta_new = beta.reshape(*shape)
    return -1*max_(-1*alpha_new).flatten(), max_(beta_new).flatten()


def compute_pool_ab(a, b, shape, max_):
    a_new = a.reshape(*shape)
    b_new = b.reshape(*shape)
    return max_(a_new).flatten(), max_(b_new).flatten()


def calculate_intervals(class_):
    with torch.no_grad():
        message = compute_input_ab(class_)
        net = load_network()

        # if type_ == "pool":
        #     alpha, beta = compute_pool_alpha_beta(alpha, beta, input_shape, layer_pointer)
        #     a, b = compute_pool_ab(a, b, input_shape, layer_pointer)

        for i, layer in enumerate(net.network):
            match layer:
                case nn.Dropout():
                    pass
                case nn.Flatten():
                    message.a = layer(message.a)
                    message.b = layer(message.b)
                    message.alpha = layer(message.alpha)
                    message.beta = layer(message.beta)
                case nn.Linear():
                    bias = layer.bias if layer.bias is not None else torch.zeros(layer.weight.shape[0]).double()
                    message, bias = shift_bias(message, layer.weight, bias)
                    message = compute_layer_ab(message, layer.weight, bias)
                    message = compute_layer_interval(message, layer.weight, bias)
                case nn.ReLU():
                    message.a = F.relu(message.a)
                    message.b = F.relu(message.b)
                    message.alpha = n_relu(message.alpha)
                    message.beta = F.relu(message.beta)
                    message.report(i)

        message.report(i)


if __name__ == "__main__":
    BATCH_SIZE = 256
    NETWORK = "models/mnist_dense_net.pt"
    MODEL = SmallDenseNet
    LAYERS = 3
    INPUT_SHAPE = (1, 28, 28)
    INPUT_LENGTH = 1 * 28 * 28

    calculate_intervals(class_=0)
