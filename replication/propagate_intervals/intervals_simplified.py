import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import SmallDenseNet
from dataset import create_dataset

ZERO = torch.tensor(0.0).double()


def negative(w):
    return torch.minimum(ZERO, w)


def positive(w):
    return torch.maximum(ZERO, w)


def compute_input_ai_bi(class_=None):
    test_data = create_dataset(train=False, batch_size=BATCH_SIZE)
    a = torch.full((INPUT_LENGTH,), torch.inf).double()
    b = torch.full((INPUT_LENGTH,), -torch.inf).double()

    for xs, ys in tqdm.tqdm(test_data):
        if class_ is not None:
            xs = xs[ys == class_]

        if len(xs) > 0:
            xs = xs.double().flatten(1)
            a_batch, _ = torch.min(xs, dim=0)
            b_batch, _ = torch.max(xs, dim=0)
            a = torch.minimum(a, a_batch)
            b = torch.maximum(b, b_batch)

    return a, b


def compute_output_ai_bi(a, b, w, bias, relu=True):
    w_neg = negative(w)
    w_pos = positive(w)

    a_new = bias + torch.matmul(w_neg, b) + torch.matmul(w_pos, a)
    b_new = bias + torch.matmul(w_neg, a) + torch.matmul(w_pos, b)

    if relu:
        return F.relu(a_new), F.relu(b_new)
    else:
        return a_new, b_new


def compute_bias_shift(a, b, w, bias):
    bias = bias + torch.matmul(w, a)
    a, b = a-a, b-a
    return bias, a, b


def load_network():
    net = MODEL()
    net.load_state_dict(torch.load(NETWORK, map_location='cpu'))
    net.eval().double()
    return net


def get_layer(net, layer):
    x = torch.zeros(INPUT_SHAPE).unsqueeze(0).double()
    input_size = INPUT_SHAPE
    layer_pointer = None

    for child in net.children():
        assert isinstance(child, nn.Sequential)
        i = 0  # count only conv, fc and pool

        for l in child.children():
            x = l(x)

            if isinstance(l, nn.Dropout) or isinstance(l, nn.Flatten) or isinstance(l, nn.ReLU):
                pass
            elif i < layer:
                input_size = x.shape
                i += 1
            else:
                layer_pointer = l
                break

    if type(layer_pointer) == nn.Linear:
        return "lin", layer_pointer.weight, layer_pointer.bias, input_size, layer_pointer

    if type(layer_pointer) == nn.MaxPool2d:
        return "pool", None, None, input_size, layer_pointer

    raise ValueError(f"{layer} is wierd")


def delta(x):
    return hat(x) - x


def hat(x):
    return x.half().double()


def compute_output_alpha_beta(alpha, beta, b, delta_weight, delta_bias, weight, bias, relu=True):
    weight_hat = hat(weight)
    bias_hat = hat(bias)

    delta_neg = negative(delta_weight)
    delta_pos = positive(delta_weight)

    alpha_new = torch.matmul(delta_neg, b)
    beta_new = torch.matmul(delta_pos, b)

    weight_hat_neg = negative(weight_hat)
    weight_hat_pos = positive(weight_hat)

    alpha_new += torch.matmul(weight_hat_pos, alpha)
    alpha_new += torch.matmul(weight_hat_neg, beta)

    beta_new += torch.matmul(weight_hat_neg, alpha)
    beta_new += torch.matmul(weight_hat_pos, beta)

    if relu:
        return (negative(alpha_new + delta_bias), F.relu(beta_new + delta_bias))
    else:
        return alpha_new, beta_new


def compute_pool_alpha_beta(alpha, beta, shape, max_):
    alpha_new = alpha.reshape(*shape)
    beta_new = beta.reshape(*shape)
    return -1*max_(-1*alpha_new).flatten(), max_(beta_new).flatten()


def compute_pool_ai_bi(a, b, shape, max_):
    a_new = a.reshape(*shape)
    b_new = b.reshape(*shape)
    return max_(a_new).flatten(), max_(b_new).flatten()


def calculate_output_intervals(class_):
    with torch.no_grad():
        a, b = compute_input_ai_bi(class_)
        alpha = torch.zeros(INPUT_LENGTH).double()
        beta = torch.zeros(INPUT_LENGTH).double()
        net = load_network()

        for layer in range(LAYERS):
            type_, weight, bias, input_shape, layer_pointer = get_layer(net, layer)

            if type_ == "pool":
                alpha, beta = compute_pool_alpha_beta(alpha, beta, input_shape, layer_pointer)
                a, b = compute_pool_ai_bi(a, b, input_shape, layer_pointer)
            else:
                bias, a, b = compute_bias_shift(a, b, weight, bias)
                delta_weight, delta_bias = delta(weight), delta(bias)
                alpha, beta = compute_output_alpha_beta(
                    alpha, beta, b, delta_weight, delta_bias, weight, bias, relu=(layer != (LAYERS-1)))
                a, b = compute_output_ai_bi(a, b, weight, bias, relu=layer != (LAYERS-1))
                del weight
                del bias

            assert torch.all(a <= b)
            assert torch.all(alpha <= 0)
            assert torch.all(beta >= 0)

            diff = beta-alpha
            imin, imax = torch.argmin(diff), torch.argmax(diff)
            print(f'layer {layer+1}:', f'min [{alpha[imin].item():.4f}, {beta[imin].item():.4f}]',
                  f'max [{alpha[imax].item():.4f}, {beta[imax].item():.4f}]')


if __name__ == "__main__":
    BATCH_SIZE = 256
    NETWORK = "models/mnist_dense_net.pt"
    MODEL = SmallDenseNet
    LAYERS = 3
    INPUT_SHAPE = (1, 28, 28)
    INPUT_LENGTH = 1 * 28 * 28

    calculate_output_intervals(class_=0)
