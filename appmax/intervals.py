import click
import torch
import torch.nn as nn
import torch.nn.functional as F

import appmax.trainable
import appmax.logger
import appmax.applications

BATCH_SIZE = 256


@click.command()
@click.argument('dataset')
@click.option('-b', '--bits', default=8, help='Number of bits used by the quantized network (default: 8).')
def main(dataset, bits):
    bundle = appmax.applications.DataBundle(dataset)
    data_split = bundle.data_split
    data_test = torch.utils.data.ConcatDataset([data_split.train, data_split.dev, data_split.test])
    model = bundle.load_model()
    model_approx = bundle.load_model()
    model_approx.round(bits=bits)

    if dataset == 'mnist':
        print('mnist: reference solution (using torch.nn.Module.half, considering only target=0)')
        model_approx = bundle.load_model()
        model_approx.round(bits=16, qt='torch')
        selected = data_split.test.targets == 0  # selects only this class
        indices = torch.nonzero(selected, as_tuple=True)[0].tolist()
        data_test = torch.utils.data.Subset(data_split.test, indices)
        model.layers, model_approx.layers = model.network, model_approx.network
    else:
        print(f'{dataset}: {bits}bit')

    find_intervals(data_test, model.layers, model_approx.layers)
    print()


def find_intervals(dataset: appmax.trainable.Dataset, layers: torch.nn.Sequential, layers_approx: torch.nn.Sequential):
    with torch.no_grad():
        message = input_ab_message(dataset)

        for i, (layer, layer_approx) in enumerate(zip(layers, layers_approx)):
            if type(layer) is nn.BatchNorm1d:
                layer = bn1d_to_linear(layer)
                layer_approx = bn1d_to_linear(layer_approx)

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
                    message = layer_alpha_beta(layer, layer_approx, message)
                case _:
                    raise NotImplementedError(
                        f"intervals.find_intervals is not implemented for '{type(layer).__name__}' object")

        message.report(i)


class Message:
    def __init__(self, shape):
        self.a = torch.full(shape, torch.inf)
        self.b = torch.full(shape, -torch.inf)
        self.a_old = torch.full(shape, torch.inf)
        self.b_old = torch.full(shape, -torch.inf)
        self.alpha = torch.zeros(shape)
        self.beta = torch.zeros(shape)
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


def input_ab_message(dataset: appmax.trainable.Dataset):
    """finds ranges (intervals) of the input neurons"""
    message = Message(shape=(1, *dataset[0][0].shape))
    test_data = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    for xs, ys in appmax.logger.progress(test_data):
        a_batch, _ = torch.min(xs, dim=0)
        b_batch, _ = torch.max(xs, dim=0)
        message.a = torch.minimum(message.a, a_batch)
        message.b = torch.maximum(message.b, b_batch)

    return message


def get_fun(layer: nn.Module):
    """for the given module, return a function with the required interface (input, weight, bias? -> output)"""
    match layer:
        case nn.Linear():
            return F.linear
        case nn.Conv2d():
            def conv(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
                return layer._conv_forward(input, weight, bias)
            return conv
        case _:
            raise NotImplementedError(f"intervals.get_fun is not implemented for '{type(layer).__name__}' object")


def n_relu(w):
    return -F.relu(-w)


def layer_ab(layer: nn.Module, message: Message):
    """current layer modifies the intervals of the inputs of the next layer"""
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


def layer_alpha_beta(layer: nn.Module, layer_approx: nn.Module, m: Message):
    """finds the error bounds of the neurons in the current layer"""
    if type(layer) is not type(layer_approx):
        raise ValueError('layer_approx has a different type than layer')

    weight_tilde = layer_approx.weight
    bias_tilde = layer_approx.bias
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


def bn1d_to_linear(layer: nn.BatchNorm1d) -> nn.Linear:
    """convert a BatchNorm1d layer to a Linear layer, so that the layer has only two sets of parameters (weight and bias)"""
    # old parameters
    gamma = layer.weight
    beta = layer.bias
    mu = layer.running_mean
    var = layer.running_var
    eps = layer.eps

    # new parameters
    w = gamma / torch.sqrt(var + eps)
    b = beta - mu * w

    # new layer
    linear = nn.Linear(layer.num_features, layer.num_features)
    linear.weight.data = torch.diag(w)
    linear.bias.data = b
    return linear.eval()


if __name__ == '__main__':
    main()
