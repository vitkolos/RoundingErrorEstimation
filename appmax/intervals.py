import click
import torch
import torch.nn as nn
import torch.nn.functional as F

import appmax.trainable
import appmax.logger
import appmax.applications

BATCH_SIZE = 256


@click.command()
@click.argument('dataset', default=None)
@click.option('-b', '--bits', default=8, help='Number of bits used by the quantized network (default: 8).')
def main(dataset, bits):
    if dataset is not None:
        return intervals_wrapper(dataset, bits)


def intervals_wrapper(dataset, bits, verbose=True, bundle=None):
    if bundle is None:
        bundle = appmax.applications.DataBundle(dataset)

    data_split = bundle.data_split
    input_shape = data_split.test[0][0].shape
    bounds_ab = torch.tensor(data_split.metadata.bounds.lb).to(dtype=torch.get_default_dtype()), \
        torch.from_numpy(data_split.metadata.bounds.ub).to(dtype=torch.get_default_dtype())
    model = bundle.load_model()
    model_approx = bundle.load_model()
    model_approx.round(bits=bits)

    if dataset == 'mnist':
        print('mnist: reference solution (using torch.nn.Module.half, considering only target=0)')
        mask = data_split.test.targets == 0  # selects only this class
        bounds_ab = find_ab_from_dataset(data_split.test, mask)
        model_approx = bundle.load_model()
        model_approx.round(bits=16, qt='torch')
        model.layers, model_approx.layers = model.network, model_approx.network
    elif verbose:
        print(f'{dataset}: {bits}bit')

    return find_intervals(input_shape, bounds_ab, model.layers, model_approx.layers, verbose)


def find_ab_from_dataset(dataset: appmax.trainable.Dataset, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is not None:
        indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    shape = (1, *dataset[0][0].shape)
    a = torch.full(shape, torch.inf)
    b = torch.full(shape, -torch.inf)

    for xs, ys in appmax.logger.progress(loader):
        a_batch, _ = torch.min(xs, dim=0)
        b_batch, _ = torch.max(xs, dim=0)
        a = torch.minimum(a, a_batch)
        b = torch.maximum(b, b_batch)

    return a, b


@torch.no_grad()
def find_intervals(
    input_shape: torch.Size,
    bounds_ab: tuple[torch.Tensor, torch.Tensor],
    layers: torch.nn.Sequential,
    layers_approx: torch.nn.Sequential,
    verbose: bool,
):
    message = Message(shape=torch.Size([1, *input_shape]), a=bounds_ab[0], b=bounds_ab[1])
    reports = []

    for i, (layer, layer_approx) in enumerate(zip(layers, layers_approx)):
        if type(layer) is not type(layer_approx):
            raise ValueError('layers at the same level have different types')

        if type(layer) is nn.BatchNorm1d and type(layer_approx) is nn.BatchNorm1d:
            layer = bn1d_to_linear(layer)
            layer_approx = bn1d_to_linear(layer_approx)

        match layer:
            case nn.Dropout():
                pass
            case nn.Flatten():
                message = message.apply_special(layer)
            case nn.ReLU() | nn.MaxPool2d():
                message = message.apply_special(layer)
                reports.append(message.report(i, verbose))
            case nn.Linear() | nn.Conv2d():
                message = layer_ab(layer, message)
                message = layer_alpha_beta(layer, layer_approx, message)
            case _:
                raise NotImplementedError(
                    f"intervals.find_intervals is not implemented for '{type(layer).__name__}' object")

    reports.append(message.report(i, verbose))
    return reports


class Message:
    def __init__(self, shape: torch.Size, a: torch.Tensor, b: torch.Tensor):
        self.a = a.reshape(shape)
        self.b = b.reshape(shape)
        self.a_old = torch.empty(0)
        self.b_old = torch.empty(0)
        self.alpha = torch.zeros(shape)
        self.beta = torch.zeros(shape)
        self.input_processed = False

    def apply_special(self, module: nn.Module):
        self.a = module(self.a)
        self.b = module(self.b)
        self.alpha = -module(-self.alpha)
        self.beta = module(self.beta)
        return self

    def report(self, layer: int, verbose: bool):
        assert torch.all(self.a <= self.b)
        alpha, beta = self.alpha.flatten(), self.beta.flatten()
        assert torch.all(alpha <= 0)
        assert torch.all(beta >= 0)

        diff = beta-alpha
        imin, imax = torch.argmin(diff), torch.argmax(diff)

        if verbose:
            print(f'layer {layer}:', f'min [{alpha[imin].item():.4f}, {beta[imin].item():.4f}]',
                  f'max [{alpha[imax].item():.4f}, {beta[imax].item():.4f}]')

        return {'min': (alpha[imin].item(), beta[imin].item()), 'max': (alpha[imax].item(), beta[imax].item())}


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
    assert isinstance(layer.weight, torch.Tensor) and isinstance(layer.bias, torch.Tensor) \
        and isinstance(layer_approx.weight, torch.Tensor) and isinstance(layer_approx.bias, torch.Tensor)

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
    if layer.training:
        raise RuntimeError('BatchNorm1d layer is in training mode')
    elif not layer.track_running_stats or layer.running_mean is None or layer.running_var is None:
        raise NotImplementedError('bn1d_to_linear does not support track_running_stats=False')

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
