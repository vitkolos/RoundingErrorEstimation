from dataclasses import dataclass, field
import copy
import warnings

import torch
from torch import nn
import appmax.evaluation


@dataclass
class Message:
    sample: torch.Tensor
    s_weight: torch.Tensor
    s_bias: torch.Tensor

    def __init__(self, sample: torch.Tensor):
        """'sample' needs to be a single sample (not a batch)"""
        self.sample = sample.unsqueeze(0)  # batch-like
        # create a unit matrix with a shape corresponding to the input
        self.s_weight = torch.eye(sample.numel()).reshape((-1, *sample.shape))
        self.s_bias = torch.zeros_like(self.sample)

    def apply(self, module: nn.Module):
        self.sample = module(self.sample)
        self.s_weight = module(self.s_weight)
        self.s_bias = module(self.s_bias)
        return self

    def cat(self, other: 'Message'):
        self.sample = torch.cat((self.sample, other.sample), dim=-1)
        self.s_weight = torch.cat((self.s_weight, other.s_weight), dim=-1)
        self.s_bias = torch.cat((self.s_bias, other.s_bias), dim=-1)
        return self


@dataclass
class Constraints:
    U_weight: list[torch.Tensor] = field(default_factory=list)
    U_bias: list[torch.Tensor] = field(default_factory=list)
    S_weight: list[torch.Tensor] = field(default_factory=list)
    S_bias: list[torch.Tensor] = field(default_factory=list)


@torch.no_grad()
def collect(module: nn.Module, message: Message, constraints: Constraints) -> Message:
    """passes message through the module and collects necessary information"""
    match module:
        case appmax.evaluation.DualStreamModel():
            message2 = copy.deepcopy(message)
            message = collect(module.first_stream, message, constraints)
            message2 = collect(module.second_stream, message2, constraints)
            message = message.cat(message2)
            return collect(module.common_stream, message, constraints)
        case nn.Sequential():
            for submodule in module:
                message = collect(submodule, message, constraints)
            return message
        case nn.ReLU():
            return collect_relu(module, message, constraints)
        case nn.Linear():
            return collect_linear(module, message, constraints)
        case nn.Conv2d():
            return collect_conv2d(module, message, constraints)
        case nn.MaxPool2d():
            return collect_max_pool2d(module, message, constraints)
        case nn.Dropout():
            if module.training:
                warnings.warn('Dropout layer is in training mode')
            return message
        case nn.Flatten():
            return message.apply(module)

    raise NotImplementedError(f"neurons.collect is not implemented for '{type(module).__name__}' object")


def collect_relu(relu: nn.ReLU, message: Message, constraints: Constraints) -> Message:
    # find unsaturated neurons
    unsaturated = message.sample >= 0
    # - non-strict inequality ensures that the unsaturated set stays the same for all the samples in the polytope
    # - strict inequality would cause smaller unsaturated set for samples from some of the edges of the polytope
    #   -> it would be possible to "move" from one polytope to another (but only in one direction)
    unsaturated_sq = unsaturated.squeeze(0)
    message.sample = relu(message.sample)

    # add constraints
    s_weight_T = message.s_weight.movedim(0, -1)
    constraints.U_weight.append(s_weight_T[unsaturated_sq])
    constraints.U_bias.append(message.s_bias[unsaturated])
    constraints.S_weight.append(s_weight_T[~unsaturated_sq])
    constraints.S_bias.append(message.s_bias[~unsaturated])

    # disable saturated neurons
    message.s_weight *= unsaturated
    message.s_bias *= unsaturated

    return message


def collect_linear(linear: nn.Linear, message: Message, constraints: Constraints) -> Message:
    message.sample = linear(message.sample)

    # s_weight = s_weight @ weight.t()
    message.s_weight = torch.nn.functional.linear(message.s_weight, linear.weight, None)

    # s_bias = s_bias @ weight.t() + bias
    message.s_bias = linear(message.s_bias)

    return message


def collect_conv2d(conv2d: nn.Conv2d, message: Message, constraints: Constraints) -> Message:
    message.sample = conv2d(message.sample)
    message.s_weight = conv2d._conv_forward(message.s_weight, conv2d.weight, None)
    message.s_bias = conv2d(message.s_bias)
    return message


def batch_channels_take(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size = data.shape[0]
    channels = data.shape[1]
    batch_indices = indices.flatten(2).expand(batch_size, -1, -1)
    gathered = torch.gather(data.flatten(2), dim=2, index=batch_indices)
    return gathered.reshape(batch_size, channels, *indices.shape[2:])


def collect_max_pool2d(max_pool2d: nn.MaxPool2d, message: Message, constraints: Constraints) -> Message:
    message.sample, indices = torch.nn.functional.max_pool2d(
        message.sample,
        max_pool2d.kernel_size,
        max_pool2d.stride,
        max_pool2d.padding,
        max_pool2d.dilation,
        ceil_mode=max_pool2d.ceil_mode,
        return_indices=True,
    )

    # add constraints
    print('TODO: add constraints')

    # disable saturated neurons (take maxima)
    message.s_weight = batch_channels_take(message.s_weight, indices)
    message.s_bias = batch_channels_take(message.s_bias, indices)

    return message
