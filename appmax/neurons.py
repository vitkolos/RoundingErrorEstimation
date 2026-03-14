from dataclasses import dataclass, field
import copy
import torch
from torch import nn
import appmax.trainable


@dataclass
class Message:
    sample: torch.Tensor
    s_weight: torch.Tensor
    s_bias: torch.Tensor

    def __init__(self, sample: torch.Tensor):
        self.sample = sample if sample.shape[0] == 1 else sample.unsqueeze(0)
        # create a unit matrix with a shape corresponding to the input
        self.s_weight = torch.eye(self.sample.numel()).reshape((-1, *self.sample.shape[1:]))
        self.s_bias = torch.zeros_like(self.sample)

    def apply(self, module: nn.Module):
        self.sample = module(self.sample)
        self.s_weight = module(self.s_weight)
        self.s_bias = module(self.s_bias)
        return self

    def cat(self, other: 'Message'):
        self.sample = torch.cat((self.sample, other.sample), dim=1)
        self.s_weight = torch.cat((self.s_weight, other.s_weight), dim=1)
        self.s_bias = torch.cat((self.s_bias, other.s_bias), dim=1)
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
        case appmax.trainable.TrainableModel():
            return collect(module.layers, message, constraints)
        case appmax.trainable.DualStreamModel():
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
        case nn.Dropout():
            return message
        case nn.Flatten():
            return message.apply(module)
        case _:
            raise NotImplementedError(f'{type(module)} neurons.collect not implemented')


def collect_relu(relu: nn.ReLU, message: Message, constraints: Constraints):
    message.sample = relu(message.sample)
    unsaturated = message.sample > 0
    unsaturated_sq = unsaturated.squeeze()

    # add constraints
    constraints.U_weight.append(message.s_weight.t()[unsaturated_sq])
    constraints.U_bias.append(message.s_bias[unsaturated])
    constraints.S_weight.append(message.s_weight.t()[~unsaturated_sq])
    constraints.S_bias.append(message.s_bias[~unsaturated])

    # disable saturated neurons
    message.s_weight *= unsaturated
    message.s_bias *= unsaturated

    return message


def collect_linear(linear: nn.Linear, message: Message, constraints: Constraints):
    message.sample = linear(message.sample)

    # s_weight = s_weight @ weight.t()
    message.s_weight = torch.nn.functional.linear(message.s_weight, linear.weight, None)

    # s_bias = s_bias @ weight.t() + bias
    message.s_bias = linear(message.s_bias)
    
    return message
