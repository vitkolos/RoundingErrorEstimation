from dataclasses import dataclass, field
import torch
from torch import nn
import appmax.trainable


@dataclass
class Message:
    sample: torch.Tensor
    s_weight: torch.Tensor
    s_bias: torch.Tensor

    # def apply(self, module: nn.Module, completely: bool = False):
    #     self.sample = module(self.sample)
    #     return self


@dataclass
class Constraints:
    U_weight: list[torch.Tensor] = field(default_factory=list)
    U_bias: list[torch.Tensor] = field(default_factory=list)
    S_weight: list[torch.Tensor] = field(default_factory=list)
    S_bias: list[torch.Tensor] = field(default_factory=list)


@torch.no_grad()
def forward(module: nn.Module, message: Message, constraints: Constraints) -> Message:
    print(message)
    match module:
        case appmax.trainable.TrainableModel():
            return forward(module.layers, message, constraints)
        case nn.Sequential():
            for submodule in module:
                message = forward(submodule, message, constraints)
            return message
        case nn.Dropout():
            return message
        case nn.Linear():
            return forward_linear(module, message, constraints)
        case nn.Flatten():
            return message
        case nn.ReLU():
            return forward_relu(module, message, constraints)
        case _:
            raise NotImplementedError(
                f'{type(module)} forward not implemented')


def forward_relu(relu: nn.ReLU, message: Message, constraints: Constraints):
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


def forward_linear(linear: nn.Linear, message: Message, constraints: Constraints):
    message.sample = linear(message.sample)
    message.s_weight = message.s_weight @ linear.weight.t()
    message.s_bias = message.s_bias @ linear.weight.t() + linear.bias
    return message
