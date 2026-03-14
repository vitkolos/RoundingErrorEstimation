from dataclasses import dataclass
import torch
from torch import nn
import appmax.trainable

Constraints = list


@dataclass
class Message:
    sample: torch.Tensor
    s_weights: torch.Tensor
    s_bias: torch.Tensor

    # def apply(self, module: nn.Module, completely: bool = False):
    #     self.sample = module(self.sample)
    #     return self


@torch.no_grad()
def forward(module: nn.Module, message: Message, constraints: Constraints):
    """returns a Message (sample, saturations, shortcut weights)"""
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
    # TODO: add constraints
    message.s_weights *= unsaturated
    message.s_bias *= unsaturated
    return message


def forward_linear(linear: nn.Linear, message: Message, constraints: Constraints):
    message.sample = linear(message.sample)
    message.s_weights = message.s_weights @ linear.weight.t()
    message.s_bias = message.s_bias @ linear.weight.t() + linear.bias
    return message
