import numpy as np
import torch


def lower_precision(net: torch.nn.Module, bits=16):
    """main interface of the module; modifies the network in-place"""
    if bits == 16:
        return net.half().to(dtype=torch.get_default_dtype())
    else:
        parameters = torch.hstack([p.flatten() for p in net.parameters()])
        min = parameters.min().item()
        max = parameters.max().item()
        quant = Quantization(min, max, bits)
        return quant.convert(net)


class Quantization():
    """uniform affine (asymmetric) quantization: maps the values of parameters onto bits and back"""

    def __init__(self, min, max, bits=8):
        # two's complement range for "the given number of" bits
        self.a = - 2**(bits-1)
        self.b = 2**(bits-1) - 1

        # converts the range of bits to the original range
        self.scale = (max - min) / (self.b - self.a)
        # finds an integer where to map zero
        self.zero_point = int((max * self.a - min * self.b) / (max - min))

    def quant_round(self, number):
        # convert (round) to "bits" bits (such as 8 bits)
        q_number = np.round(number / self.scale + self.zero_point)
        q_number = np.clip(q_number, a_min=self.a, a_max=self.b)

        # convert back to float64
        q_number = q_number.astype(np.int64)
        return self.scale * (q_number - self.zero_point)

    def convert(self, network: torch.nn.Module):
        for p in network.parameters():
            p_value = p.cpu().detach().numpy()
            new_p_value = self.quant_round(p_value)
            new_p = torch.Tensor(new_p_value)
            # p.copy_ requires grad, p.data is not very nice way :(
            p.data = new_p

        return network
