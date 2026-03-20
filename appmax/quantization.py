import torch

def lower_precision(net: torch.nn.Module, bits=16) -> torch.nn.Module:
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

    def __init__(self, min: float, max: float, bits: int = 8):
        # two's complement range for "bits" bits (but could be also from 0 to 2**bits-1)
        self.a = - 2**(bits-1)
        self.b = 2**(bits-1) - 1

        # converts the range of bits to the original range
        self.scale = (max - min) / (self.b - self.a)
        # finds an integer where to map zero
        self.zero_point = int((max * self.a - min * self.b) / (max - min))

    def quant_round(self, number: torch.Tensor) -> torch.Tensor:
        # convert (round) to "bits" bits (such as 8 bits)
        q_number = torch.round(number / self.scale + self.zero_point)
        q_number = torch.clip(q_number, self.a, self.b)

        # convert back to float64
        return self.scale * (q_number - self.zero_point)

    @torch.no_grad()
    def convert(self, network: torch.nn.Module) -> torch.nn.Module:
        for p in network.parameters():
            p.copy_(self.quant_round(p))

        return network
