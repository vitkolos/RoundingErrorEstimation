import numpy as np
import torch


def lower_precision(net, bits=16):
    """main interface of the module; modifies the network in-place"""
    if bits == 16:
        return net.half().to(dtype=torch.get_default_dtype())
    else:
        quant = Quantization(net, bits)
        return quant.convert(net)


class Quantization():
    def __init__(self, net, bits=8):
        parameters = torch.hstack([p.flatten() for p in net.parameters()])
        self.min = parameters.min().item()
        self.max = parameters.max().item()

        self.a = - 2**(bits-1)
        self.b = 2**(bits-1) - 1

        self.s = (self.max - self.min) / (self.b - self.a)
        self.z = int((self.max * self.a - self.min * self.b) / (self.max - self.min))

    def quant_round(self, number):
        # convert down to "bits" bits (such as 8 bits)
        q_number = np.round(1 / self.s * number + self.z, decimals=0)
        q_number = np.clip(q_number, a_min=self.a, a_max=self.b)

        # convert back to float64
        q_number = q_number.astype(np.int64)
        new_number = self.s * (q_number - self.z)
        new_number = new_number.astype(np.float64)

        return new_number

    def convert(self, network):
        for p in network.parameters():
            p_value = p.cpu().detach().numpy()
            new_p_value = self.quant_round(p_value)
            new_p = torch.Tensor(new_p_value)
            # p.copy_ requires grad, p.data is not very nice way :(
            # p.copy_(new_p)
            p.data = new_p

        return network
