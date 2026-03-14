import torch
from appmax.quantization import Quantization

tt = torch.tensor

def test_quantization():
    q = Quantization(-1.5, 1.5, bits=2)
    assert torch.equal(q.quant_round(tt(0)), tt(0))
    assert torch.equal(q.quant_round(tt(0.803)), q.quant_round(tt(0.804)))
    assert torch.equal(q.quant_round(tt(5)), q.quant_round(tt(6)))
    range_ = torch.arange(-1.5, 1.6, 0.1)
    assert len(set(q.quant_round(range_).tolist())) == 4
