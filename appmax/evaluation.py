import torch
from torch import nn
import torchmetrics
from appmax.trainable import Bounds


@torch.no_grad()
def compute_error_tensor(first: nn.Module, second: nn.Module, xs: torch.Tensor) -> torch.Tensor:
    pred1 = first(xs)
    pred2 = second(xs)
    l1_norms = (pred1 - pred2).abs().sum(dim=1)
    return l1_norms


@torch.no_grad()
def compute_error_aggregate(first: nn.Module, second: nn.Module, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
    device = next(first.parameters()).device
    second = second.to(device)
    first.eval()
    second.eval()
    max_err = torchmetrics.MaxMetric()
    avg_err = torchmetrics.MeanMetric()

    for xs, _ in loader:
        xs = xs.to(device)
        error_tensor = compute_error_tensor(first, second, xs)
        max_err.update(error_tensor)
        avg_err.update(error_tensor)

    return max_err.compute().item(), avg_err.compute().item()


class DualStreamModel(nn.Module):
    def __init__(self, first_stream: nn.Module, second_stream: nn.Module, common_stream: nn.Module):
        super().__init__()
        self.first_stream = first_stream
        self.second_stream = second_stream
        self.common_stream = common_stream

    def forward(self, x):
        x = torch.cat((self.first_stream(x), self.second_stream(x)), dim=-1)
        return self.common_stream(x)


class EvaluationNet(DualStreamModel):
    @torch.no_grad()
    def __init__(self, first: nn.Module, second: nn.Module, bounds: Bounds, seq_name: str = 'layers'):
        """bounds are then used by the LP solver;
        seq_name is the name of the attribute where the nn.Sequential module is stored in both models"""

        self.bounds = bounds

        # getattr(first, seq_name) is usually equivalent to first.layers
        layers_first = list(getattr(first, seq_name))
        layers_second = list(getattr(second, seq_name))
        seq_first = nn.Sequential(*layers_first[:-1])
        seq_second = nn.Sequential(*layers_second[:-1])
        last_first = layers_first[-1]
        last_second = layers_second[-1]

        if not (isinstance(last_first, nn.Linear) and isinstance(last_second, nn.Linear)):
            raise ValueError('last layers are not linear')
        elif last_first.out_features != last_second.out_features:
            raise ValueError('last layers have different numbers of neurons')

        in_features = last_first.in_features + last_second.in_features
        out_features = last_first.out_features + last_second.out_features
        seq_both = nn.Sequential(
            magic_layer := nn.Linear(in_features, out_features),
            nn.ReLU(),
            output_layer := nn.Linear(out_features, 1, bias=False)
        )

        W1, b1 = last_first.weight, last_first.bias
        W2, b2 = last_second.weight, last_second.bias
        # magic W  =  W1 -W2
        #            -W1  W2
        magic_W = torch.vstack((
            torch.hstack((W1, -W2)),
            torch.hstack((-W1, W2))
        ))
        magic_layer.weight.copy_(magic_W)
        # magic_b1 = b1 - b2
        # mabic_b2 = b2 - b1
        magic_layer.bias.copy_(torch.cat((b1-b2, b2-b1)))

        nn.init.ones_(output_layer.weight)
        super().__init__(seq_first, seq_second, seq_both)
