from dataclasses import dataclass
import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
import torchmetrics
import appmax.quantization

Bounds = list[tuple[float, float]]


@dataclass
class DataSplit:
    train: Dataset
    dev: Dataset
    test: Dataset
    bounds: Bounds


class BaseModel(nn.Module):
    """base for a single-stream model"""

    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def callback_stopping(self, epoch: int, metric_dev: float):
        return False

    def save(self, path):
        data = self.state_dict()
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path, weights_only=True, map_location=self.device)
        self.load_state_dict(data)
        return self

    def round(self, **kwargs):
        """modifies the network in-place (converts the network to its approximation)"""
        return appmax.quantization.lower_precision(self, **kwargs)


class TrainableModel(BaseModel):
    def __init__(self, layers: nn.Sequential):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        return self.layers(x)

    def configure(
        self,
        loss_fn,
        optimizer,
        metric_fn: torchmetrics.Metric,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn

    def fit(
        self,
        data_train: Dataset,
        data_dev: Dataset,
        batch_size: int = 64,
        epochs: int = 20,
    ):
        loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
        loader_dev = torch.utils.data.DataLoader(data_dev, batch_size=batch_size)

        for epoch in range(1, epochs+1):
            loss_train, metric_train = self.train_epoch(loader_train)
            loss_dev, metric_dev = self.evaluate(loader_dev)
            print(f"epoch {epoch} done | train: loss {loss_train:.2f}, metric {metric_train:.2f}",
                  f"| dev: loss {loss_dev:.2f}, metric {metric_dev:.2f}")
            is_stopping = self.callback_stopping(epoch, metric_dev)

            if is_stopping:
                print("stopping")
                return

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        self.train()
        return self._execute_epoch(loader, train=True)

    def evaluate(self, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        with torch.no_grad():
            self.eval()
            return self._execute_epoch(loader, train=False)

    def _execute_epoch(self, loader: torch.utils.data.DataLoader, train: bool) -> tuple[float, float]:
        device = self.device
        loss_sum = torch.tensor(0.0, device=device)
        self.metric_fn = self.metric_fn.to(device)
        self.metric_fn.reset()

        for X, y in tqdm.tqdm(loader, leave=False):
            X, y = X.to(device), y.to(device)
            pred = self(X)
            loss = self.loss_fn(pred, y)
            loss_sum += loss
            self.metric_fn.update(pred, y)

            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        loss = loss_sum.item() / len(loader)
        metric = self.metric_fn.compute().item()
        return loss, metric


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
