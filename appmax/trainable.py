from dataclasses import dataclass
from typing import Any

import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torchmetrics

import appmax.quantization


class Bounds:
    def __init__(self, bounds: list[tuple[float | None, float | None]], lb: np.ndarray | None = None, ub: np.ndarray | None = None):
        self.seq = bounds

        if lb is not None and ub is not None:
            self.lb, self.ub = lb, ub
        else:
            self.lb = np.array([lb if lb is not None else float('-inf') for lb, _ in bounds])
            self.ub = np.array([ub if ub is not None else float('inf') for _, ub in bounds])


@dataclass
class Metadata:
    bounds: Bounds | None = None
    scaler: Any = None
    error_scaling: float = 1.0
    sl_data: slice = slice(1, None)
    sl_target: slice = slice(0, 1)

    def fit_bounds(self, data_full: np.ndarray, padding: float = 0.05):
        data = data_full[:, self.sl_data]
        mins, maxs = np.min(data, axis=0), np.max(data, axis=0)
        ranges = maxs - mins
        lower_bounds = mins - (ranges * padding)
        upper_bounds = maxs + (ranges * padding)
        bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
        self.bounds = Bounds(bounds, lower_bounds, upper_bounds)

    def remove_outliers(self, data_full: np.ndarray):
        if self.bounds is None:
            raise AttributeError('bounds not fitted')

        data = data_full[:, self.sl_data]
        valid_cells = (data >= self.bounds.lb) & (data <= self.bounds.ub)
        valid_rows = valid_cells.all(axis=1)
        print(f'removed {(~valid_rows).sum()} outliers')
        return data_full[valid_rows]


@dataclass
class DataSplit:
    train: Dataset
    dev: Dataset
    test: Dataset
    metadata: Metadata


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
        scheduler=None,
        epochs=20,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.scheduler = scheduler
        self.epochs = epochs

    def fit(
        self,
        data_train: Dataset,
        data_dev: Dataset,
        batch_size: int = 64,
    ):
        loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
        loader_dev = torch.utils.data.DataLoader(data_dev, batch_size=batch_size)

        for epoch in range(1, self.epochs+1):
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
        loss_sum = 0.0
        total_samples = 0
        self.metric_fn = self.metric_fn.to(device)
        self.metric_fn.reset()

        for X, y in tqdm.tqdm(loader, leave=False):
            X, y = X.to(device), y.to(device)
            N = X.shape[0]
            pred = self(X)
            loss = self.loss_fn(pred, y)
            loss_sum += loss.item()*N
            total_samples += N
            self.metric_fn.update(pred, y)

            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler is not None and self.scheduler.step()

        loss = loss_sum / total_samples
        metric = self.metric_fn.compute().item()
        return loss, metric


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
