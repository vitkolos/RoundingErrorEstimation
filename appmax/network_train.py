from dataclasses import dataclass

import tqdm
import torch
from torch import nn
import torchmetrics
import dataset_prepare

@dataclass
class TrainingArgs:
    device: torch.device
    batch_size: int = 64
    epochs: int = 20


class TrainableModel(nn.Module):
    def __init__(self):
        super().__init__()

    def callback_stopping(self, epoch: int, metric_dev: float):
        return False

    def configure(
        self,
        args: TrainingArgs,
        loss_fn,
        optimizer,
        metric_fn: torchmetrics.Metric = None,
    ):
        self.args = args
        self.to(args.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn.to(args.device)

    def fit(
        self,
        data_train: dataset_prepare.Dataset,
        data_dev: dataset_prepare.Dataset | None = None
    ):
        loader_train = torch.utils.data.DataLoader(
            data_train, batch_size=self.args.batch_size)
        loader_dev = torch.utils.data.DataLoader(
            data_dev, batch_size=self.args.batch_size)

        for epoch in range(self.args.epochs):
            self.train_step(loader_train)
            print(f"epoch {epoch} done", end=" ", flush=True)
            loss_train, metric_train = self.evaluate(loader_train)
            print(
                f"| train: loss {loss_train:.2f}, metric {metric_train:.2f}", end=" ", flush=True)
            loss_dev, metric_dev = self.evaluate(loader_dev)
            print(f"| dev: loss {loss_dev:.2f}, metric {metric_dev:.2f}")

            is_stopping = self.callback_stopping(epoch, metric_dev)

            if is_stopping:
                print("stopping")
                return

    def train_step(self, loader_train: torch.utils.data.DataLoader):
        self.train()

        for X, y in tqdm.tqdm(loader_train):
            X, y = X.to(self.args.device), y.to(self.args.device)
            pred = self(X)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def evaluate(self, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        with torch.no_grad():
            self.eval()
            loss_sum = 0.0
            self.metric_fn.reset()

            for X, y in loader:
                X, y = X.to(self.args.device), y.to(self.args.device)
                pred = self(X)
                loss_sum += self.loss_fn(pred, y).item()
                self.metric_fn.update(pred, y)

        loss = loss_sum / len(loader)
        metric = self.metric_fn.compute().item()
        return loss, metric
    
    def save(self, path):
        data = self.state_dict()
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path, weights_only=True)
        self.load_state_dict(data)
