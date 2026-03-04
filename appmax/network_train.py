import tqdm
import torch
from torch import nn
import torchmetrics
import dataset_prepare


class TrainableModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def callback_stopping(self, epoch: int, metric_dev: float):
        return False

    def configure(
        self,
        loss_fn,
        optimizer,
        batch_size: int = 64,
        epochs: int = 20,
        metric_fn: torchmetrics.Metric = None,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.metric_fn = metric_fn

    def fit(
        self,
        data_train: dataset_prepare.Dataset,
        data_dev: dataset_prepare.Dataset | None = None
    ):
        loader_train = torch.utils.data.DataLoader(
            data_train, batch_size=self.batch_size)
        loader_dev = torch.utils.data.DataLoader(
            data_dev, batch_size=self.batch_size)

        for epoch in range(self.epochs):
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
            X, y = X.to(self.device), y.to(self.device)
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
                X, y = X.to(self.device), y.to(self.device)
                pred = self(X)
                loss_sum += self.loss_fn(pred, y).item()
                self.metric_fn.update(pred, y)

        loss = loss_sum / len(loader)
        metric = self.metric_fn.compute().item()
        return loss, metric


class SmallDenseNet(TrainableModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
        self.configure(
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.parameters(), lr=0.001),
            metric_fn=torchmetrics.Accuracy('multiclass', num_classes=10),
            epochs=10,
            batch_size=64,
        )

    def forward(self, x):
        return self.network(x)
    
    def callback_stopping(self, epoch, metric_dev):
        return metric_dev > 0.9
