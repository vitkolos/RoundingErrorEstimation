import tqdm
import torch
from torch import nn
import torchmetrics
import dataset_prepare
import quant_utils


class TrainableModel(nn.Module):
    def __init__(self, network: nn.Sequential):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def callback_stopping(self, epoch: int, metric_dev: float):
        return False

    def configure(
        self,
        loss_fn,
        optimizer,
        metric_fn: torchmetrics.Metric = None,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn

    def fit(
        self,
        data_train: dataset_prepare.Dataset,
        data_dev: dataset_prepare.Dataset | None = None,
        batch_size: int = 64,
        epochs: int = 20,
    ):
        loader_train = torch.utils.data.DataLoader(
            data_train, batch_size=batch_size)
        loader_dev = torch.utils.data.DataLoader(
            data_dev, batch_size=batch_size)

        for epoch in range(1, epochs+1):
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
        device = self.device
        self.train()

        for X, y in tqdm.tqdm(loader_train):
            X, y = X.to(device), y.to(device)
            pred = self(X)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad
    def evaluate(self, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        device = self.device
        self.eval()
        loss_sum = 0.0
        self.metric_fn = self.metric_fn.to(device)
        self.metric_fn.reset()

        for X, y in loader:
            X, y = X.to(device), y.to(device)
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
        data = torch.load(path, weights_only=True, map_location=self.device)
        self.load_state_dict(data)
        return self

    def round(self, bits=16):
        """modifies the network in-place (converts the network to its approximation)"""
        quant_utils.lower_precision(self, bits)
        return self

    @torch.no_grad
    def compute_error(self, other: nn.Module, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        device = self.device
        other = other.to(device)
        self.eval()
        other.eval()
        max_err, total_err = 0.0, 0.0
        num_samples = 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred1 = self(X)
            pred2 = other(X)
            num_samples += y.shape[0]
            l1_norms = (pred1 - pred2).abs().sum(dim=1)
            max_err = max(max_err, l1_norms.max().item())
            total_err += l1_norms.sum().item()

        avg_err = total_err / num_samples
        return max_err, avg_err

    @torch.no_grad
    def create_evaluation_network(self, other: 'TrainableModel'):
        layers_self = list(self.network)
        layers_other = list(other.network)
        seq_self = nn.Sequential(*layers_self[:-1])
        seq_other = nn.Sequential(*layers_other[:-1])
        last_self = layers_self[-1]
        last_other = layers_other[-1]

        if not (isinstance(last_self, nn.Linear) and isinstance(last_other, nn.Linear)):
            raise Exception('last layers are not linear')

        in_features = last_self.in_features + last_other.in_features
        out_features = last_self.out_features + last_other.out_features
        seq_both = nn.Sequential(
            magic_layer := nn.Linear(in_features, out_features),
            nn.ReLU(),
            output_layer := nn.Linear(out_features, 1, bias=False)
        )

        W1, b1 = last_self.weight, last_self.bias
        W2, b2 = last_other.weight, last_other.bias
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

        class EvaluationNetwork(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.cat((seq_self(x), seq_other(x)), dim=1)
                return seq_both(x)

        return EvaluationNetwork()
