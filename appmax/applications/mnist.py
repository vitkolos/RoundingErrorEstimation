import torch
import torchvision
import torchmetrics
from appmax.trainable import BaseModel, TrainableModel, nn
from appmax.dataset import DataSplit


class SmallDenseNet(TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 10)
            )
        )
        self.configure(
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.parameters(), lr=0.001),
            metric_fn=torchmetrics.Accuracy('multiclass', num_classes=10),
        )

    def callback_stopping(self, epoch, metric_dev):
        return metric_dev > 0.9


class SmallDenseNetLegacy(BaseModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 2000),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        return self.network(x)


class MnistSplit(DataSplit):
    def __init__(self,):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        params = {"root": "datasets", "download": True, "transform": transform}
        train_dev = torchvision.datasets.MNIST(train=True, **params)
        self.train, self.dev = torch.utils.data.random_split(train_dev, [4/5, 1/5])
        self.test = torchvision.datasets.MNIST(train=False, **params)
