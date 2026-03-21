import torch
import torchmetrics
import sklearn.datasets
from appmax.trainable import nn, TrainableModel, DataSplit


class SimpleNet(TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Linear(8, 100),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(100, 1),
            )
        )
        self.configure(
            loss_fn=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(self.parameters(), lr=0.001),
            metric_fn=torchmetrics.MeanSquaredError(),
        )


class CaliforniaHousingDataset(torch.utils.data.Dataset):
    def __init__(self):
        data, target = sklearn.datasets.fetch_california_housing(data_home='datasets', return_X_y=True)
        self.data = torch.from_numpy(data).to(dtype=torch.get_default_dtype())
        self.target = torch.from_numpy(target).to(dtype=torch.get_default_dtype())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class CaliforniaHousingSplit(DataSplit):
    def __init__(self):
        dataset = CaliforniaHousingDataset()
        self.train, self.dev, self.test = torch.utils.data.random_split(dataset, [6/8, 1/8, 1/8])
        self.bounds = [(0.4, 15.0), (1.0, 52.0), (0.8, 142.0), (0.3, 34.1),
                       (3.0, 35682.0), (0.6, 1243.4), (32.5, 42.0), (-124.4, -114.3)]
