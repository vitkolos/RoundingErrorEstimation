import math
import pandas as pd
import torch
import torchmetrics
import sklearn.base
import sklearn.preprocessing
from appmax.trainable import nn, TrainableModel, DataSplit

DATA_HOME = 'datasets'


class YearPredictionDataset(torch.utils.data.Dataset):
    """https://web.archive.org/web/20260405131946/https://archive.ics.uci.edu/dataset/203/yearpredictionmsd"""

    def __init__(self, scaler: sklearn.base.TransformerMixin, bounds: list, train: bool):
        # train: first 463,715 examples
        # test: last 51,630 examples
        T_LEN = 463_715
        rows = {('nrows' if train else 'skiprows'): T_LEN}
        data = pd.read_csv(f'{DATA_HOME}/YearPredictionMSD.txt', header=None, **rows).to_numpy()  # type: ignore

        if train:
            scaler.fit(data)

        data = scaler.transform(data)
        self.data = torch.from_numpy(data[:, 1:]).to(dtype=torch.get_default_dtype())
        self.target = torch.from_numpy(data[:, 0:1]).to(dtype=torch.get_default_dtype())

        # 90 attributes, 12 = timbre average, 78 = timbre covariance
        # The first value is the year (target), ranging from 1922 to 2011.
        # Features extracted from the 'timbre' features from The Echo Nest API.

        if train:
            for lb, ub in zip(self.data.min(dim=0).values.tolist(), self.data.max(dim=0).values.tolist()):
                bounds.append((math.floor(lb), math.ceil(ub)))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class YearPredictionSplit(DataSplit):
    def __init__(self):
        self.bounds = []
        self.scaler = sklearn.preprocessing.StandardScaler()
        train_dev = YearPredictionDataset(self.scaler, self.bounds, train=True)
        self.test = YearPredictionDataset(self.scaler, self.bounds, train=False)
        self.train, self.dev = torch.utils.data.random_split(train_dev, [4/5, 1/5])


class YearNet(TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Linear(90, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
            )
        )
        self.configure(
            loss_fn=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(self.parameters(), lr=0.001),
            metric_fn=torchmetrics.MeanSquaredError(),
        )
