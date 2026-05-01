import pandas as pd
import torch
from torch import nn
import torchmetrics
import numpy as np
import sklearn.preprocessing
import appmax.trainable

DATA_HOME = 'datasets'



class YearPredictionDataset(torch.utils.data.Dataset):
    """https://web.archive.org/web/20260405131946/https://archive.ics.uci.edu/dataset/203/yearpredictionmsd"""
    # train: first 463,715 examples
    # test: last 51,630 examples
    TD_LEN = 463_715
    T_LEN = TD_LEN - 10_000

    def __init__(self, metadata: appmax.trainable.Metadata, train: bool):
        rows = {('nrows' if train else 'skiprows'): self.TD_LEN}
        data = pd.read_csv(f'{DATA_HOME}/YearPredictionMSD.txt', header=None, **rows).to_numpy()  # type: ignore

        if metadata.scaler is None:
            metadata.scaler = sklearn.preprocessing.StandardScaler()
            metadata.scaler.fit(data)
            metadata.error_scaling = metadata.scaler.scale_[0]

        data = metadata.scaler.transform(data)

        if metadata.bounds is None:
            metadata.fit_bounds(data)
        else:
            data = metadata.remove_outliers(data)

        self.data = torch.from_numpy(data[:, 1:]).to(dtype=torch.get_default_dtype())
        self.target = torch.from_numpy(data[:, 0:1]).to(dtype=torch.get_default_dtype())

        # 90 attributes, 12 = timbre average, 78 = timbre covariance
        # The first value is the year (target), ranging from 1922 to 2011.
        # Features extracted from the 'timbre' features from The Echo Nest API.

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class YearPredictionSplit(appmax.trainable.DataSplit):
    def __init__(self):
        self.metadata = appmax.trainable.Metadata()
        train_dev = YearPredictionDataset(self.metadata, train=True)
        self.test = YearPredictionDataset(self.metadata, train=False)
        self.train = torch.utils.data.Subset(train_dev, range(0, YearPredictionDataset.T_LEN))
        self.dev = torch.utils.data.Subset(train_dev, range(YearPredictionDataset.T_LEN, len(train_dev)))


class YearNet(appmax.trainable.TrainableModel):
    def __init__(self):
        dropout = 0.2
        super().__init__(
            nn.Sequential(
                nn.Linear(90, dim := 512),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim := 256),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim := 128),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, 1),
            )
        )
        self.configure(
            loss_fn=torch.nn.MSELoss(),
            optimizer=torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-5),
            metric_fn=torchmetrics.MeanSquaredError(),
            epochs=50,
            batch_size=512,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs * (YearPredictionDataset.T_LEN // self.batch_size))

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                module.bias.data.fill_(0.01)

        self.apply(init_weights)
