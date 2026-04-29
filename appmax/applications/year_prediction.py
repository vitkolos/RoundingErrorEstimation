import pandas as pd
import torch
from torch import nn
import torchmetrics
import sklearn.preprocessing
import appmax.trainable

DATA_HOME = 'datasets'


class YearPredictionDataset(torch.utils.data.Dataset):
    """https://web.archive.org/web/20260405131946/https://archive.ics.uci.edu/dataset/203/yearpredictionmsd"""

    def __init__(self, metadata: appmax.trainable.Metadata, train: bool):
        # train: first 463,715 examples
        # test: last 51,630 examples
        T_LEN = 463_715
        rows = {('nrows' if train else 'skiprows'): T_LEN}
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
        DEV_START = len(train_dev)-10_000
        self.train = torch.utils.data.Subset(train_dev, range(0, DEV_START))
        self.dev = torch.utils.data.Subset(train_dev, range(DEV_START, len(train_dev)))


class YearNet(appmax.trainable.TrainableModel):
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
            epochs=50,
        )
