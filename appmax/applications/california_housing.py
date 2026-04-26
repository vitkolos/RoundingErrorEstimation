import torch
from torch import nn
import torchmetrics
import sklearn.datasets
import sklearn.preprocessing
import numpy as np
import appmax.trainable


class CaliforniaHousingDataset(torch.utils.data.Dataset):
    """https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset"""

    def __init__(self, metadata: appmax.trainable.Metadata):
        data, target = sklearn.datasets.fetch_california_housing(data_home='datasets', return_X_y=True)
        data = np.column_stack((target, data))

        if metadata.scaler is None:
            metadata.scaler = sklearn.preprocessing.StandardScaler()
            metadata.scaler.fit(data)
            metadata.error_scaling = metadata.scaler.scale_[0]

        data = metadata.scaler.transform(data)

        if metadata.bounds is None:
            metadata.fit_bounds(data)
        else:
            data = metadata.remove_outliers(data)

        self.data = torch.from_numpy(data[:, metadata.sl_data]).to(dtype=torch.get_default_dtype())
        self.target = torch.from_numpy(data[:, metadata.sl_target]).to(dtype=torch.get_default_dtype())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class CaliforniaHousingSplit(appmax.trainable.DataSplit):
    def __init__(self):
        self.metadata = appmax.trainable.Metadata()
        dataset = CaliforniaHousingDataset(self.metadata)
        self.train, self.dev, self.test = torch.utils.data.random_split(dataset, [6/8, 1/8, 1/8])


class SimpleNet(appmax.trainable.TrainableModel):
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


class HousingMLP(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Linear(8, 256),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(128, 64),
                nn.ReLU(),

                nn.Linear(64, 1)
            )
        )
        self.configure(
            loss_fn=nn.HuberLoss(),
            optimizer=torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5),
            metric_fn=torchmetrics.MeanSquaredError(),
        )

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                module.bias.data.fill_(0.01)

        self.apply(init_weights)


class ResBlock(nn.Module):
    # https://arxiv.org/abs/2106.11959
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class HousingResNet(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Linear(8, 128),
                ResBlock(128),
                ResBlock(128),
                ResBlock(128),
                nn.Linear(128, 1)
            )
        )

        self.configure(
            loss_fn=nn.HuberLoss(),
            optimizer=torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4),
            metric_fn=torchmetrics.MeanSquaredError(),
        )

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0.01)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        self.apply(init_weights)
