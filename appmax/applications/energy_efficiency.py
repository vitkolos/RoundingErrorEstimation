import pandas as pd
import torch
import torchmetrics
import sklearn.datasets
from appmax.trainable import nn, TrainableModel, DataSplit

DATA_HOME = 'datasets'


class EnergyEfficiencyDataset(torch.utils.data.Dataset):
    """https://archive.ics.uci.edu/dataset/242/energy+efficiency"""

    def __init__(self):
        df = pd.read_excel(f'{DATA_HOME}/ENB2012_data.xlsx')
        self.data = torch.from_numpy(df.loc[:, 'X1':'X8'].to_numpy()).to(dtype=torch.get_default_dtype())
        self.target = torch.from_numpy(df.loc[:, 'Y1':].to_numpy()).to(dtype=torch.get_default_dtype())
        # min, max [(0.62, 0.98), (514.5, 808.5), (245.0, 416.5), (110.25, 220.5), (3.5, 7.0), (2.0, 5.0), (0.0, 0.4), (0.0, 5.0), (6.01, 43.1), (10.9, 48.03)]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class EnergyEfficiencySplit(DataSplit):
    def __init__(self):
        dataset = EnergyEfficiencyDataset()
        self.train, self.dev, self.test = torch.utils.data.random_split(dataset, [6/8, 1/8, 1/8])
        self.bounds = [(0.6, 1.0), (514.5, 808.5), (245.0, 416.5), (110.2, 220.5), (3.5, 7.0),
                       (2.0, 5.0), (0.0, 0.4), (0.0, 5.0)]


class SimpleNet(TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Linear(8, 20),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(20, 2),
            )
        )
        self.configure(
            loss_fn=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam(self.parameters(), lr=0.001),
            metric_fn=torchmetrics.MeanSquaredError(),
        )
