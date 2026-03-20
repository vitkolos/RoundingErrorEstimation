import torch
import torchvision
import torchmetrics
from appmax.trainable import nn, BaseModel, TrainableModel, DataSplit


class CaliforniaHousingSplit(DataSplit):
    def __init__(self):
        # transform = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize((0.1307,), (0.3081,))
        # ])
        # params = {"root": "datasets", "download": True, "transform": transform}
        # train_dev = torchvision.datasets.MNIST(train=True, **params)
        # self.train, self.dev = torch.utils.data.random_split(train_dev, [4/5, 1/5])
        # self.test = torchvision.datasets.MNIST(train=False, **params)
        ...
