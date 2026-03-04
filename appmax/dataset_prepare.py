import torch
import torchvision
from torch.utils.data import Dataset


class DataSplit:
    def __init__(self, train: Dataset, dev: Dataset, test: Dataset):
        self.train = train
        self.dev = dev
        self.test = test


def get_mnist_split() -> DataSplit:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    params = {"root": "datasets", "download": True, "transform": transform}
    train_dev = torchvision.datasets.MNIST(train=True, **params)
    train, dev = torch.utils.data.random_split(train_dev, [4/5, 1/5])
    test = torchvision.datasets.MNIST(train=False, **params)
    return DataSplit(train, dev, test)
