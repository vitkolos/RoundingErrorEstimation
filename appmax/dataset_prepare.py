from dataclasses import dataclass
from torch.utils.data import Dataset

@dataclass
class DataSplit:
    train: Dataset
    dev: Dataset
    test: Dataset
