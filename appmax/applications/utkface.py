import os
import glob
import PIL.Image

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchmetrics
import torchvision.transforms
import sklearn.preprocessing
import sklearn.model_selection

import appmax.trainable

DATA_HOME = 'datasets'


def load_utkface():
    """https://www.kaggle.com/datasets/jangedoo/utkface-new/data"""

    files = glob.glob(f'{DATA_HOME}/UTKFace/*.jpg')
    images, targets = [], []
    to_tensor = torchvision.transforms.ToTensor()  # scales pixel values to [0.0, 1.0]

    for file_path in files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        age = float(parts[0])
        targets.append([age])
        img = PIL.Image.open(file_path).convert('RGB')  # ensures image is RGB
        images.append(to_tensor(img))

    data = torch.stack(images).to(dtype=torch.get_default_dtype())
    target = torch.tensor(targets, dtype=torch.get_default_dtype())
    return data, target


class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, metadata: appmax.trainable.Metadata):
        if metadata.scaler is None:
            ...
            # normalize ages?

        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class UTKFaceSplit(appmax.trainable.DataSplit):
    def __init__(self):
        data, target = load_utkface()
        breakpoint()
        data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(
            data, target, test_size=1/8, random_state=42, stratify=target)

        bounds = appmax.trainable.Bounds([(0.0, 1.0)] * (200*200*3))
        self.metadata = appmax.trainable.Metadata(bounds=bounds)
        train_dev = UTKFaceDataset(data_train, target_train, self.metadata)
        self.test = UTKFaceDataset(data_test, target_test, self.metadata)
        self.train, self.dev = torch.utils.data.random_split(train_dev, [6/7, 1/7])
