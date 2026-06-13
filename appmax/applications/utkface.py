import os
import glob
import PIL.Image

import torch
from torch import nn
import torchmetrics
import torchvision.transforms
import sklearn.preprocessing
import sklearn.model_selection

import appmax.trainable
import appmax.logger

DATA_HOME = 'datasets'
DATASET_FILE = f'{DATA_HOME}/utkface.pt'
IMG_CHANNELS = 3
IMG_SIZE = 32  # original is 200


def load_utkface() -> tuple[torch.Tensor, torch.Tensor]:
    if os.path.isfile(DATASET_FILE):
        dataset = torch.load(DATASET_FILE)
        N, C, H, W = dataset[0].shape

        if C == IMG_CHANNELS and H == IMG_SIZE and W == IMG_SIZE:
            return dataset
        else:
            print('dataset shape does not match')

    dataset = load_utkface_from_images()
    torch.save(dataset, DATASET_FILE)
    return dataset


def load_utkface_from_images():
    """expects the UTKFace folder containing the images in the 'datasets' folder
    or at least the utkface.pt file
    https://www.kaggle.com/datasets/jangedoo/utkface-new/data"""

    files = glob.glob(f'{DATA_HOME}/UTKFace/*.jpg')
    images, targets = [], []
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),  # scale down to 100×100
        torchvision.transforms.PILToTensor(),  # convert to tensor (keeps its dtype)
    ])

    for file_path in appmax.logger.progress(files, desc='Preparing dataset'):
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        age = float(parts[0])
        targets.append([age])
        img = PIL.Image.open(file_path).convert('RGB')  # ensures image is in RGB format
        images.append(image_transforms(img))

    data = torch.stack(images)
    target = torch.tensor(targets, dtype=torch.get_default_dtype())
    return data, target


class UTKFaceDataset(appmax.trainable.Dataset):
    def __init__(self, data: torch.Tensor, target: torch.Tensor, metadata: appmax.trainable.Metadata):
        self.data = data
        target = target.numpy()

        if metadata.scaler is None:
            metadata.scaler = sklearn.preprocessing.StandardScaler()
            metadata.scaler.fit(target)
            metadata.error_scaling = metadata.scaler.scale_[0]

        target = metadata.scaler.transform(target)
        self.target = torch.from_numpy(target).to(dtype=torch.get_default_dtype())

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = self.data[index].to(dtype=torch.get_default_dtype()) / 255.0  # converts [0, 255] to [0.0, 1.0]
        image = image * 2.0 - 1.0  # convert [0.0, 1.0] to [-1.0, 1.0]
        return image, self.target[index]


def buckets(target: torch.Tensor) -> list[int]:
    return [min(age.int().item() // 10, 8) for age in target]


class UTKFaceSplit(appmax.trainable.DataSplit):
    C = IMG_CHANNELS
    SIZE = IMG_SIZE

    def __init__(self):
        data, target = load_utkface()
        data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(
            data, target, test_size=1/8, random_state=42, stratify=buckets(target))
        data_train, data_dev, target_train, target_dev = sklearn.model_selection.train_test_split(
            data_train, target_train, test_size=1/7, random_state=43, stratify=buckets(target_train))
        bounds = appmax.trainable.Bounds([(-1.0, 1.0)] * (IMG_SIZE*IMG_SIZE*IMG_CHANNELS))
        self.metadata = appmax.trainable.Metadata(bounds=bounds)
        self.train = UTKFaceDataset(data_train, target_train, self.metadata)
        self.dev = UTKFaceDataset(data_dev, target_dev, self.metadata)
        self.test = UTKFaceDataset(data_test, target_test, self.metadata)


class FaceConvNet(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                # (3)×100×100 -> (32)×50×50
                nn.Conv2d(3, 32, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # (32)×50×50 -> (64)×25×25
                nn.Conv2d(32, 64, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # (64)×25×25 -> (128)×12×12
                nn.Conv2d(64, 128, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),
                nn.Linear(128 * 12 * 12, 256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 64),
                nn.ReLU(),

                nn.Linear(64, 1),
            )
        )
        self.configure(
            loss_fn=nn.MSELoss(),
            optimizer=torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4),
            metric_fn=torchmetrics.MeanSquaredError(),
            epochs=70,
        )

        def init_weights(module):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)

        self.apply(init_weights)


class FaceConvNetSmall(appmax.trainable.TrainableModel):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                # (3)×32×32 -> (32)×16×16
                nn.Conv2d(3, 32, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # (32)×16×16 -> (64)×8x8
                nn.Conv2d(32, 64, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # (64)×8x8 -> (128)×4x4
                nn.Conv2d(64, 128, 3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 64),
                nn.ReLU(),

                nn.Linear(64, 1),
            )
        )
        self.configure(
            loss_fn=nn.MSELoss(),
            optimizer=torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4),
            metric_fn=torchmetrics.MeanSquaredError(),
            epochs=70,
        )

        def init_weights(module):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)

        self.apply(init_weights)
