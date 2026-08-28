import torch

from appmax.applications import california_housing, year_prediction, utkface, mnist
import appmax.trainable


class DataBundle:
    def __init__(self, dataset: str):
        self.model_class: type[appmax.trainable.BaseModel]
        self.data_split: appmax.trainable.DataSplit

        match dataset.lower():
            case 'california':
                self.model_file = "models/california_housing_mlp.pt"
                self.model_class = california_housing.HousingMLP
                self.data_split = california_housing.CaliforniaHousingSplit()
            case 'year':
                self.model_file = "models/year_prediction_net.pt"
                self.model_class = year_prediction.YearNet
                self.data_split = year_prediction.YearPredictionSplit()
            case 'utkface':
                self.model_file = "models/utkface_smaller.pt"
                self.model_class = utkface.FaceConvNetSmaller
                self.data_split = utkface.UTKFaceSplit()
            case 'mnist':
                self.model_file = "models/mnist_dense_net.pt"
                self.model_class = mnist.SmallDenseNetLegacy
                self.data_split = mnist.MnistSplit()
            case _:
                raise NotImplementedError(f"'{dataset}' dataset is not available")

    def load_model(self) -> torch.nn.Module:
        model = self.model_class()
        model.load(self.model_file).eval()
        return model
