import torch
import joblib

from appmax.applications import california_housing, mnist
import appmax.evaluation
import appmax.experiment


def main():
    """
    1) prepare a dataset
    2) train (or provide) a network
    3) generate (or provide) an approximated network
    ---
    4) simplify networks (to ReLU & linear layers + normalize)
    5) combine them into an evaluation network
    6) perform (parallel) linear optimimization to find maxima in polytopes
    7) report results

    AppMax
    input: original network, approximated network, data samples
    output: reported errors (single sample × polytope; maximum × average)
    """
    torch.manual_seed(42)
    if True:
        MODEL_FILE = "models/california_housing_simple_net.pt"
        MODEL_CLASS = california_housing.SimpleNet
        data_split = california_housing.CaliforniaHousingSplit()
    else:
        # MODEL_FILE = "models/mnist_small_dense.pt"
        # MODEL_CLASS = mnist.SmallDenseNet
        MODEL_FILE = "models/mnist_dense_net.pt"
        MODEL_CLASS = mnist.SmallDenseNetLegacy
        data_split = mnist.MnistSplit()
    model = MODEL_CLASS()

    if False:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        model.to(device)
        model.fit(data_split.train, data_split.dev)
        model.cpu()
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE).eval()
        model_approx = MODEL_CLASS()
        model_approx.load(MODEL_FILE).eval()
        model_approx.round(bits=8)

        # loader_dev = torch.utils.data.DataLoader(data_split.dev, batch_size=64)
        # print('metric', model.evaluate(loader_dev), model_approx.evaluate(loader_dev))

        eval_net = appmax.evaluation.EvaluationNet(model, model_approx, data_split.bounds, seq_name='layers').eval()

        # input_sample = data_split.test[0][0]
        # result = appmax.experiment.step('', 0, eval_net, input_sample)
        # print('errors', result['error_sample'], result['error_nearby'])

        with joblib.parallel_config(backend='threading', n_jobs=-1):
            appmax.experiment.run('experiments/california', '2', eval_net, data_split.test, first_k=10)


if __name__ == '__main__':
    main()
