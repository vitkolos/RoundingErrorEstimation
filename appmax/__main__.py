import torch

from appmax.applications import mnist
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
    model = mnist.SmallDenseNet()
    data_split = mnist.MnistSplit()
    MODEL_FILE = "models/small_dense.pth"
    # MODEL_FILE = "models/mnist_dense_net.pt"

    if False:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        model.to(device)
        model.fit(data_split.train, data_split.dev)
        model.cpu()
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE).eval()
        model_approx = mnist.SmallDenseNet()
        model_approx.load(MODEL_FILE).eval()
        model_approx.round(bits=8)
        eval_net = appmax.evaluation.EvaluationNet(model, model_approx).eval()

        # TODO: add joblib wrapper (set n_jobs=-1 & threading backend)
        appmax.experiment.run('experiments/mnist', '1', eval_net, data_split.test, 10)


if __name__ == '__main__':
    main()
