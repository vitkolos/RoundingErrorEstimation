import torch
from appmax.applications import mnist
import appmax.neurons


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
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = mnist.SmallDenseNet().to(device)
    data_split = mnist.MnistSplit()
    MODEL_FILE = "models/small_dense.pth"

    if False:
        model.fit(data_split.train, data_split.dev)
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE)
        loader_dev = torch.utils.data.DataLoader(data_split.dev, batch_size=64)
        # print(model.evaluate(loader_dev))

        model_approx = mnist.SmallDenseNet().to(device)
        model_approx.load(MODEL_FILE)
        model_approx.round(bits=8)

        # max_err, avg_err = model.compute_error(model_approx, loader_dev)
        # print(max_err, avg_err)

        X, y = data_split.dev[0]
        # eval_net = model.create_evaluation_network(model_approx)
        # print((pred1 - pred2).abs().sum(), eval_net(X))
        model.eval()
        constraints = appmax.neurons.Constraints()
        message = appmax.neurons.Message(X)
        message = appmax.neurons.forward(model, message, constraints)
        output = X.flatten() @ message.s_weight + message.s_bias


if __name__ == '__main__':
    main()
