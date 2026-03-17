import torch
from appmax.applications import mnist
import appmax.evaluation
import appmax.optimize


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
    model = mnist.SmallDenseNetLegacy()
    data_split = mnist.MnistSplit()
    # MODEL_FILE = "models/small_dense.pth"
    MODEL_FILE = "models/mnist_dense_net.pt"

    if False:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        model.to(device)
        model.fit(data_split.train, data_split.dev)
        model.cpu()
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE).eval()
        loader_dev = torch.utils.data.DataLoader(data_split.dev, batch_size=64)
        # print(model.evaluate(loader_dev))

        model_approx = mnist.SmallDenseNetLegacy()
        model_approx.load(MODEL_FILE).eval()
        model_approx.round(bits=8)

        for i in range(0, 1):
            sample, y = data_split.test[i]
            eval_net = appmax.evaluation.EvaluationNet(model, model_approx, 'network').eval()
            err_found = appmax.optimize.find_appmax(eval_net, sample, verbose=False)
            print(i, 'optimized', eval_net(sample).item(), err_found)

        # max_err, avg_err = model.compute_error(model_approx, loader_dev)
        # print('on samples: max', max_err, '/ avg', avg_err)


if __name__ == '__main__':
    main()
