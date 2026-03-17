import torch
import joblib
import pandas as pd
import tqdm

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

        for i in range(0, 1):
            sample = data_split.test[i][0]
            result = appmax.experiment.run('mnist_0', eval_net, sample)
            print(result['error_nearby'])

        total = 50
        # memory = joblib.Memory('mem')
        mem_run = appmax.experiment.run  # memory.cache(appmax.experiment.run)
        p = joblib.Parallel(n_jobs=-1, return_as='generator_unordered')
        # TODO: verify best practices -- how to share the eval_net and the dataset between the processes?
        results_gen = p(joblib.delayed(mem_run)('mnist_0', eval_net, data_split.test[i][0]) for i in range(total))
        df = pd.DataFrame(tqdm.tqdm(results_gen, leave=False, total=total))
        print(df.describe())


if __name__ == '__main__':
    main()
