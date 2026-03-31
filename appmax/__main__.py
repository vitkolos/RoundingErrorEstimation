import torch
import joblib
import click

from appmax.applications import california_housing, mnist, energy_efficiency
import appmax.evaluation
import appmax.experiment


@click.command()
@click.argument('dataset')
@click.argument('run-id', default='run')
@click.option('--train', is_flag=True)
@click.option('-b', '--bits', default=8)
@click.option('-n', 'samples', default=-1)
def main(dataset, run_id, train, bits, samples):
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
    seq_name = 'layers'

    match dataset:
        case 'california':
            MODEL_FILE = "models/california_housing_simple_net.pt"
            MODEL_CLASS = california_housing.SimpleNet
            data_split = california_housing.CaliforniaHousingSplit()
        case 'energy':
            MODEL_FILE = "models/energy_efficiency_simple_net.pt"
            MODEL_CLASS = energy_efficiency.SimpleNet
            data_split = energy_efficiency.EnergyEfficiencySplit()
        case 'mnist':
            # MODEL_FILE = "models/mnist_new_dense.pt"
            # MODEL_CLASS = mnist.SmallDenseNet
            MODEL_FILE = "models/mnist_dense_net.pt"
            MODEL_CLASS = mnist.SmallDenseNetLegacy
            data_split = mnist.MnistSplit()
            seq_name = 'network'
        case 'mnist-conv':
            MODEL_FILE = "models/mnist_conv_net.pt"
            MODEL_CLASS = mnist.SmallConvNetLegacy
            data_split = mnist.MnistSplit()
            seq_name = 'network'
        case _:
            raise NotImplementedError(f"'{dataset}' dataset is not available")

    model = MODEL_CLASS()

    if train:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
        model.to(device)
        model.fit(data_split.train, data_split.dev)
        model.cpu()
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE).eval()
        model_approx = MODEL_CLASS()
        model_approx.load(MODEL_FILE).eval()
        model_approx.round(bits=bits)

        # loader_dev = torch.utils.data.DataLoader(data_split.dev, batch_size=64)
        # print('metric', model.evaluate(loader_dev), model_approx.evaluate(loader_dev))

        eval_net = appmax.evaluation.EvaluationNet(model, model_approx, data_split.bounds, seq_name=seq_name).eval()

        input_sample = data_split.test[0][0]
        result = appmax.experiment.step('', 0, eval_net, input_sample, debug=True)
        print('errors', result['error_sample'], result['error_nearby'])

        # loader_test = torch.utils.data.DataLoader(data_split.test, batch_size=64)
        # max, avg = appmax.evaluation.compute_error_aggregate(model, model_approx, loader_test)
        # print(f"{max=}, {avg=}")

        # with joblib.parallel_config(backend='threading', n_jobs=1):
        #     appmax.experiment.run(f'experiments/{dataset}', run_id, eval_net, data_split.test, first_k=samples)


if __name__ == '__main__':
    main()
