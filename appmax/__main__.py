import torch
import joblib
import click

from appmax.applications import california_housing, mnist, energy_efficiency, year_prediction
import appmax.evaluation
import appmax.experiment


@click.command()
@click.argument('dataset')
@click.argument('run-id', default='run')
@click.option('--train', is_flag=True)
@click.option('-a', '--approach', type=click.Choice(appmax.experiment.Approach, case_sensitive=False), default=appmax.experiment.Approach.STANDARD)
@click.option('-b', '--bits', default=8)
@click.option('-s', '--solver', default=appmax.experiment.SOLVER_DEFAULT)
@click.option('-n', 'samples', default=-1)
def main(dataset: str, run_id: str, approach: appmax.experiment.Approach, train: bool, bits: int, solver: str, samples: int):
    """
    AppMax \n
    input: evaluation network (original net. & approximated net. combined), data samples \n
    output: reported errors (on a single sample or on the polytope; maximum and average)
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
        case 'year':
            MODEL_FILE = "models/year_prediction_net.pt"
            MODEL_CLASS = year_prediction.YearNet
            data_split = year_prediction.YearPredictionSplit()
            print('year scaling', data_split.scaler.scale_[0])
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

        loader_dev = torch.utils.data.DataLoader(data_split.dev, batch_size=64)
        print('metric', model.evaluate(loader_dev), model_approx.evaluate(loader_dev))
        # x, y = data_split.test[50]
        # print('prediction', pred := model(x).item(), 'true', true := y.item(), 'difference', abs(pred-true) * 10.939755)

        eval_net = appmax.evaluation.EvaluationNet(model, model_approx, data_split.bounds, seq_name=seq_name).eval()

        input_sample = data_split.test[0][0]
        result = appmax.experiment.single(eval_net, input_sample, approach, solver, debug=True)
        print('errors', result['error_sample'], result['error_nearby'], result['polytope_width'])
        print('california reference\nerrors 0.6520774364471436 0.7007212460728052')
        # print('mnist-conv reference\nerrors 0.5852416753768921 0.6485908165661114')

        # loader_test = torch.utils.data.DataLoader(data_split.test, batch_size=64)
        # max, avg = appmax.evaluation.compute_error_aggregate(model, model_approx, loader_test)
        # print(f"{max=}, {avg=}")

        with joblib.parallel_config(backend='threading', n_jobs=1):
            appmax.experiment.run(f'experiments/{dataset}', run_id, eval_net,
                                  data_split.test, approach, first_k=samples)


if __name__ == '__main__':
    main()
