import torch
import joblib
import click

from appmax.applications import california_housing, mnist, energy_efficiency, year_prediction
import appmax.evaluation
import appmax.experiment


def metrics_callback(ctx, param, value):
    metrics = appmax.experiment.Metrics(0)
    for m in value:
        metrics |= m
    return metrics


@click.command()
@click.argument('dataset')
@click.argument('run-id', default='run')
@click.option('--train', is_flag=True)
@click.option('-m', '--metrics', type=click.Choice(appmax.experiment.Metrics, case_sensitive=False), multiple=True, default=appmax.experiment.Metrics.MAXIMUM, callback=metrics_callback)
@click.option('-b', '--bits', default=8)
@click.option('-s', '--solver', default=appmax.experiment.SOLVER_DEFAULT)
@click.option('-n', '--samples', default=-1)
def main(dataset, run_id, metrics, train, bits, solver, samples):
    """
    AppMax \n
    input: evaluation network (original net. & approximated net. combined), data samples \n
    output: reported errors (on a single sample or on the polytope; maximum and average)
    """

    torch.manual_seed(42)
    seq_name = 'layers'

    match dataset:
        case 'california':
            MODEL_FILE = "models/california_housing_mlp.pt"
            MODEL_CLASS = california_housing.HousingMLP
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
        case _:
            raise NotImplementedError(f"'{dataset}' dataset is not available")

    model = MODEL_CLASS()

    if train:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
        model.to(device)
        model.fit(data_split.train, data_split.dev, epochs=50)
        model.cpu()
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE).eval()
        # model_approx = MODEL_CLASS()
        # model_approx.load(MODEL_FILE).eval()
        # model_approx.round(bits=bits)

        loader_dev = torch.utils.data.DataLoader(data_split.dev, batch_size=64)
        print('MSE', mse := model.evaluate(loader_dev)[1])
        print('RMSE', rmse := mse ** 0.5)
        print('RMSE scaled', rmse * data_split.metadata.error_scaling)

        x, y = data_split.dev[20]
        print('prediction', pred := model(x).item(), 'true', true := y.item(), 'difference', abs(pred-true))

        return
        eval_net = appmax.evaluation.EvaluationNet(
            model, model_approx, data_split.metadata.bounds, seq_name=seq_name).eval()

        input_sample = data_split.test[0][0]
        result = appmax.experiment.single(eval_net, input_sample, metrics, solver, debug=True)
        errors = [result['error_sample'], result['error_nearby'], result['polytope_width']]
        print('errors', *errors)
        print('scaled', *[x*data_split.metadata.error_scaling if x is not None else '' for x in errors])
        print('california reference\nerrors 0.6520774364471436 0.7007212460728052')
        # print('mnist-conv reference\nerrors 0.5852416753768921 0.6485908165661114')

        # loader_test = torch.utils.data.DataLoader(data_split.test, batch_size=64)
        # max, avg = appmax.evaluation.compute_error_aggregate(model, model_approx, loader_test)
        # print(f"{max=}, {avg=}")

        # with joblib.parallel_config(backend='threading', n_jobs=1):
        #     appmax.experiment.run(f'experiments/{dataset}', run_id, eval_net,
        #                           data_split.test, metrics, first_k=samples)


if __name__ == '__main__':
    main()
