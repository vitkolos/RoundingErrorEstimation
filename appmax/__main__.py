import torch
import joblib
import click

from appmax.applications import california_housing, mnist, energy_efficiency, year_prediction
import appmax.evaluation
import appmax.experiment
import appmax.solving
import appmax.optimization
import appmax.visualization


def metrics_callback(ctx, param, value):
    metrics = appmax.optimization.Metrics(0)
    for m in value:
        metrics |= m
    return metrics


@click.command()
@click.argument('dataset')
@click.argument('run-id', default='run')
@click.option('--train', is_flag=True)
@click.option('-m', '--metrics', type=click.Choice(appmax.optimization.Metrics, case_sensitive=False), multiple=True, default=appmax.optimization.Metrics.MAXIMUM, callback=metrics_callback)
@click.option('-b', '--bits', default=8)
@click.option('-s', '--solver', default='')
@click.option('-n', '--num_samples', default=-1)
@click.option('-j', '--jobs', default=1)
def main(dataset, run_id, metrics, train, bits, solver, num_samples, jobs):
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
        case 'year':
            MODEL_FILE = "models/year_prediction_net.pt"
            MODEL_CLASS = year_prediction.YearNet
            data_split = year_prediction.YearPredictionSplit()
        case _:
            raise NotImplementedError(f"'{dataset}' dataset is not available")

    model = MODEL_CLASS()

    if train:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
        print('accelerator', device)
        model.to(device)
        model.fit(data_split.train, data_split.dev)
        model.cpu()
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE).eval()
        model_approx = MODEL_CLASS()
        model_approx.load(MODEL_FILE).eval()
        model_approx.round(bits=bits)

        eval_net = appmax.evaluation.EvaluationNet(model, model_approx, data_split.metadata, seq_name=seq_name).eval()
        samples = appmax.experiment.get_samples(data_split.test, num_samples)

        # with joblib.parallel_config(backend='loky', n_jobs=jobs), appmax.solving.solver_config(solver):
        #     appmax.experiment.run(f'experiments/{dataset}', run_id, eval_net, samples, metrics)

        # appmax.experiment.track_widths(f'experiments/{dataset}/widths', eval_net, samples_dev, num_directions=300)
        # appmax.visualization.plot_tracked_widths({'california': f'experiments/california/widths', 'year': f'experiments/year/widths'})

        appmax.visualization.plot_results(f'experiments/{dataset}', run_id)


if __name__ == '__main__':
    main()
