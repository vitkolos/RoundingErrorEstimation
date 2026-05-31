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

        # appmax.visualization.plot_results(f'experiments/{dataset}', run_id)

        aliases = {'run': 'asym8', 'second': 'asym4'}

        with open('experiments/comparison.html', 'w') as f:
            items = [
                ('experiments/california', ['run', 'sym8']),
                ('experiments/california', ['second', 'sym4']),
                ('experiments/year', ['run', 'sym8']),
                ('experiments/year', ['second', 'sym4']),
            ]
            tables = [appmax.visualization.compare_results(*item, aliases) for item in items]
            f.write(appmax.visualization.tables_to_html(tables))

        indices = sorted(torch.randperm(1000)[:20].tolist())
        datasets = [
            ('experiments/california', california_housing.CaliforniaHousingSplit().metadata.error_scaling),
            ('experiments/year', year_prediction.YearPredictionSplit().metadata.error_scaling),
        ]
        runs = ('run', 'sym8', 'second', 'sym4')

        with open('experiments/points.html', 'w') as f:
            tables = [appmax.visualization.list_points(d, r, 1.0, indices, aliases) for d, _ in datasets for r in runs]
            f.write(appmax.visualization.tables_to_html(tables, into_one=False))

        with open('experiments/points_unscaled.html', 'w') as f:
            tables = [appmax.visualization.list_points(d, r, s, indices, aliases) for d, s in datasets for r in runs]
            f.write(appmax.visualization.tables_to_html(tables, into_one=False))


if __name__ == '__main__':
    main()
