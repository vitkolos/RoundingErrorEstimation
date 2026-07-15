import torch
import joblib
import click

import appmax.applications
import appmax.evaluation
import appmax.experiment
import appmax.solving
import appmax.optimization


def metrics_callback(ctx, param, value):
    metrics = appmax.optimization.Metrics(0)
    for m in value:
        metrics |= m
    return metrics


@click.command()
@click.argument('experiment')
@click.argument('dataset')
@click.argument('run-id', default='run')
@click.option('-m', '--metrics', type=click.Choice(appmax.optimization.Metrics, case_sensitive=False), multiple=True, default=appmax.optimization.METRICS_ALL, callback=metrics_callback)
@click.option('-b', '--bits', default=8, help='Number of bits used by the quantized network (default: 8).')
@click.option('-s', '--solver', default=appmax.solving.SOLVER_DEFAULT, help='Best options: gurobi, gurobi-barrier, highs (default).')
@click.option('-n', '--num_samples', default='', help='Usage: 3 (samples 0, 1, 2), 5:8 (samples 5, 6, 7); all the samples if left empty.')
@click.option('-j', '--jobs', default=1, help='Number of CPUs used (default: 1).')
def main(experiment, dataset, run_id, metrics, bits, solver, num_samples, jobs):
    """
    AppMax \n
    input: evaluation network (original net. & approximated net. combined), data samples \n
    output: reported metrics
    """

    torch.manual_seed(42)

    bundle = appmax.applications.DataBundle(dataset)
    data_split = bundle.data_split

    model = bundle.load_model()
    model_approx = bundle.load_model()
    model_approx.round(bits=bits)

    eval_net = appmax.evaluation.EvaluationNet(model, model_approx, data_split.metadata).eval()

    with joblib.parallel_config(backend='loky', n_jobs=jobs), appmax.solving.solver_config(solver):
        match experiment:
            case 'quality':
                print('rmse', model.quality('rmse', data_split.test, data_split.metadata.error_scaling))
                print('rmse approx', model_approx.quality('rmse', data_split.test, data_split.metadata.error_scaling))
                print('mae', model.quality('mae', data_split.test, data_split.metadata.error_scaling))
                print('mae approx', model_approx.quality('mae', data_split.test, data_split.metadata.error_scaling))

            case 'widths' | 'union':
                samples_dev = appmax.experiment.get_samples(model.subset(data_split.dev), num_samples)

                if experiment == 'widths':
                    appmax.experiment.track_widths(
                        f'experiments/{dataset}/widths', eval_net, samples_dev, num_directions=300)
                else:
                    appmax.experiment.track_union(
                        f'experiments/{dataset}/union', eval_net, model.layers, samples_dev, num_samples=150)

            case 'single' | 'parallel' | 'mcmc':
                samples_test = appmax.experiment.get_samples(model.subset(data_split.test), num_samples)

                if experiment == 'single':
                    results = appmax.experiment.single(eval_net, model.layers, samples_test[1][1], metrics, debug=True)
                    print(results)
                elif experiment == 'mcmc':
                    sample_initial = samples_test[2][1]
                    union_lp = appmax.optimization.lp_from_net(model.layers, eval_net.metadata.bounds, sample_initial)
                    samples = appmax.optimization.samples_in_polytope(union_lp, sample_initial, num_points=3)
                    print(samples)
                else:
                    appmax.experiment.run_parallel(
                        f'experiments/{dataset}', run_id, eval_net, model.layers, samples_test, metrics)


if __name__ == '__main__':
    main()
