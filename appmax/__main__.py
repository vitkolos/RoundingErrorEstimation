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
@click.argument('dataset')
@click.argument('run-id', default='run')
@click.option('-m', '--metrics', type=click.Choice(appmax.optimization.Metrics, case_sensitive=False), multiple=True, default=appmax.optimization.Metrics.MAXIMUM, callback=metrics_callback)
@click.option('-b', '--bits', default=8)
@click.option('-s', '--solver', default='')
@click.option('-n', '--num_samples', default=-1)
@click.option('-j', '--jobs', default=1)
def main(dataset, run_id, metrics, bits, solver, num_samples, jobs):
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
    samples = appmax.experiment.get_samples(model.subset(data_split.test), num_samples)

    with appmax.solving.solver_config(solver):
        results = appmax.experiment.single(eval_net, model.layers, samples[1], metrics, debug=True)
        print(results)

    lp = appmax.optimization.lp_from_net(eval_net, eval_net.metadata.bounds, samples[1])
    print(lp.b_ub.shape[0], 'constraints')

    print('rmse', model.quality('rmse', data_split.test, data_split.metadata.error_scaling))
    print('rmse approx', model_approx.quality('rmse', data_split.test, data_split.metadata.error_scaling))
    print('mae', model.quality('mae', data_split.test, data_split.metadata.error_scaling))
    print('mae approx', model_approx.quality('mae', data_split.test, data_split.metadata.error_scaling))

    # with joblib.parallel_config(backend='loky', n_jobs=jobs), appmax.solving.solver_config(solver):
    #     appmax.experiment.run(f'experiments/{dataset}', run_id, eval_net, model.layers, samples, metrics)

    # appmax.experiment.track_widths(f'experiments/{dataset}/widths', eval_net, samples_dev, num_directions=300)


if __name__ == '__main__':
    main()
