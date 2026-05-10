from pathlib import Path
import typing
import time

import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import appmax.evaluation
import appmax.optimization
import appmax.trainable
import appmax.logger as logger


def get_samples(dataset: appmax.trainable.Dataset, first_k: int | None = None) -> list[torch.Tensor]:
    total_length = len(dataset)  # type: ignore

    if first_k is not None and first_k > 0:
        total_length = min(total_length, first_k)

    # clone to break connection with Storage
    return [dataset[i][0].clone() for i in range(total_length)]


def run(
    experiment_path: Path | str,
    run_id: str,
    eval_net: appmax.evaluation.EvaluationNet,
    samples: list[torch.Tensor],
    metrics: appmax.optimization.Metrics,
    use_memory: bool = True,
    show_tensors: bool = False
):
    # prepare output dir
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    with (experiment_path / '_runs.txt').open('a') as f:
        print(run_id, len(samples), eval_net.metadata.error_scaling, sep='\t', file=f)

    # activate memory (optional)
    wrapped_step = step
    if use_memory:
        memory = joblib.Memory(experiment_path / 'memory', verbose=0)
        wrapped_step = memory.cache(wrapped_step, ignore=['eval_net', 'input_sample'])

    # setup generators
    wrapped_step = joblib.delayed(wrapped_step)
    para = joblib.Parallel(return_as='generator_unordered')
    results_gen = para(wrapped_step(run_id, i, metrics, eval_net, sample) for i, sample in enumerate(samples))
    progress_gen = logger.progress(results_gen, total=len(samples), smoothing=0, main=True)

    # run & process output
    df = pd.DataFrame(progress_gen)
    df = df.set_index('sample_index').sort_index()
    error_cols = ['error_sample', 'error_nearby']
    df_results = df[error_cols + ['polytope_width', 'integral', 'time']]
    df_results.to_csv(experiment_path / f'{run_id}_results.csv')
    df_described = describe(df_results)
    df_described.to_csv(experiment_path / f'{run_id}_described.csv')

    # unscale the errors and integrals
    df_results_unscaled = df_results.copy()
    df_results_unscaled.loc[:, error_cols + ['integral']] *= eval_net.metadata.error_scaling
    describe(df_results_unscaled).to_csv(experiment_path / f'{run_id}_described_unscaled.csv')

    # save found points where error is maximum
    if not df['input_nearby'].isna().any():
        input_nearby_stack = torch.stack(df['input_nearby'].to_list())
        torch.save(input_nearby_stack, experiment_path / f'{run_id}_tensors.pt')

    # show both the sample and nearby points
    if show_tensors:
        def ten2strs(tensor):
            return [f'{x:.2f}' for x in tensor.flatten().tolist()]

        with open(experiment_path / f'{run_id}_tensors.tsv', 'w') as f:
            for i, (tensor_sample, tensor_nearby) in df[['input_sample', 'input_nearby']].iterrows():
                print(i, 'sample', *ten2strs(tensor_sample), sep='\t', file=f)
                print(i, 'nearby', *ten2strs(tensor_nearby), sep='\t', file=f)


def describe(df_results: pd.DataFrame) -> pd.DataFrame:
    described = df_results.describe(percentiles=[0.5])
    weights = df_results.get('polytope_width')

    if weights is not None and weights.sum() > 0:
        weighted = {
            k: np.average(df_results[k], weights=weights)
            for k in ['error_sample', 'error_nearby'] if not df_results[k].isna().any()
        }

        if not df_results['integral'].isna().any():
            # integrals are already "weighted"
            weighted['integral'] = df_results['integral'].sum() / weights.sum()

        described.loc['weighted'] = pd.Series(weighted)

    return described


def step(
    run_id: str,
    sample_index: int,
    metrics: appmax.optimization.Metrics,
    eval_net: appmax.evaluation.EvaluationNet,
    input_sample: torch.Tensor
) -> dict:
    """function for parallel execution
    (run_id & sample_index & metrics are used for caching, eval_net & input_sample are ignored)"""
    start_time = time.time()
    result = single(eval_net, input_sample, metrics)
    result['sample_index'] = sample_index
    result['time'] = time.time() - start_time
    return result


def single(
    eval_net: appmax.evaluation.EvaluationNet,
    input_sample: torch.Tensor,
    metrics: appmax.optimization.Metrics,
    debug: bool = False
) -> dict:
    input_sample_b = input_sample.unsqueeze(0)  # sample -> batch (to support any PyTorch network)

    with torch.no_grad():
        error_sample = eval_net(input_sample_b).item()

    result = appmax.optimization.analyze_linear_region(eval_net, input_sample, metrics, debug=debug)

    return {
        'input_sample': input_sample,
        'error_sample': error_sample,
        'input_nearby': result.x,
        'error_nearby': result.fun,
        'polytope_width': result.width,
        'integral': result.integral,
    }


def track_widths(experiment_path: Path | str, eval_net: appmax.evaluation.EvaluationNet, samples: list[torch.Tensor], num_directions: int):
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    data = []

    def extend_data(i: int, type_: str, widths: torch.Tensor):
        data.extend([{'sample': i, 'type': type_, 'directions': d+1, 'width': w.item()} for d, w in enumerate(widths)])

    for i, sample in enumerate(logger.progress(samples, main=True)):
        lp = appmax.optimization.lp_from_eval_net(eval_net, sample)
        polytope_widths = appmax.optimization.polytope_widths(lp, num_directions, cummulative_avg=True)
        extend_data(i, 'polytope', polytope_widths)
        extended_polytope = appmax.optimization.prepare_integral(lp)
        integral_widths = appmax.optimization.polytope_widths(extended_polytope, num_directions, cummulative_avg=True)
        extend_data(i, 'integral', integral_widths)

    pd.DataFrame(data).to_csv(experiment_path / 'data.csv')


def plot_tracked_widths(experiments: dict[str, str]):
    experiment_paths = {e: Path(p) for e, p in experiments.items()}
    data, grouped = {}, {}
    types = ['polytope', 'integral']
    first_k = 10

    for e, p in experiment_paths.items():
        data[e] = pd.read_csv(p / 'data.csv', index_col=0)
        grouped[e] = data[e].groupby(['sample', 'type'])

    def plot_chart(category, name, identifiers):
        for experiment, key, label in identifiers:
            group_data = grouped[experiment].get_group(key)
            plt.plot(group_data['directions'], group_data['width'], label=label)

        line = {'c': 'black', 'ls': 'dotted'}
        plt.axvline(50, **line)
        plt.axvline(100, **line, lw=2)
        plt.axvline(150, **line)
        plt.axvline(200, **line)

        if any(x[2] for x in identifiers):
            plt.legend()

        experiment_first = identifiers[0][0]
        category_path = experiment_paths[experiment_first] / category
        category_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(category_path / f'{name}.png')
        plt.close()

    def plot_charts(category, name, identifiers):
        num = len(identifiers)
        fig, axes = plt.subplots(num, figsize=(6.4, 3*num))

        for ax, (experiment, key, label) in zip(axes, identifiers):
            group_data = grouped[experiment].get_group(key)
            ax.plot(group_data['directions'], group_data['width'], label=label)
            line = {'c': 'black', 'ls': 'dotted'}
            ax.axvline(50, **line)
            ax.axvline(100, **line, lw=2)
            ax.axvline(150, **line)
            ax.axvline(200, **line)
            ax.legend()

        experiment_first = identifiers[0][0]
        category_path = experiment_paths[experiment_first] / category
        category_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(category_path / f'{name}.png')
        plt.close()

    # one chart per polytope
    for experiment in experiments.keys():
        for key in grouped[experiment].groups.keys():
            sample, type_ = typing.cast(tuple[int, str], key)
            plot_chart('single', f'{type_}_{sample+1:02d}', [(experiment, key, None)])

    # polytope and integral in the same chart
    for experiment in experiments.keys():
        for sample in range(first_k):
            plot_chart('both', f'{sample+1:02d}', [(experiment, (sample, t), t) for t in types])

    # several polytopes in one chart
    for experiment in experiments.keys():
        for type_ in types:
            plot_chart('combined', type_, [(experiment, (i, type_), None) for i in range(first_k)])

    # several datasets in one chart
    for type_ in types:
        for sample in range(first_k):
            plot_charts('different', f'{type_}_{sample+1:02d}', [(e, (sample, type_), e) for e in experiments.keys()])
