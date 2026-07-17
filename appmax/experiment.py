from pathlib import Path
import time
import signal
import sys
import datetime
import json
import dataclasses

import torch
import joblib
import joblib.externals.loky
import pandas as pd
import numpy as np

import appmax.evaluation
import appmax.optimization
import appmax.trainable
import appmax.solving
import appmax.logger as logger

ERROR_COLS = ['error_sample', 'error_nearby', 'union_error']
RESULT_COLS = ERROR_COLS + ['polytope_width', 'integral', 'union_width', 'union_polytopes', 'time']
UNSCALED_COLS = ERROR_COLS + ['integral']


def get_samples(dataset: appmax.trainable.Dataset, items: str = '') -> list[tuple[int, torch.Tensor]]:
    start, stop = 0, len(dataset)  # type: ignore

    if ':' in items:
        a, b = items.split(':', 2)
        if a:
            start = max(start, int(a))
        if b:
            stop = min(stop, int(b))
    elif items:
        # single item
        start = int(items)
        stop = start + 1

    # clone to break connection with Storage
    return [(i, dataset[i][0].clone()) for i in range(start, stop)]


def run_parallel(
    experiment_path: Path | str,
    run_id: str,
    eval_net: appmax.evaluation.EvaluationNet,
    original_net: torch.nn.Module,
    samples: list[tuple[int, torch.Tensor]],
    metrics: appmax.optimization.Metrics,
    use_memory: bool = True,
    show_tensors: bool = False
):
    # prepare output dir
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    with (experiment_path / '_runs.txt').open('a') as f:
        print(run_id, f'{samples[0][0]}:{samples[-1][0]+1}', eval_net.metadata.error_scaling, sep='\t', file=f)

    # activate memory (optional)
    wrapped_step = step
    if use_memory:
        memory = joblib.Memory(experiment_path / 'memory', verbose=0)
        wrapped_step = memory.cache(wrapped_step, ignore=['eval_net', 'original_net', 'input_sample'])

    # handle SIGTERM (raise SystemExit)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

    try:
        # setup generators & ensure correct solver
        wrapped_step = joblib.delayed(wrapped_step)
        init_kwargs = {'initializer': appmax.solving.init_worker, 'initargs': (appmax.solving._active_solver.get(),)}
        with joblib.Parallel(return_as='generator_unordered', **init_kwargs) as para:
            results_gen = para(wrapped_step(run_id, i, metrics, eval_net, original_net, sample)
                               for i, sample in samples)
            progress_gen = logger.progress(results_gen, total=len(samples), smoothing=0, main=True)

            # run & save output
            rows = []
            for row in progress_gen:
                df_running = pd.DataFrame([row]).set_index('sample_index')[RESULT_COLS]
                df_running.to_csv(experiment_path / f'{run_id}_running.csv', mode='a', header=not rows)
                rows.append(row)
    finally:
        # shutdown dangling loky workers
        joblib.externals.loky.get_reusable_executor().shutdown(wait=True, kill_workers=True)

    # process output
    df = pd.DataFrame(rows)
    df = df.set_index('sample_index').sort_index()
    df_results = df[RESULT_COLS]
    df_results.to_csv(experiment_path / f'{run_id}_results.csv')
    describe(df_results).to_csv(experiment_path / f'{run_id}_described.csv')

    # unscale the errors and integrals
    df_results_unscaled = df_results.copy()
    df_results_unscaled.loc[:, UNSCALED_COLS] *= eval_net.metadata.error_scaling
    describe(df_results_unscaled).to_csv(experiment_path / f'{run_id}_described_unscaled.csv')

    # save found points where error is maximum
    found = {}

    for column in ['input_nearby', 'union_input']:
        if not df[column].isna().any():
            found[column] = torch.stack(df[column].to_list())

    if found:
        torch.save(found, experiment_path / f'{run_id}_tensors.pt')

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
    weighted = {}
    polytope_widths = df_results.get('polytope_width')
    union_widths = df_results.get('union_width')

    def compute_weighted_average(column, weights, sum_only=False):
        if weights is not None and weights.sum() > 0 and not df_results[column].isna().any():
            if sum_only:
                weighted[column] = df_results[column].sum() / weights.sum()
            else:
                weighted[column] = np.average(df_results[column], weights=weights)

    compute_weighted_average('error_sample', polytope_widths)
    compute_weighted_average('error_nearby', polytope_widths)
    compute_weighted_average('integral', polytope_widths, sum_only=True)  # integrals are already "weighted"
    compute_weighted_average('union_error', union_widths)

    if weighted:
        described.loc['weighted'] = pd.Series(weighted)

    return described


def step(
    run_id: str,
    sample_index: int,
    metrics: appmax.optimization.Metrics,
    eval_net: appmax.evaluation.EvaluationNet,
    original_net: torch.nn.Module,
    input_sample: torch.Tensor
) -> dict:
    """function for parallel execution
    (run_id & sample_index & metrics are used for caching, eval_net & original_net & input_sample are ignored)"""
    start_time = time.time()
    result = single(eval_net, original_net, input_sample, metrics)
    result['sample_index'] = sample_index
    result['time'] = time.time() - start_time
    return result


def run_batch(
    experiment_path: Path | str,
    run_id: str,
    eval_net: appmax.evaluation.EvaluationNet,
    original_net: torch.nn.Module,
    samples: list[tuple[int, torch.Tensor]],
    metrics: appmax.optimization.Metrics
):
    directory = Path(experiment_path) / run_id
    directory.mkdir(parents=True, exist_ok=True)

    for i, input_sample in logger.progress(samples, main=True, disable=(len(samples) == 1)):
        file_stem = f'point_{i:04d}'

        if (directory / f'{file_stem}.pt').is_file():
            print('skipping', file_stem, file=sys.stderr)
            continue

        data = {}
        data['sample_index'] = i
        data['date'] = str(datetime.datetime.now())

        start_time = time.time()
        results = single(eval_net, original_net, input_sample, metrics, preserve_structure=True)
        data['time'] = time.time() - start_time

        data['result_sample'] = dataclasses.asdict(results['result_sample'])
        data['result_nearby'] = dataclasses.asdict(results['result_nearby'])

        torch.save(data, directory / f'{file_stem}.pt')

        with open(directory / f'{file_stem}.json', 'w') as file_json:
            torch.set_printoptions(threshold=50)
            json.dump(data, file_json, default=str, indent=4)


def load_batch_results(experiment_path: Path | str, run_id: str) -> list[dict]:
    directory = Path(experiment_path) / run_id
    results = [torch.load(f) for f in directory.glob('point_*.pt')]
    results.sort(key=lambda x: x['sample_index'])
    return results


def single(
    eval_net: appmax.evaluation.EvaluationNet,
    original_net: torch.nn.Module,
    input_sample: torch.Tensor,
    metrics: appmax.optimization.Metrics,
    preserve_structure: bool = False,
    debug: bool = False
) -> dict:
    input_sample_b = input_sample.unsqueeze(0)  # sample -> batch (to support any PyTorch network)

    with torch.no_grad():
        error_sample = eval_net(input_sample_b).item()

    result = appmax.optimization.analyze_linear_region(eval_net, original_net, input_sample, metrics, debug=debug)

    if preserve_structure:
        return {
            'result_sample': appmax.optimization.PolytopeResult(input_sample, error_sample),
            'result_nearby': result
        }
    else:
        to_flat_dict(input_sample, error_sample, result)


def to_flat_dict(input_sample, error_sample, result: appmax.optimization.PolytopeResult) -> dict:
    return {
        'input_sample': input_sample,
        'error_sample': error_sample,
        'input_nearby': result.x,
        'error_nearby': result.fun,
        'polytope_width': result.width,
        'integral': result.integral,
        'union_input': result.union.x if result.union else None,
        'union_error': result.union.fun if result.union else None,
        'union_width': result.union.width if result.union else None,
        'union_polytopes': result.union.polytopes if result.union else None,
    }


def track_widths(experiment_path: Path | str, eval_net: appmax.evaluation.EvaluationNet, samples: list[tuple[int, torch.Tensor]], num_directions: int):
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    assert eval_net.metadata.bounds is not None
    data = []

    def extend_data(i: int, type_: str, widths: torch.Tensor):
        data.extend([{'sample': i, 'type': type_, 'directions': d+1, 'width': w.item()} for d, w in enumerate(widths)])

    for i, sample in logger.progress(samples, main=True):
        lp = appmax.optimization.lp_from_net(eval_net, eval_net.metadata.bounds, sample)
        polytope_widths = appmax.optimization.polytope_widths(lp, num_directions, cummulative_avg=True)
        extend_data(i, 'polytope', polytope_widths)
        extended_polytope = appmax.optimization.prepare_integral(lp)
        integral_widths = appmax.optimization.polytope_widths(extended_polytope, num_directions, cummulative_avg=True)
        extend_data(i, 'integral', integral_widths)

    pd.DataFrame(data).to_csv(experiment_path / 'data.csv')


def track_union(
    experiment_path: Path | str,
    eval_net: appmax.evaluation.EvaluationNet,
    original_net: torch.nn.Module,
    samples_initial: list[tuple[int, torch.Tensor]],
    num_samples: int
):
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    assert eval_net.metadata.bounds is not None
    data = []

    for i, sample in logger.progress(samples_initial, main=True):
        lp = appmax.optimization.lp_from_net(eval_net, eval_net.metadata.bounds, sample)
        opt_result_initial = appmax.solving.solve(lp)
        maximum = opt_result_initial.fun
        result = appmax.optimization.analyze_union(
            eval_net, original_net, sample, lp, opt_result_initial, num_samples=num_samples, compute_width=False)

        for j, (polytopes, fun) in enumerate(result.progress):
            maximum = max(maximum, fun)
            data.append({'sample': i, 'point': j, 'polytopes': polytopes, 'fun': fun, 'max': maximum})

    pd.DataFrame(data).to_csv(experiment_path / 'data.csv')
