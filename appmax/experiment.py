from pathlib import Path

import torch
import joblib
import pandas as pd
import tqdm

import appmax.evaluation
import appmax.optimization
from appmax.solving import SOLVER_DEFAULT


def run(
    experiment_path: Path | str,
    run_id: str,
    eval_net: appmax.evaluation.EvaluationNet,
    samples: list,
    first_k: int | None = None,
    use_memory: bool = True,
    show_tensors: bool = False
):
    # prepare dataset
    def get_sample(i): return samples[i][0].clone()  # clone to break connection with Storage
    total_length = len(samples)

    if first_k is not None and first_k > 0:
        total_length = min(total_length, first_k)

    # prepare output dir
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)
    with (experiment_path / '_runs.txt').open('a') as f:
        print(run_id, file=f)

    # activate memory (optional)
    wrapped_step = step
    if use_memory:
        memory = joblib.Memory(experiment_path / 'memory', verbose=0)
        wrapped_step = memory.cache(wrapped_step, ignore=['eval_net', 'input_sample'])

    # setup generators
    wrapped_step = joblib.delayed(wrapped_step)
    para = joblib.Parallel(return_as='generator_unordered')
    results_gen = para(wrapped_step(run_id, i, eval_net, get_sample(i)) for i in range(total_length))
    progress_gen = tqdm.tqdm(results_gen, leave=False, total=total_length)

    # run & process output
    df = pd.DataFrame(progress_gen)
    df = df.set_index('sample_index').sort_index()
    df_errors = df[['error_sample', 'error_nearby']]
    df_errors.to_csv(experiment_path / f'{run_id}_results.csv')
    df_errors.describe(percentiles=[0.5]).to_csv(experiment_path / f'{run_id}_described.csv')
    input_nearby_stack = torch.stack(df['input_nearby'].to_list())
    torch.save(input_nearby_stack, experiment_path / f'{run_id}_tensors.pt')

    if show_tensors:
        def ten2strs(tensor):
            return [f'{x:.2f}' for x in tensor.flatten().tolist()]

        with open(experiment_path / f'{run_id}_tensors.tsv', 'w') as f:
            for i, (tensor_sample, tensor_nearby) in df[['input_sample', 'input_nearby']].iterrows():
                print(i, 'sample', *ten2strs(tensor_sample), sep='\t', file=f)
                print(i, 'nearby', *ten2strs(tensor_nearby), sep='\t', file=f)


def step(run_id: str, sample_index: int, eval_net: appmax.evaluation.EvaluationNet, input_sample: torch.Tensor) -> dict:
    """function for parallel execution
    (run_id and sample_index are used for caching, eval_net and input_sample are ignored)"""
    result = single(eval_net, input_sample)
    result['sample_index'] = sample_index
    return result


def single(eval_net: appmax.evaluation.EvaluationNet, input_sample: torch.Tensor, solver: str = SOLVER_DEFAULT, debug: bool = False) -> dict:
    input_sample_b = input_sample.unsqueeze(0)  # sample -> batch (to support any PyTorch network)

    with torch.no_grad():
        error_sample = eval_net(input_sample_b).item()

    approach = appmax.optimization.Approach.WEIGHTED
    result = appmax.optimization.find_appmax(eval_net, input_sample, solver, approach, debug=debug)

    return {
        'input_sample': input_sample,
        'error_sample': error_sample,
        'input_nearby': result.x,
        'error_nearby': result.fun,
        'polytope_width': result.width,
    }
