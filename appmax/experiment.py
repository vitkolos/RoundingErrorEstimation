import torch
import joblib
import pandas as pd
import tqdm
from pathlib import Path

import appmax.evaluation
import appmax.optimize


def run(
    experiment_path: Path | str,
    run_id: str,
    eval_net: appmax.evaluation.EvaluationNet,
    samples: list,
    first_k: int | None = None
):
    def get_sample(i):
        return samples[i][0]

    total_length = len(samples)

    if first_k is not None:
        total_length = min(total_length, first_k)

    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)

    with (experiment_path / '_runs.txt').open('a') as f:
        print(run_id, file=f)

    memory = joblib.Memory(experiment_path / 'memory', verbose=0)
    mem_step = memory.cache(step, ignore=['eval_net', 'input_sample'])
    del_step = joblib.delayed(mem_step)
    para = joblib.Parallel(return_as='generator_unordered')
    results_gen = para(del_step(run_id, i, eval_net, get_sample(i)) for i in range(total_length))
    progress_gen = tqdm.tqdm(results_gen, leave=False, total=total_length)

    desired_keys = ['sample_index', 'error_sample', 'error_nearby']
    def filter_keys(result):
        return {key: result[key] for key in desired_keys}

    df = pd.DataFrame(filter_keys(result) for result in progress_gen)
    df.to_csv(experiment_path / f'{run_id}_results.csv')
    df.describe().to_csv(experiment_path / f'{run_id}_described.csv')


def step(run_id: str, sample_index: int, eval_net: appmax.evaluation.EvaluationNet, input_sample: torch.Tensor):
    with torch.no_grad():
        error_sample = eval_net(input_sample).item()
        input_nearby, error_nearby = appmax.optimize.find_appmax(eval_net, input_sample, verbose=False)

    return {
        'sample_index': sample_index,
        'input_sample': input_sample,
        'error_sample': error_sample,
        'input_nearby': input_nearby,
        'error_nearby': error_nearby,
    }
