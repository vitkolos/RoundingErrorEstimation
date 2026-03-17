import torch
from appmax.applications import mnist
import appmax.evaluation
import appmax.optimize


def run(variant_name, eval_net: appmax.evaluation.EvaluationNet, input_sample: torch.Tensor):
    error_sample = eval_net(input_sample).item()
    input_nearby, error_nearby = appmax.optimize.find_appmax(eval_net, input_sample, verbose=False)
    return {
        'variant_name': variant_name,
        # 'input_sample': input_sample,
        'error_sample': error_sample,
        # 'input_nearby': input_nearby,
        'error_nearby': error_nearby,
    }
