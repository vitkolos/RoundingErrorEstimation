import torch
import scipy.optimize
import appmax.neurons
import appmax.evaluation
from appmax.trainable import Bounds


def find_appmax(eval_net: appmax.evaluation.EvaluationNet, sample: torch.Tensor, verbose: bool = True) -> tuple[torch.Tensor, float]:
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(eval_net, message, constraints)
    sample_found, err_found = optimize(message, constraints, bounds=eval_net.bounds, verbose=verbose)
    return sample_found.reshape_as(sample), err_found


def optimize(message: appmax.neurons.Message, constraints: appmax.neurons.Constraints, bounds: Bounds, verbose: bool) -> tuple[torch.Tensor, float]:
    TOL = 0  # 1e-8

    # objective to minimize
    objective = -message.s_weight

    # (U)  Ax + b >= 0
    #         -Ax <= b
    U_weight = -torch.cat(constraints.U_weight)
    U_bias = torch.cat(constraints.U_bias) + TOL

    # (S)  Ax + b <= 0
    #          Ax <= -b
    S_weight = torch.cat(constraints.S_weight)
    S_bias = -torch.cat(constraints.S_bias) + TOL

    c = objective.squeeze().cpu().numpy()
    A_ub = torch.cat((U_weight, S_weight)).cpu().numpy()
    b_ub = torch.cat((U_bias, S_bias)).cpu().numpy()
    result = scipy.optimize.linprog(c, A_ub, b_ub, bounds=bounds, options={'disp': verbose})

    if not result.success:
        match result.status:
            case 2:
                raise RuntimeError('optimization failed, problem infeasible (check if the bounds are set correctly)')
            case 3:
                raise RuntimeError('optimization failed, problem unbounded (check if the bounds are set correctly)')
            case _:
                raise RuntimeError('optimization failed')
    elif result.x is None or result.fun is None:
        raise RuntimeError('result is empty')

    found_x = torch.from_numpy(result.x).to(dtype=torch.get_default_dtype())
    found_maximum = -result.fun + message.s_bias.item()
    return found_x, found_maximum
