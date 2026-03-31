import torch
import numpy as np
import scipy.optimize
import appmax.neurons
import appmax.evaluation
from appmax.trainable import Bounds


def find_appmax(eval_net: appmax.evaluation.EvaluationNet, sample: torch.Tensor, debug: bool = False) -> tuple[torch.Tensor, float]:
    """'sample' needs to be a single sample (not a batch)"""
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(eval_net, message, constraints)
    c, bias, A_ub, b_ub = prepare_lp(message, constraints)

    if debug:
        check_feasibility(sample, A_ub, b_ub, eval_net.bounds)

    sample_found, err_found = optimize(c, bias, A_ub, b_ub, eval_net.bounds, verbose=debug)
    return sample_found.reshape_as(sample), err_found


def prepare_lp(message: appmax.neurons.Message, constraints: appmax.neurons.Constraints) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    TOL = 0  # 1e-8
    A_ub = []
    b_ub = []

    # (U)  Ax + b >= 0
    #         -Ax <= b
    if constraints.U_weight:
        A_ub.append(-torch.cat(constraints.U_weight))
        b_ub.append(torch.cat(constraints.U_bias) + TOL)

    # (S)  Ax + b <= 0
    #          Ax <= -b
    if constraints.S_weight:
        A_ub.append(torch.cat(constraints.S_weight))
        b_ub.append(-torch.cat(constraints.S_bias) + TOL)

    # objective to minimize
    objective = -message.s_weight
    c = objective.squeeze()
    bias = message.s_bias

    return c, bias, torch.cat(A_ub), torch.cat(b_ub)


def optimize(c: torch.Tensor, bias: torch.Tensor, A_ub: torch.Tensor, b_ub: torch.Tensor, bounds: Bounds, verbose: bool) -> tuple[torch.Tensor, float]:
    result = scipy.optimize.linprog(c.numpy(), A_ub.numpy(), b_ub.numpy(), bounds=bounds, options={'disp': verbose})

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
    found_maximum = -result.fun + bias.item()
    return found_x, found_maximum


def check_feasibility(sample: torch.Tensor, A_ub: torch.Tensor, b_ub: torch.Tensor, bounds: Bounds, abs_tol: float = 0.0):
    infeasible = False
    sample_flat = sample.flatten()

    bounds_tensor = torch.atleast_2d(torch.tensor(np.array(bounds, dtype=float)))
    too_low = torch.nonzero(sample_flat < bounds_tensor[:, 0]).flatten().tolist()
    too_high = torch.nonzero(sample_flat > bounds_tensor[:, 1]).flatten().tolist()

    if too_low:
        infeasible = True
        print(f'indices {too_low} < lower bounds')

    if too_high:
        infeasible = True
        print(f'indices {too_high} > upper bounds')

    left_side = A_ub @ sample_flat
    infeasible_rows = torch.nonzero(left_side > b_ub + abs_tol).flatten()

    if len(infeasible_rows) > 0:
        infeasible = True

        for i in infeasible_rows:
            print(f'infeasible constraint {i}: {left_side[i].item():.2f} <= {b_ub[i].item():.2f}')

    if infeasible:
        raise RuntimeError(f'infeasible (check the output above); input tensor:\n{sample}')
