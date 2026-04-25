from typing import NamedTuple
import enum

import torch
import numpy as np

import appmax.neurons
import appmax.evaluation
import appmax.solving
from appmax.solving import Polytope, LinearProgram
from appmax.trainable import Bounds


class Approach(enum.Enum):
    STANDARD = enum.auto()
    WEIGHTED = enum.auto()
    INTEGRAL = enum.auto()


class PolytopeResult(NamedTuple):
    x: torch.Tensor | None  # point where the error function reaches its maximum
    fun: float | None  # value of the error function (in its maximum)
    width: float | None  # polytope mean width


def find_appmax(eval_net: appmax.evaluation.EvaluationNet, sample: torch.Tensor, solver: str, approach: Approach = Approach.STANDARD, debug: bool = False) -> PolytopeResult:
    """'sample' needs to be a single sample (not a batch);\n
    note that in weighted approach, the error is not weighted yet (weight is returned separately)"""
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(eval_net, message, constraints)
    lp = prepare_lp(message, constraints, eval_net.bounds)

    if debug:
        check_feasibility(sample, lp)

    sample_found, err_found, width = None, None, None

    if approach != Approach.INTEGRAL:
        sample_found, err_found = appmax.solving.solve(lp, solver, verbose=debug)
        sample_found = sample_found.reshape_as(sample)

    if approach != Approach.STANDARD:
        measured_polytope = prepare_integral(lp) if approach == Approach.INTEGRAL else lp
        width = polytope_widths(measured_polytope, solver).mean().item()

    return PolytopeResult(sample_found, err_found, width)


def prepare_lp(message: appmax.neurons.Message, constraints: appmax.neurons.Constraints, bounds: Bounds) -> LinearProgram:
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

    objective = message.s_weight.squeeze()
    bias = message.s_bias.item()

    return LinearProgram(bounds, torch.cat(A_ub), torch.cat(b_ub), objective, bias)


def prepare_integral(lp: LinearProgram) -> Polytope:
    # add one variable (error is always non-negative)
    A_ub = torch.hstack([lp.A_ub, torch.zeros(lp.A_ub.shape[0], 1)])
    bounds = Bounds(lp.bounds.seq + [(0.0, None)])

    # add one constraint
    last_row = torch.hstack([-lp.objective, torch.tensor(1.0)])
    A_ub = torch.vstack([A_ub, last_row])
    b_ub = torch.hstack([lp.b_ub, torch.tensor(lp.bias)])

    return Polytope(bounds, A_ub, b_ub)


def polytope_widths(polytope: Polytope, solver: str, num_directions: int = 100, cummulative: bool = False) -> torch.Tensor:
    # variables == dimensions
    num_variables = polytope.A_ub.shape[1]
    directions = torch.randn(num_directions, num_variables)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    lp = LinearProgram(polytope.bounds, polytope.A_ub, polytope.b_ub, objective=torch.empty(0))
    results = appmax.solving.solve(lp, solver, multiple_objectives=directions)
    widths = torch.tensor([(res_max.fun - res_min.fun) for res_min, res_max in results])
    return widths if not cummulative else widths.cumsum(dim=0) / torch.arange(1, num_directions+1)


def check_feasibility(sample: torch.Tensor, polytope: Polytope, abs_tol: float = 1e-6):
    infeasible = False
    sample_flat = sample.flatten()

    if len(polytope.bounds.seq) != sample.numel():
        raise RuntimeError(f'{len(polytope.bounds.seq)} bounds were provided, but there are {sample.numel()} input neurons')

    too_low = torch.nonzero(sample_flat < torch.from_numpy(polytope.bounds.lb)).flatten().tolist()
    too_high = torch.nonzero(sample_flat > torch.from_numpy(polytope.bounds.ub)).flatten().tolist()

    if too_low:
        infeasible = True
        print(f'indices {too_low} < lower bounds')

    if too_high:
        infeasible = True
        print(f'indices {too_high} > upper bounds')

    left_side = polytope.A_ub @ sample_flat
    infeasible_rows = torch.nonzero(left_side > polytope.b_ub + abs_tol).flatten()

    if len(infeasible_rows) > 0:
        infeasible = True

        for i in infeasible_rows:
            print(f'infeasible constraint {i}: {left_side[i].item():.6f} <= {polytope.b_ub[i].item():.6f}')

    if infeasible:
        raise RuntimeError(f'infeasible (check the output above); input tensor:\n{sample}')
