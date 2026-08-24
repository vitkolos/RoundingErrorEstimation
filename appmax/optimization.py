from __future__ import annotations
from dataclasses import dataclass
import enum
import sys

import torch
import numpy as np
import polytopewalk  # type: ignore[import-not-found]

import appmax.neurons
import appmax.evaluation
import appmax.solving
import appmax.logger
from appmax.solving import Polytope, LinearProgram, PolytopeHashable, OptimizationResult
from appmax.trainable import Bounds

NUM_DIRECTIONS = 100
MCMC_NUM_POINTS = 150
MCMC_MAX_POLYTOPES = 40


class Metrics(enum.Flag):
    MAXIMUM = enum.auto()
    WIDTH = enum.auto()
    INTEGRAL = enum.auto()
    UNION = enum.auto()


METRICS_ALL = Metrics.MAXIMUM | Metrics.WIDTH | Metrics.INTEGRAL | Metrics.UNION


@dataclass
class PolytopeResult:
    x: torch.Tensor | None = None  # point where the error function reaches its maximum
    fun: float | None = None  # value of the error function (in its maximum)
    width: float | None = None  # polytope mean width
    width_pt: torch.Tensor | None = None
    integral: float | None = None  # mean width of the extended polytope
    integral_pt: torch.Tensor | None = None
    union: PolytopeResult | None = None  # features of the larger polytope (taken from the original network)
    polytopes: int | None = None  # number of checked polytopes (used only as union.polytopes)
    # logged progress of checking polytopes (used only as union.progress)
    progress: list[tuple[int, float]] | None = None


def analyze_linear_region(
    eval_net: appmax.evaluation.EvaluationNet,
    original_net: torch.nn.Module,
    sample: torch.Tensor,
    metrics: Metrics = Metrics.MAXIMUM,
    num_jobs: int = 1,
    debug: bool = False,
) -> PolytopeResult:
    """'sample' needs to be a single sample (not a batch)"""
    lp = lp_from_net(eval_net, eval_net.metadata.bounds, sample)

    if debug:
        check_feasibility(sample, lp)

    result = PolytopeResult()

    if Metrics.MAXIMUM in metrics or Metrics.UNION in metrics:
        opt_result_initial = appmax.solving.solve(lp, verbose=debug)
        result.x, result.fun = opt_result_initial
        result.x = result.x.reshape_as(sample)

    if Metrics.WIDTH in metrics:
        result.width_pt = polytope_widths(lp, num_jobs=num_jobs)
        result.width = result.width_pt.mean().item()

    if Metrics.INTEGRAL in metrics:
        extended_polytope = prepare_integral(lp)
        result.integral_pt = polytope_widths(extended_polytope, num_jobs=num_jobs)
        result.integral = result.integral_pt.mean().item()

    if Metrics.UNION in metrics:
        lp_initial_hashable = lp.to_polytope_hashable()
        del lp  # to save some memory
        result.union = analyze_union(eval_net, original_net, sample, lp_initial_hashable,
                                     opt_result_initial, num_jobs=num_jobs)

    return result


def lp_from_net(net: torch.nn.Module, bounds: Bounds, sample: torch.Tensor) -> LinearProgram:
    """'sample' needs to be a single sample (not a batch)"""
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(net, message, constraints)
    return lp_from_collected(message, constraints, bounds)


def lp_from_collected(message: appmax.neurons.Message, constraints: appmax.neurons.Constraints, bounds: Bounds) -> LinearProgram:
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

    return LinearProgram(bounds, torch.cat(A_ub), torch.cat(b_ub), objective, bias, maximize=True, neuron_states=constraints.neuron_states)


def prepare_integral(lp: LinearProgram) -> Polytope:
    """generates an extended polytope based on a linear program"""
    # add one variable (error is always non-negative)
    A_ub = torch.hstack([lp.A_ub, torch.zeros(lp.A_ub.shape[0], 1)])
    bounds = Bounds(lp.bounds.seq + [(0.0, None)])

    # add one constraint
    last_row = torch.hstack([-lp.objective, torch.tensor(1.0)])
    A_ub = torch.vstack([A_ub, last_row])
    b_ub = torch.hstack([lp.b_ub, torch.tensor(lp.bias)])

    return Polytope(bounds, A_ub, b_ub)


def polytope_widths(polytope: Polytope, num_directions: int = NUM_DIRECTIONS, cummulative_avg: bool = False, num_jobs: int = 1) -> torch.Tensor:
    """returns widths of the polytope computed from many random directions (or the cummulative average in each step)"""
    # variables == dimensions
    num_variables = polytope.A_ub.shape[1]
    directions = torch.randn(num_directions, num_variables)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    lp = LinearProgram(polytope.bounds, polytope.A_ub, polytope.b_ub, objective=torch.empty(0))
    results = appmax.solving.solve_parallel(lp, directions, num_jobs)
    widths = torch.tensor([(res_max.fun - res_min.fun) for res_min, res_max in results])
    return widths if not cummulative_avg else widths.cumsum(dim=0) / torch.arange(1, num_directions+1)


def analyze_union(
    eval_net: appmax.evaluation.EvaluationNet,
    original_net: torch.nn.Module,
    sample_initial: torch.Tensor,
    lp_initial_hashable: PolytopeHashable,
    opt_result_initial: OptimizationResult,
    num_points: int = MCMC_NUM_POINTS,
    max_polytopes: int = MCMC_MAX_POLYTOPES,
    compute_width: bool = True,
    num_jobs: int = 1,
) -> PolytopeResult:
    union_lp = lp_from_net(original_net, eval_net.metadata.bounds, sample_initial)
    union_result = PolytopeResult()

    if compute_width:
        union_result.width_pt = polytope_widths(union_lp, num_jobs=num_jobs)
        union_result.width = union_result.width_pt.mean().item()

    union = {lp_initial_hashable: opt_result_initial}
    tracking_list = [(1, opt_result_initial)]
    samples = samples_in_polytope(union_lp, sample_initial, num_points)
    union_extend(union, eval_net, samples, max_polytopes, tracking_list)
    union_result.x, union_result.fun = max(union.values(), key=lambda result: result.fun)
    union_result.x = union_result.x.reshape_as(sample_initial)
    union_result.polytopes = len(union)
    union_result.progress = [(n, res.fun) for n, res in tracking_list]
    return union_result


def samples_in_polytope(polytope: Polytope, sample_initial: torch.Tensor, num_points: int, seed: int = 42) -> torch.Tensor:
    A_full, b_full = polytope.get_full_constraints()
    safe_init = move_point_inside(sample_initial.flatten().numpy(), A_full, b_full)

    if safe_init is not None:
        walker = polytopewalk.dense.HitAndRun()
        samples = walker.generateCompleteWalk(
            niter=num_points,
            init=safe_init,
            A=A_full,
            b=b_full,
            burnin=0,  # discard the first few samples
            thin=1,  # only keep every n-th sample to reduce correlation
            seed=seed,
        )
        samples_tensor = torch.from_numpy(samples).to(dtype=torch.get_default_dtype())
        samples_tensor = samples_tensor.reshape(-1, *sample_initial.shape)
    else:
        samples_tensor = torch.empty(0, *sample_initial.shape)

    return samples_tensor


def move_point_inside(point_initial: np.ndarray, A_full: np.ndarray, b_full: np.ndarray) -> np.ndarray | None:
    """the point may be outside the polytope (or too close to its edge) due to rounding errors
    -> we find the Chebyshev center to be on the safe side
    (see https://en.wikipedia.org/w/index.php?title=Chebyshev_center&oldid=1340455583#Linear_programming_problem)"""

    try:
        A_pt, b_pt = torch.from_numpy(A_full), torch.from_numpy(b_full)
        num_vars = point_initial.size
        bounds = Bounds([(None, None)] * (num_vars + 1))
        row_norms = torch.linalg.vector_norm(A_pt, dim=1, keepdim=True)
        A_ex = torch.hstack([A_pt, row_norms])
        objective = torch.tensor([0.0] * num_vars + [1.0])
        lp = LinearProgram(bounds, A_ex, b_pt, objective)
        result = appmax.solving.solve(lp)
        point_new = result.x[:-1].numpy()

        if (A_full @ point_new <= b_full).all():
            # this condition likely always holds
            return point_new
    except RuntimeError as error:
        print(error, file=sys.stderr)

    return None


def union_extend(
    union: dict[PolytopeHashable, OptimizationResult],
    eval_net: appmax.evaluation.EvaluationNet,
    samples: torch.Tensor,
    max_polytopes: int,
    tracking_list: list | None = None
):
    for sample in appmax.logger.progress(samples):
        # we construct the polytope and check if it has already been analyzed
        # (alternative approach: check if 'sample' belongs to any of the analyzed polytopes)
        lp = lp_from_net(eval_net, eval_net.metadata.bounds, sample)
        h = lp.to_polytope_hashable()

        if h not in union:
            union[h] = appmax.solving.solve(lp)

        if tracking_list is not None:
            tracking_list.append((len(union), union[h]))

        if len(union) >= max_polytopes:
            break


def check_feasibility(sample: torch.Tensor, polytope: Polytope, abs_tol: float = 1e-6):
    infeasible = False
    sample_flat = sample.flatten()

    if len(polytope.bounds.seq) != sample.numel():
        raise RuntimeError(
            f'{len(polytope.bounds.seq)} bounds were provided, but there are {sample.numel()} input neurons')

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
