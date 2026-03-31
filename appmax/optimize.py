from typing import NamedTuple
import torch
import numpy as np
import scipy.optimize
from ortools.linear_solver import pywraplp
import gurobipy

import appmax.neurons
import appmax.evaluation
from appmax.trainable import Bounds, bounds2list

SOLVER_HIGHS = 'highs'
SOLVER_GUROBI = 'gurobi'
SOLVER_DEFAULT = SOLVER_HIGHS


class LinearProgram(NamedTuple):
    objective: torch.Tensor
    bias: torch.Tensor
    A_ub: torch.Tensor
    b_ub: torch.Tensor


class OptimizationResult(NamedTuple):
    x: torch.Tensor
    fun: float


def find_appmax(eval_net: appmax.evaluation.EvaluationNet, sample: torch.Tensor, solver: str, debug: bool = False) -> OptimizationResult:
    """'sample' needs to be a single sample (not a batch)"""
    constraints = appmax.neurons.Constraints()
    message = appmax.neurons.Message(sample)
    message = appmax.neurons.collect(eval_net, message, constraints)
    lp = prepare_lp(message, constraints)

    if debug:
        check_feasibility(sample, lp.A_ub, lp.b_ub, eval_net.bounds)

    sample_found, err_found = optimize(lp, eval_net.bounds, solver, verbose=debug)
    return OptimizationResult(sample_found.reshape_as(sample), err_found)


def prepare_lp(message: appmax.neurons.Message, constraints: appmax.neurons.Constraints) -> LinearProgram:
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
    bias = message.s_bias

    return LinearProgram(objective, bias, torch.cat(A_ub), torch.cat(b_ub))


def optimize(lp: LinearProgram, bounds: Bounds, solver: str = SOLVER_DEFAULT, verbose: bool = False) -> OptimizationResult:
    if solver == SOLVER_HIGHS:
        return optimize_scipy(lp, bounds, verbose)
    elif solver == SOLVER_GUROBI:
        return optimize_gurobi(lp, bounds, verbose)
    else:
        return optimize_ortools(lp, bounds, solver, verbose)


def optimize_scipy(lp: LinearProgram, bounds: Bounds, verbose: bool) -> OptimizationResult:
    # we want to maximize the objective -> we minimize cx (for Ax <= b)
    c = -lp.objective
    result = scipy.optimize.linprog(c.numpy(), lp.A_ub.numpy(), lp.b_ub.numpy(),
                                    bounds=bounds, options={'disp': verbose})

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
    found_maximum = -result.fun + lp.bias.item()
    return OptimizationResult(found_x, found_maximum)


def optimize_ortools(lp: LinearProgram, bounds: Bounds, solver_name: str, verbose: bool) -> OptimizationResult:
    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver(solver_name)
    if not solver:
        raise ValueError(f"solver '{solver_name}' could not be created")

    if verbose:
        solver.EnableOutput()

    num_constraints, num_variables = lp.A_ub.shape
    bounds = bounds2list(bounds, num_variables)

    vars = []
    for j in range(num_variables):
        lb, ub = bounds[j]
        vars.append(solver.NumVar(
            lb if lb is not None else -solver.infinity(),
            ub if ub is not None else solver.infinity(),
            f'x{j}'
        ))

    b_ub = lp.b_ub.tolist()
    for i in range(num_constraints):
        constraint = solver.RowConstraint(-solver.infinity(), b_ub[i])
        for var, coef in zip(vars, lp.A_ub[i].tolist()):
            if coef != 0:
                constraint.SetCoefficient(var, coef)

    objective = solver.Objective()
    for var, coef in zip(vars, lp.objective.tolist()):
        objective.SetCoefficient(var, coef)
    objective.SetMaximization()

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status != solver.OPTIMAL:
        if status == solver.FEASIBLE:
            raise RuntimeError('a potentially suboptimal solution was found')
        else:
            raise RuntimeError('the solver could not solve the problem')

    found_x = torch.tensor([var.solution_value() for var in vars])
    found_maximum = objective.Value() + lp.bias.item()
    return OptimizationResult(found_x, found_maximum)


def optimize_gurobi(lp: LinearProgram, bounds: Bounds, verbose: bool) -> OptimizationResult:
    num_variables = lp.objective.shape[0]
    bounds = bounds2list(bounds, num_variables)
    lb = [lb if lb is not None else float('-inf') for lb, _ in bounds]
    ub = [ub if ub is not None else float('inf') for _, ub in bounds]

    with gurobipy.Model() as model:
        x = model.addMVar(shape=num_variables, lb=lb, ub=ub)
        model.setObjective(lp.objective.numpy() @ x, gurobipy.GRB.MAXIMIZE)
        model.addConstr(lp.A_ub.numpy() @ x <= lp.b_ub.numpy())
        model.optimize()

        if model.Status == gurobipy.GRB.OPTIMAL:
            found_x = torch.from_numpy(x.X).to(dtype=torch.get_default_dtype())
            found_maximum = model.ObjVal + lp.bias.item()
            return OptimizationResult(found_x, found_maximum)
        else:
            raise RuntimeError(f'optimization ended with status {model.Status}')


def check_feasibility(sample: torch.Tensor, A_ub: torch.Tensor, b_ub: torch.Tensor, bounds: Bounds, abs_tol: float = 1e-6):
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
            print(f'infeasible constraint {i}: {left_side[i].item():.6f} <= {b_ub[i].item():.6f}')

    if infeasible:
        raise RuntimeError(f'infeasible (check the output above); input tensor:\n{sample}')
