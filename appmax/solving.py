import dataclasses
from typing import NamedTuple

import torch
import numpy as np
import tqdm
import scipy.optimize
import gurobipy

from appmax.trainable import Bounds


@dataclasses.dataclass
class Polytope:
    bounds: Bounds
    A_ub: torch.Tensor
    b_ub: torch.Tensor


@dataclasses.dataclass
class LinearProgram(Polytope):
    objective: torch.Tensor
    bias: float = 0.0
    maximize: bool = True  # used only if multiple_objectives == False
    multiple_objectives: bool = False


class OptimizationResult(NamedTuple):
    x: torch.Tensor
    fun: float


SOLVER_DEFAULT = 'highs'


def get_min_max_lps(lp: LinearProgram, objective: torch.Tensor) -> tuple[LinearProgram, LinearProgram]:
    lp_min = dataclasses.replace(lp, objective=objective, multiple_objectives=False, maximize=False)
    lp_max = dataclasses.replace(lp, objective=objective, multiple_objectives=False, maximize=True)
    return lp_min, lp_max


def solve(lp: LinearProgram, solver: str = SOLVER_DEFAULT, verbose: bool = False) -> OptimizationResult | list[tuple[OptimizationResult, OptimizationResult]]:
    solver_lower = solver.lower()

    if lp.multiple_objectives and solver_lower != 'gurobi':
        lps = (get_min_max_lps(lp, objective) for objective in tqdm.tqdm(lp.objective, leave=False))
        return [(solve(lp_min, solver, verbose), solve(lp_max, solver, verbose)) for lp_min, lp_max in lps]

    match solver_lower:
        case 'highs': return solve_scipy(lp, verbose)
        case 'gurobi': return solve_gurobi(lp, verbose)
        case 'cuopt': return solve_cuopt(lp, verbose)

    return solve_ortools(lp, solver, verbose)


def solve_scipy(lp: LinearProgram, verbose: bool) -> OptimizationResult:
    # we want to maximize the objective -> we minimize cx (for Ax <= b)
    sense = -1 if lp.maximize else 1
    c = sense * lp.objective
    result = scipy.optimize.linprog(c.numpy(), lp.A_ub.numpy(), lp.b_ub.numpy(),
                                    bounds=lp.bounds, options={'disp': verbose})

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
    found_maximum = sense * result.fun + lp.bias
    return OptimizationResult(found_x, found_maximum)


def solve_ortools(lp: LinearProgram, solver_name: str, verbose: bool) -> OptimizationResult:
    from ortools.linear_solver import pywraplp

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver(solver_name)
    if not solver:
        raise ValueError(f"solver '{solver_name}' could not be created")

    if verbose:
        solver.EnableOutput()

    num_constraints, num_variables = lp.A_ub.shape

    vars = []
    for j in range(num_variables):
        lb, ub = lp.bounds[j]
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
    objective.SetOptimizationDirection(lp.maximize)

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status != solver.OPTIMAL:
        if status == solver.FEASIBLE:
            raise RuntimeError('a potentially suboptimal solution was found')
        else:
            raise RuntimeError('the solver could not solve the problem')

    found_x = torch.tensor([var.solution_value() for var in vars])
    found_maximum = objective.Value() + lp.bias
    return OptimizationResult(found_x, found_maximum)


def solve_cuopt(lp: LinearProgram, verbose: bool) -> OptimizationResult:
    from cuopt.linear_programming import problem, solver, solver_settings  # pyright: ignore[reportMissingImports]

    p = problem.Problem('AppMax')
    num_constraints, num_variables = lp.A_ub.shape

    vars = []
    for j in range(num_variables):
        lb, ub = lp.bounds[j]
        vars.append(p.addVariable(
            lb if lb is not None else -solver.infinity(),
            ub if ub is not None else solver.infinity(),
        ))

    b_ub = lp.b_ub.tolist()
    for i in range(num_constraints):
        nz = torch.nonzero(lp.A_ub[i]).squeeze()
        active_vars = [vars[j] for j in nz.tolist()]
        coeffs = lp.A_ub[i, nz].tolist()
        expr = problem.LinearExpression(active_vars, coeffs, 0.0)
        p.addConstraint(expr <= b_ub[i])

    sense = problem.MAXIMIZE if lp.maximize else problem.MINIMIZE
    p.setObjective(problem.LinearExpression(vars, lp.objective.tolist(), 0.0), sense=sense)

    settings = solver_settings.SolverSettings()
    settings.set_parameter(solver.solver_parameters.CUOPT_METHOD, solver_settings.SolverMethod.PDLP)
    p.solve(settings)

    if p.Status.name != 'Optimal':
        raise RuntimeError(f'Problem status: {p.Status.name}')

    found_x = torch.tensor([var.getValue() for var in vars])
    found_maximum = p.ObjValue + lp.bias
    return OptimizationResult(found_x, found_maximum)


def solve_gurobi(lp: LinearProgram, verbose: bool) -> OptimizationResult | list[tuple[OptimizationResult, OptimizationResult]]:
    num_variables = lp.A_ub.shape[1]
    lb = [lb if lb is not None else float('-inf') for lb, _ in lp.bounds]
    ub = [ub if ub is not None else float('inf') for _, ub in lp.bounds]

    with gurobipy.Env(empty=True) as env:
        env.setParam('LogToConsole', int(verbose))
        env.start()

        with gurobipy.Model(env=env) as model:
            x = model.addMVar(shape=num_variables, lb=np.array(lb), ub=np.array(ub))
            model.addConstr(lp.A_ub.numpy() @ x <= lp.b_ub.numpy())

            def optimize(objective: torch.Tensor, maximize: bool) -> OptimizationResult:
                sense = gurobipy.GRB.MAXIMIZE if maximize else gurobipy.GRB.MINIMIZE
                model.setObjective(objective.numpy() @ x, sense)
                model.optimize()

                if model.Status == gurobipy.GRB.OPTIMAL:
                    found_x = torch.from_numpy(x.X).to(dtype=torch.get_default_dtype())
                    found_maximum = model.ObjVal + lp.bias
                    return OptimizationResult(found_x, found_maximum)
                else:
                    raise RuntimeError(f'optimization ended with status {model.Status}')

            if lp.multiple_objectives:
                return [(optimize(o, maximize=False), optimize(o, maximize=True)) for o in tqdm.tqdm(lp.objective, leave=False)]
            else:
                return optimize(lp.objective, lp.maximize)
