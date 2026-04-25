import torch
import scipy

import appmax.optimization
from appmax.solving import SOLVER_DEFAULT

ABS_TOL = 0.05


def vertices_to_H(vertices):
    """converts a list of vertices to A_ub @ x <= b_ub representation"""
    hull = scipy.spatial.ConvexHull(vertices)
    A_ub = hull.equations[:, :-1]  # normals
    b_ub = -hull.equations[:, -1]  # offsets
    return torch.from_numpy(A_ub), torch.from_numpy(b_ub)


def mean_width(polytope):
    return appmax.optimization.polytope_widths(polytope, SOLVER_DEFAULT).mean().item()


def test_cube():
    vertices = [
        [x,  y,  z]
        for x in [-1, 1]
        for y in [-1, 1]
        for z in [-1, 1]
    ]
    polytope = appmax.optimization.Polytope([(-2, 2)] * 3, *vertices_to_H(vertices))
    torch.testing.assert_close(mean_width(polytope), expected=3.0, atol=ABS_TOL, rtol=0)


def test_tetrahedron():
    vertices = [
        [1,  1,  1],
        [1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ]
    polytope = appmax.optimization.Polytope([(-2, 2)] * 3, *vertices_to_H(vertices))
    torch.testing.assert_close(mean_width(polytope), expected=2.58, atol=ABS_TOL, rtol=0)


def test_octahedron():
    vertices = [
        [1,  0,  0], [-1,  0,  0],
        [0,  1,  0], [0, -1,  0],
        [0,  0,  1], [0,  0, -1]
    ]
    polytope = appmax.optimization.Polytope([(-2, 2)] * 3, *vertices_to_H(vertices))
    torch.testing.assert_close(mean_width(polytope), expected=1.66, atol=ABS_TOL, rtol=0)
