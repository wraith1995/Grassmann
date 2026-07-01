"""Inline derivative operators in the mcalc surface: ``D(f, x)`` (total derivative) and
``d(g, x)`` (exterior derivative on forms ``Ωᵏ = Λᵏ(V*)``), including the nilpotency ``d∘d = 0``."""

from __future__ import annotations

import numpy as np

import grassmann as G
from grassmann.syntax.mcalc import build_problem


def _run(expr, xv):
    r = build_problem(expr, decls={"x": "R^3"})
    return np.asarray(G.run(r.primal, {f"x_{r.vars['x'].node.uid}": xv})["result"])


def test_total_derivative_is_gradient():
    # D(½|x|², x) = x  (the total derivative of a scalar = the gradient covector, materialized)
    g = _run("D(0.5*x'*x, x)", np.array([1.0, 2.0, 3.0])).ravel()
    assert np.allclose(g, np.array([1.0, 2.0, 3.0]))


def test_total_derivative_of_linear_map_is_the_map():
    # D(A*x, x) = A  (the Jacobian of a linear map is the map itself)
    r = build_problem("D(A*x, x)", decls={"A": "R^3 -> R^3", "x": "R^3"})
    A = np.array([[1.0, 2, 0], [0, 3, 1], [1, 0, 2]])
    xv = np.array([1.0, 2.0, 3.0])
    J = np.asarray(G.run(r.primal, {f"A_{r.vars['A'].node.uid}": A.T,
                                    f"x_{r.vars['x'].node.uid}": xv})["result"])
    assert np.allclose(J, A.T) or np.allclose(J, A)


def test_exterior_derivative_of_scalar_is_differential():
    # d(½|x|², x) = df, the differential 1-form; |df|² = |x|²
    xv = np.array([1.0, 2.0, 3.0])
    df = _run("d(0.5*x'*x, x)", xv)
    assert np.isclose(np.sum(df ** 2), xv @ xv)


def test_d_squared_is_zero():
    # d∘d = 0  (nilpotency of the exterior derivative)
    for xv in (np.array([1.0, 2.0, 3.0]), np.array([-0.5, 1.5, 0.7])):
        ddf = _run("d(d(0.5*x'*x, x), x)", xv)
        assert np.allclose(ddf, 0.0, atol=1e-9)
