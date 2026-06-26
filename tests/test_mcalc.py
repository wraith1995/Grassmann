"""Unit tests for the matrixcalculus-style ASCII frontend (`grassmann.syntax.mcalc`).

Offline + self-contained (no network oracle — that's `tests/test_mcalc_oracle.py`). Pins the
parser, the type-directed build, the coordinate-free derivatives, the numpy I/O convention
(matrices are rows=output × cols=input, `A*x == A@x`), and user-named dimension variables.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.syntax.mcalc import (
    McalcBuildError,
    build_problem,
    mcbuild,
    parse_ast,
)


# --- parsing ---------------------------------------------------------------

GOLD = [
    "0.5*x'*A*x", "A*exp(x)", "sin(x)'*y", "(y.*v)'*x", "a^b",
    "norm2(A*x-y)^2", "sum(log(exp(-y.*(X*w)) + vector(1)))",
    "tr(A*X'*B*X*C)", "log(det(inv(X)))",
]


@pytest.mark.parametrize("src", GOLD)
def test_gold_examples_parse(src):
    parse_ast(src)   # all 9 matrixcalculus front-page examples must tokenize + parse


def test_precedence_transpose_binds_tighter_than_mul():
    # 0.5*x'*A*x  ==  (((0.5 * (x')) * A) * x)
    from grassmann.syntax.parse import Bin, Post
    ast = parse_ast("0.5*x'*A*x")
    assert isinstance(ast, Bin) and ast.op == "*"
    assert isinstance(ast.left, Bin) and ast.left.op == "*"
    inner = ast.left.left          # (0.5 * x')
    assert isinstance(inner, Bin) and isinstance(inner.right, Post) and inner.right.op == "'"


# --- build + coordinate-free derivative ------------------------------------

def test_quadratic_form_is_coordinate_free_inner():
    p = mcbuild("0.5*x'*A*x", {"x": "vector", "A": "matrix"})
    s = G.unparse(p)
    assert "inner" in s and "A(x)" in s            # ½⟨x, A x⟩, not a concrete matrix product


def test_diff_builds_for_working_examples():
    cases = [
        ("0.5*x'*A*x", "x", {"x": "vector", "A": "matrix"}),
        ("sin(x)'*y", "x", {"x": "vector", "y": "vector"}),
        ("A*exp(x)", "x", {"A": "matrix", "x": "vector"}),
        ("a^b", "a", {"a": "scalar", "b": "scalar"}),
        ("norm2(A*x-y)^2", "x", {"A": "matrix", "x": "vector", "y": "vector"}),
        ("tr(A*X'*B*X*C)", "X",
         {"A": "matrix", "B": "matrix", "C": "matrix", "X": "matrix"}),
    ]
    for expr, wrt, decls in cases:
        assert build_problem(expr, wrt=wrt, decls=decls).derivative is not None


# --- numeric value + gradient vs numpy (matrixcalculus convention) ---------

def test_quadratic_value_and_grad_match_numpy():
    r = build_problem("0.5*x'*A*x", wrt="x", decls={"x": "vector", "A": "matrix"})
    rng = np.random.default_rng(0)
    A, x = rng.standard_normal((5, 5)), rng.standard_normal(5)
    dims = {d: 5 for d in r.dim_names}
    assert np.allclose(r.value({"A": A, "x": x}, dims), 0.5 * x @ A @ x)
    assert np.allclose(r.grad({"A": A, "x": x}, dims), 0.5 * (A + A.T) @ x)


def test_matrix_apply_is_numpy_orientation():
    # A*x must act as numpy A@x (matrix = rows×cols = output×input).
    r = build_problem("A*x", decls={"A": "matrix", "x": "vector"})
    rng = np.random.default_rng(3)
    M, x = rng.standard_normal((3, 4)), rng.standard_normal(4)
    assert np.allclose(r.value({"A": M, "x": x}, {"n0": 4, "n1": 3}), M @ x)


def test_lsq_grad_and_vector_jacobian_match_numpy():
    rng = np.random.default_rng(1)
    M, x, y = rng.standard_normal((3, 4)), rng.standard_normal(4), rng.standard_normal(3)
    r = build_problem("norm2(A*x-y)^2", wrt="x",
                      decls={"A": "matrix", "x": "vector", "y": "vector"})
    d = {"n0": 4, "n1": 3}
    assert np.allclose(r.grad({"A": M, "x": x, "y": y}, d), 2 * M.T @ (M @ x - y))
    j = build_problem("A*exp(x)", wrt="x", decls={"A": "matrix", "x": "vector"})
    assert np.allclose(j.grad({"A": M, "x": x}, d), M @ np.diag(np.exp(x)))


# --- user-named dimension variables ----------------------------------------

def test_named_dims_are_shared_and_run():
    r = build_problem("norm2(A*x-y)^2", wrt="x",
                      decls={"A": "matrix(m,n)", "x": "vector(n)", "y": "vector(m)"})
    assert r.dim_names == ["m", "n"]
    s = G.unparse(r.primal)
    assert "(ℝ^n)* ⊗ ℝ^m" in s and "x : ℝ^n" in s and "y : ℝ^m" in s
    rng = np.random.default_rng(2)
    M, x, y = rng.standard_normal((3, 4)), rng.standard_normal(4), rng.standard_normal(3)
    assert np.allclose(r.grad({"A": M, "x": x, "y": y}, {"m": 3, "n": 4}),
                       2 * M.T @ (M @ x - y))


def test_conflicting_named_dims_raise():
    with pytest.raises(McalcBuildError):
        build_problem("x'*y", wrt="x", decls={"x": "vector(n)", "y": "vector(k)"})


# --- known gaps fail loudly, not silently ----------------------------------

@pytest.mark.parametrize("expr", ["adj(X)"])
def test_known_gaps_raise_clearly(expr):
    with pytest.raises(McalcBuildError):
        mcbuild(expr, {"X": "matrix"})


def test_log_det_inv_builds_diffs_runs():
    # gold example 9: log(det(inv(X)))  ==  -log(det(X))
    r = build_problem("log(det(inv(X)))", wrt="X", decls={"X": "matrix(n,n)"})
    assert r.derivative is not None
    rng = np.random.default_rng(9)
    X = rng.standard_normal((4, 4)) + 4 * np.eye(4)        # well-conditioned, det > 0
    val = r.value({"X": X}, {"n": 4})
    assert np.allclose(val, np.log(np.linalg.det(np.linalg.inv(X))))
    # d/dX log(det(inv(X))) = -X^{-T}, shaped (output, input) like numpy
    g = np.asarray(r.grad({"X": X}, {"n": 4}))
    assert g.shape == (4, 4)
    assert np.allclose(g, -np.linalg.inv(X).T)


# --- constants + relu (gold example 7: logistic loss) ----------------------

def test_logistic_loss_builds_and_runs():
    # gold example 7: vector(1) constant + elementwise + matvec, named dims m=samples, n=features
    expr = "sum(log(exp(-y.*(X*w)) + vector(1)))"
    r = build_problem(expr, wrt="w",
                      decls={"X": "matrix(m,n)", "w": "vector(n)", "y": "vector(m)"})
    rng = np.random.default_rng(7)
    m, n = 5, 3
    X = rng.standard_normal((m, n)); w = rng.standard_normal(n)
    y = rng.choice([-1.0, 1.0], size=m)
    ref = float(np.sum(np.log(np.exp(-y * (X @ w)) + 1.0)))
    assert np.allclose(r.value({"X": X, "w": w, "y": y}, {"m": m, "n": n}), ref)


def test_relu_matches_numpy():
    r = build_problem("relu(x)", decls={"x": "vector"})
    rng = np.random.default_rng(8)
    x = rng.standard_normal(6)
    assert np.allclose(r.value({"x": x}, {d: 6 for d in r.dim_names}), np.maximum(0.0, x))
