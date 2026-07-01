"""Dimension-generic codegen (`106`): a symbolic ``DimVar('n')`` lowers as a RUNTIME dimension, so
``G.compile`` on the (un-instantiated) symbolic node emits numpy that takes ``n`` from the input
shapes at run time (``np.eye(n)``) and works for ANY ``n`` — no hardcoded literal / unroll."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import grassmann as G
import polyarray as pa
from grassmann.lower.numpy_renderers import op_renderers
from grassmann.syntax.mcalc import build_problem

CASES = [
    ("D(0.5*x'*A*x, x)", {"x": "R^n", "A": "R^n -> R^n"}),        # gradient
    ("D(A*x, x)", {"A": "R^n -> R^n", "x": "R^n"}),               # Jacobian
    ("D(A*exp(x), x)", {"A": "R^n -> R^n", "x": "R^n"}),          # raw-λ vmap (vector wrt)
    ("D(norm2(A*x-y)^2, x)", {"A": "R^n -> R^n", "x": "R^n", "y": "R^n"}),
    ("D(tr(A*X'*B*X*C), X)", {"A": "R^n -> R^n", "B": "R^n -> R^n",
                              "C": "R^n -> R^n", "X": "R^n -> R^n"}),   # matrix wrt (gradient)
    ("D(log(det(inv(X))), X)", {"X": "R^n -> R^n"}),              # matrix wrt via inv/det
    ("D(X*X, X)", {"X": "R^n -> R^n"}),                           # matrix wrt (4-tensor)
    ("x (+) y", {"x": "R^n", "y": "R^n"}),                        # direct sum (⊕) concat
    ("pinv(A)*y", {"A": "R^n -> R^n", "y": "R^n"}),               # pseudo-inverse apply
    ("A\\y", {"A": "R^n -> R^n", "y": "R^n"}),                    # solve / backslash
    ("outer(x, y)", {"x": "R^n", "y": "R^n"}),                    # outer product (matrix)
]


def _emit(expr, decls):
    r = build_problem(expr, decls=decls)
    prog = G.compile(r.primal)   # symbolic — NOT instantiated to a concrete dim
    src = pa.to_numpy_source(prog, func_name="f", op_renderers=op_renderers())
    ns: dict = {}
    exec(src, ns)  # noqa: S102 - the whole point is to run the generated source
    return r, src, ns["f"]


def _vals(decls, n, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in decls.items():
        out[k] = (rng.standard_normal((n, n)) + n * np.eye(n)) if "->" in v else rng.standard_normal(n)
    return out


@pytest.mark.parametrize("expr,decls", CASES)
def test_codegen_is_dimension_generic(expr, decls):
    r, src, f = _emit(expr, decls)
    # no hardcoded 3-dim literal / unrolled element access
    assert "np.eye(3)" not in src and "[0, 0]" not in src, f"non-generic source for {expr}"
    # runs and matches the engine at TWO different n (proving it is genuinely n-polymorphic)
    ps = list(inspect.signature(f).parameters)
    for n, seed in ((4, 1), (7, 2)):
        vals = _vals(decls, n, seed)
        feed = {f"{k}_{r.vars[k].node.uid}": vals[k] for k in vals}
        engine = np.asarray(G.run(r.primal, feed)["result"])
        got = np.asarray(f(*[vals[p.rsplit("_", 1)[0]] for p in ps]))
        assert got.shape == engine.shape
        assert np.allclose(got, engine), f"{expr} codegen ≠ engine at n={n}"
