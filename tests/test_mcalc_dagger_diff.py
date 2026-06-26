"""Regression: ``G.diff`` (covariant AD) through a ``Dagger`` (matrix transpose).

Differentiating ``tr(A·Xᵀ·B·X·C)`` w.r.t. ``X`` used to crash in the synthesizer's
trace rule: the contraction ``unify(factors[j], dual(s_i))`` builds ``Dual(Dual(V))``
when ``s_i`` is already a ``Dual`` (which is exactly what a ``Dagger`` produces), and
the first-order unifier had no double-dual collapse, so

    UnifyError: cannot unify ⊕ⁿ ℝ ~ Dual(Dual(⊕ⁿ ℝ))

The fix canonicalizes the involution ``s** ≡ s`` in :mod:`grassmann.infer.unify`
(``_collapse_double_dual``), which for these finite-dim Euclidean spaces is an
identity, not merely an isomorphism.
"""

from __future__ import annotations

import numpy as np

import grassmann as G
from grassmann.core.space import DimLit
from grassmann.embed.ops import Expr
from grassmann.syntax.mcalc import mcbuild, mcdiff
from grassmann.transform.rewrite import _subst_nat_in_term

_DECLS = {"A": "matrix", "B": "matrix", "C": "matrix", "X": "matrix"}


def _subst_dims(expr: Expr, dims: dict[str, int]) -> Expr:
    node = expr.node
    for name, val in dims.items():
        node = _subst_nat_in_term(node, name, DimLit(val))
    return Expr(node)


def _feed(prog, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Feed each input by its base variable name (``A_3`` ← ``values['A']``)."""
    return {inp.name: values[inp.name.split("_")[0]] for inp in prog.inputs}


def test_dagger_trace_diff_builds() -> None:
    # The bug: this raised UnifyError(... Dual(Dual(...)) ...) before the fix.
    grad = mcdiff("tr(A*X'*B*X*C)", "X", _DECLS)
    assert grad is not None
    # Round-trips through the unparser and mentions the differential var.
    assert "dX" in G.unparse(grad)


def test_dagger_trace_diff_matches_finite_difference() -> None:
    primal = mcbuild("tr(A*X'*B*X*C)", _DECLS)
    grad = mcdiff("tr(A*X'*B*X*C)", "X", _DECLS)

    # Instantiate the symbolic dims to small concrete ints, then compile/run.
    dims = {"n0": 3, "n4": 2, "n6": 4}
    prim_prog = G.compile(_subst_dims(primal, dims))
    grad_prog = G.compile(_subst_dims(grad, dims))

    rng = np.random.default_rng(1)
    # A map space ``~dom ⊗ cod`` lowers to a (dom, cod) matrix.
    values = {
        "A": rng.standard_normal((3, 4)),
        "B": rng.standard_normal((2, 2)),
        "C": rng.standard_normal((4, 3)),
        "X": rng.standard_normal((3, 2)),
    }

    # The lowered derivative is the full Jacobian over the (3, 2)-shaped dX space.
    jac = np.asarray(grad_prog.run(_feed(grad_prog, values))["result"]).reshape(3, 2)

    dX = rng.standard_normal((3, 2))
    grad_dot_dX = float(np.sum(jac * dX))

    def run_primal(vals: dict[str, np.ndarray]) -> float:
        return float(np.asarray(prim_prog.run(_feed(prim_prog, vals))["result"]))

    h = 1e-6
    fd = (
        run_primal({**values, "X": values["X"] + h * dX})
        - run_primal({**values, "X": values["X"] - h * dX})
    ) / (2 * h)

    np.testing.assert_allclose(grad_dot_dX, fd, atol=1e-4)


def test_simpler_dagger_cases_build() -> None:
    # ⟨x, A y⟩ differentiated w.r.t. the matrix A.
    g1 = mcdiff("x'*A*y", "A", {"x": "vector", "y": "vector", "A": "matrix"})
    assert "dA" in G.unparse(g1)
    # tr(A·Xᵀ) — a single Dagger under a trace.
    g2 = mcdiff("tr(A*X')", "X", {"A": "matrix", "X": "matrix"})
    assert "dX" in G.unparse(g2)
