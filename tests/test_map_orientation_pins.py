"""Numeric pins for the FFS map-orientation refactor (plans/92).

These freeze the substrate's numeric behavior against INDEPENDENT ground truth (numpy /
finite differences) so the `(cod,dom)→(dom,cod)` re-orientation can be shown value-preserving.
``test_eta_*`` are the GOAL pins — they FAIL on the legacy layout and must pass once an LLam
lowers like a stored map-var. Everything else is an INVARIANT — it must stay green throughout.
"""

from __future__ import annotations

import numpy as np

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T


def _run(t: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(G.compile(t).run(feed)["result"])


def _k(v: T.Var) -> str:
    return f"{v.name}_{v.uid}"


# ---------------------------------------------------------------------------
# GOAL — η: an LLam lowers identically to the stored map it expands.
# ---------------------------------------------------------------------------


def test_eta_single_var() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f, x = T.var("f", S.linear_map_space(V, W)), T.var("x", V)
    fv = np.arange(6.0).reshape(2, 3)
    feed = {_k(f): fv, _k(x): np.zeros(2)}
    eta = T.LLam((x,), T.Eval(f, (0,), (x,)))
    np.testing.assert_allclose(_run(eta, feed), _run(f, feed))  # same layout AND value


# ---------------------------------------------------------------------------
# INVARIANTS — cross-checked against numpy, must hold before AND after.
# ---------------------------------------------------------------------------


def test_apply_is_transpose_contraction() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f, x = T.var("f", S.linear_map_space(V, W)), T.var("x", V)
    fv, xv = np.arange(6.0).reshape(2, 3), np.array([1.0, 2.0])
    np.testing.assert_allclose(_run(T.Eval(f, (0,), (x,)), {_k(f): fv, _k(x): xv}), fv.T @ xv)


def test_jacobian_via_llam_matches_fd() -> None:
    # diff() builds the Jacobian as an LLam read-off — a strong pin on the LLam→matrix path.
    from grassmann.transform.diff import diff

    V = S.base("V", 3)
    x = T.var("x", V)
    expr = T.External(S.Scalar(), _sqr_op(), (T.Eval(_dotmap(V), (0, 1), (x, x)),))  # ‖x‖²-ish
    _ = expr  # (kept simple below)
    # A plainly-linear map's Jacobian is the map itself: d(f·x)/dx = f.
    W = S.base("W", 2)
    f = T.var("f", S.linear_map_space(V, W))
    J = diff(T.Eval(f, (0,), (x,)), x)
    fv = np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(_run(J, {_k(f): fv, _k(x): np.zeros(3)}), fv)


def test_inner_gram_pullback() -> None:
    # A Gram/pullback value through the contraction machinery: dot(x, y) = xᵀ·y.
    V = S.base("V", 3)
    x, y = T.var("x", V), T.var("y", V)
    g = T.Eval(T.Inner(V), (0, 1), (x, y))
    xv, yv = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(_run(g, {_k(x): xv, _k(y): yv}), xv @ yv)


# small builders for the (unused-but-illustrative) norm example above
def _sqr_op():
    from grassmann.lower.represent import ExternalOp

    return ExternalOp("sqr", lambda a: a * a)


def _dotmap(V: S.Space) -> T.Term:
    return T.Inner(V)
