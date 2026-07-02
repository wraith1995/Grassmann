"""Determinant ``det`` and log-determinant ``logdet`` of a square linear map.

Covers all four engine layers for the new ``Det``/``LogDet`` nodes:

* build + type/space synthesis (``Scalar``; non-square raises),
* numeric lowering vs ``numpy.linalg.det`` / ``slogdet``,
* the matrix-calculus derivatives ``d det(X)[dX] = det(X)·tr(X⁻¹·dX)`` and
  ``d logdet(X)[dX] = tr(X⁻¹·dX)`` vs a central finite difference,
* the matrixcalculus gold composition ``log(det(inv(X)))``.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
import grassmann.stdlib as L
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.infer.synth import InferError, synth_space
from grassmann.lower import represent
from grassmann.transform.diff import diff

V = S.base("V", 3)
DV = S.dual(V)
MS = S.oprod(DV, V)  # V ⊸ V, stored (dom, cod)

_A = np.array([[3.0, 0.0, 1.0], [1.0, 4.0, 0.0], [0.0, 1.0, 3.0]])  # det = 37 > 0


def _key(v: T.Var) -> str:
    return f"{v.name}_{v.uid}"


def _run(term: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(represent(term).run(feed)["result"], float)


# --- build + synthesis ------------------------------------------------------


def test_det_logdet_synth_scalar():
    X = T.var("X", MS, props=G.bij)
    assert synth_space(T.det(X)) == S.Scalar()
    assert synth_space(T.logdet(X)) == S.Scalar()
    # surface builders agree with the core constructors
    Xe = G.var("X", MS, props=G.bij)
    assert G.det(Xe).node == T.Det(Xe.node)
    assert G.logdet(Xe).node == T.LogDet(Xe.node)


def test_det_requires_square():
    W = S.base("W", 2)
    f = T.var("f", S.oprod(DV, W))  # V ⊸ W, not square
    with pytest.raises(InferError):
        synth_space(T.det(f))
    with pytest.raises(InferError):
        synth_space(T.logdet(f))


# --- numeric forward value --------------------------------------------------


def test_det_numeric():
    X = T.var("X", MS, props=G.bij)
    got = float(_run(T.det(X), {_key(X): _A}))
    assert got == pytest.approx(np.linalg.det(_A))


def test_logdet_numeric():
    X = T.var("X", MS, props=G.bij)
    got = float(_run(T.logdet(X), {_key(X): _A}))
    sign, logabs = np.linalg.slogdet(_A)
    assert sign > 0
    assert got == pytest.approx(logabs)
    assert got == pytest.approx(np.log(np.linalg.det(_A)))


# --- derivatives vs central finite difference -------------------------------


def _jacobian_dot_h(term: T.Term, X: T.Var, A: np.ndarray, H: np.ndarray) -> float:
    """``J·H`` — seed the Jacobian ``LLam`` body directly (the directional-derivative value)."""
    J = diff(term, X)
    seed = J.vars[0]
    return float(_run(J.body, {_key(X): A, f"{seed.name}_{seed.uid}": H}))


@pytest.mark.parametrize(
    ("build", "ref"),
    [
        (T.det, np.linalg.det),
        (T.logdet, lambda M: np.linalg.slogdet(M)[1]),
    ],
)
def test_det_logdet_derivative_matches_fd(build, ref):
    X = T.var("X", MS, props=G.bij)
    rng = np.random.default_rng(20260626)
    # A few random well-conditioned directions about the fixed (det>0) base point.
    for _ in range(4):
        H = rng.standard_normal((3, 3))
        got = _jacobian_dot_h(build(X), X, _A, H)
        eps = 1e-6
        fd = (ref(_A + eps * H) - ref(_A - eps * H)) / (2.0 * eps)
        assert got == pytest.approx(fd, abs=1e-6)


# --- matrixcalculus gold composition: log(det(inv(X))) ----------------------


def test_logdet_of_inverse_composition():
    """``log(det(inv(X)))`` — the gold example this op unblocks; build + lower + diff."""
    X = T.var("X", MS, props=G.bij)
    e = L.log(G.det(G.solve(G.wrap(X)))).node  # log(det(X⁻¹))
    assert synth_space(e) == S.Scalar()

    got = float(_run(e, {_key(X): _A}))
    assert got == pytest.approx(np.log(np.linalg.det(np.linalg.inv(_A))))

    # derivative: d log(det(X⁻¹)) = −tr(X⁻¹·dX)
    rng = np.random.default_rng(7)
    H = rng.standard_normal((3, 3))
    got_d = _jacobian_dot_h(e, X, _A, H)
    ref = lambda M: np.log(np.linalg.det(np.linalg.inv(M)))  # noqa: E731
    fd = (ref(_A + 1e-6 * H) - ref(_A - 1e-6 * H)) / (2e-6)
    assert got_d == pytest.approx(fd, abs=1e-6)
