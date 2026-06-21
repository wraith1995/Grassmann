"""Audit Fix 5 — ``OrthonormalBasis`` peels ``Dual``/``Primal`` to find the carried metric.

``Orth(Dual(WithInner(b,g)))`` must orthonormalize against the **dual metric** ``G⁻¹`` (its
basis lives in the dual space), not the standard product.  Both ``Orth(Dual(WithInner))`` and
``Dual(Orth(WithInner))`` are accepted spellings and realize ``G⁻¹``-orthonormal columns;
``Orth(WithInner)`` stays ``G``-orthonormal; plain ``Orth(b)`` stays standard-orthonormal.
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa

import grassmann.core as C
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B

_RM = sys.modules["grassmann.lower.represent"]


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _ctx_run(build, feeds):
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", build(ctx))
    return np.asarray(prog.run(feeds).get("M"), float)


def _realize(basis, feeds):
    return _ctx_run(lambda ctx: _RM.realize_basis_cols(ctx, basis), feeds)


def _cob(a, b, feeds):
    return _ctx_run(lambda ctx: _RM._basis_cob_matrix(ctx, a, b), feeds)


def _setup():
    R2 = C.base("R2", 2)
    g = C.var("g", C.linear_map_space(R2, C.dual(R2)))
    b0, b1 = C.var("b0", R2), C.var("b1", R2)
    b = B.list_basis(b0, b1)
    G = np.array([[2.0, 0.5], [0.5, 1.0]])
    # a non-trivial frame so orthonormalization actually rotates the columns
    feeds = {_k(g): G, _k(b0): np.array([1.0, 1.0]), _k(b1): np.array([0.0, 2.0])}
    return b, g, G, feeds


def test_orth_dual_withinner_is_dual_metric_orthonormal() -> None:
    """``Orth(Dual(WithInner(b,g)))`` realizes ``G⁻¹``-orthonormal columns (``QᵀG⁻¹Q ≈ I``)."""
    b, g, G, feeds = _setup()
    o = B.orthonormal_basis(B.dual_basis(B.with_inner_basis(b, g)))
    Q = _realize(o, feeds)
    Ginv = np.linalg.inv(G)
    assert np.allclose(Q.T @ Ginv @ Q, np.eye(2))
    # explicitly NOT standard-orthonormal (the bug would give QᵀQ = I instead)
    assert not np.allclose(Q.T @ Q, np.eye(2))


def test_dual_orth_withinner_is_dual_metric_orthonormal() -> None:
    """``Dual(Orth(WithInner(b,g)))`` — the dual of the g-orthonormal basis — is ``G⁻¹``-orthonormal."""
    b, g, G, feeds = _setup()
    o = B.dual_basis(B.orthonormal_basis(B.with_inner_basis(b, g)))
    Q = _realize(o, feeds)
    Ginv = np.linalg.inv(G)
    assert np.allclose(Q.T @ Ginv @ Q, np.eye(2))


def test_orth_withinner_is_g_orthonormal() -> None:
    """``Orth(WithInner(b,g))`` (even parity) stays ``G``-orthonormal (``QᵀGQ ≈ I``)."""
    b, g, G, feeds = _setup()
    o = B.orthonormal_basis(B.with_inner_basis(b, g))
    Q = _realize(o, feeds)
    assert np.allclose(Q.T @ G @ Q, np.eye(2))


def test_plain_orth_is_standard_orthonormal() -> None:
    """Plain ``Orth(b)`` (no metric anywhere) stays standard-orthonormal (``QᵀQ ≈ I``)."""
    b, g, G, feeds = _setup()
    o = B.orthonormal_basis(b)
    Q = _realize(o, feeds)
    assert np.allclose(Q.T @ Q, np.eye(2))
    assert _RM._is_standard_orthonormal(o)
    assert not _RM._is_standard_orthonormal(B.orthonormal_basis(B.with_inner_basis(b, g)))
    assert not _RM._is_standard_orthonormal(
        B.orthonormal_basis(B.dual_basis(B.with_inner_basis(b, g)))
    )


def test_orth_dual_cob_round_trips() -> None:
    """``Orth(Dual(WithInner))``'s cob round-trips (identity short-circuit)."""
    b, g, G, feeds = _setup()
    o = B.orthonormal_basis(B.dual_basis(B.with_inner_basis(b, g)))
    m = _cob(o, o, feeds)
    assert np.allclose(m, np.eye(2))


def test_cob_between_dual_metric_spellings_round_trips() -> None:
    """A cob between the two ``G⁻¹``-orthonormal spellings round-trips (same dual space)."""
    b, g, G, feeds = _setup()
    o1 = B.orthonormal_basis(B.dual_basis(B.with_inner_basis(b, g)))
    o2 = B.dual_basis(B.orthonormal_basis(B.with_inner_basis(b, g)))
    m = _cob(o1, o2, feeds)
    back = _cob(o2, o1, feeds)
    assert np.allclose(m @ back, np.eye(2))
