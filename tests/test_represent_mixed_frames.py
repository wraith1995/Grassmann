"""Regression tests for cob'ing a MULTI-AXIS operand between frames.

Two lowering paths assumed one array axis per basis factor, which is false for any
⊗-space operand:

* ``_coerce_basis`` (used by ``_rep_sum`` to reconcile summands, among others) emitted a
  rank-1 ``"ij,j->i"`` cob regardless of the operand's rank, so a ``Sum`` over a ⊗-space
  whose summands realized in different bases died in numpy;
* ``_coerce_rep_to_target`` walked ``BasisProd`` factor ``i`` against array axis ``i``,
  but an ``LLam`` tags its result with the body's codomain as ONE nested factor spanning
  several axes, so a factor-sized cob landed on a single (wrongly sized) axis.

Both surfaced only at ``run`` time: polyarray's compile-time ``fold_numeric`` executes and
swallows a failing statement, so ``compile_in`` succeeded and the crash came later. Every
test here therefore RUNS the program — compiling is not evidence.

Found by ``gfuzz.frames``; ``gfuzz.frames_matrix`` is the full (term × frame) grid.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.stdlib.exterior import alt


def _e(i: int, s: float = 1.0, n: int = 3) -> G.Expr:
    """Build the ``i``-th standard basis vector of ``ℝⁿ``, scaled by ``s``."""
    return G.ostack(*[G.lit(s if j == i else 0.0) for j in range(n)])


@pytest.fixture
def bases() -> tuple[G.Expr, G.Expr]:
    """Give two list bases of ``ℝ³``: the identity one, and one with e₀ scaled by 2."""
    ident = G.list_basis(_e(0), _e(1), _e(2))
    scaled = G.list_basis(_e(0, 2.0), _e(1), _e(2))
    return ident, scaled


def _run(prog: object, **feed: np.ndarray) -> np.ndarray:
    return np.asarray(prog.run(feed)["result"], float)


# ---------------------------------------------------------------------------
# Cluster A — a Sum over a ⊗-space whose summands sit in different bases.
# ---------------------------------------------------------------------------


def test_sum_over_tensor_space_reconciles_differing_frames(bases) -> None:
    """A ``Sum`` on ``V⊗V`` must cob its summands per axis, not with a rank-1 einsum."""
    ident, scaled = bases
    V = G.space("R3", 3)
    T = G.tensor(V, V)
    a, b = G.var("a", T), G.var("b", T)
    prog = G.compile_in(a + b, {a: G.tensor(ident, ident), b: G.tensor(scaled, scaled)})
    av = np.arange(9.0).reshape(3, 3)
    bv = np.arange(9.0).reshape(3, 3)[::-1]
    out = _run(prog, **{f"a_{a.node.uid}": av, f"b_{b.node.uid}": bv})
    assert out.shape == (3, 3)
    assert np.isfinite(out).all()
    # `b`'s frame scales e₀ by 2, so b's coordinates mean 2x along both first axes;
    # reconciled into a's frame the sum is a + S·b·Sᵀ with S = diag(2,1,1).
    S = np.diag([2.0, 1.0, 1.0])
    assert np.allclose(out, av + S @ bv @ S.T)


def test_sum_over_tensor_space_with_mixed_factor_frame(bases) -> None:
    """The same, when one summand's frame uses a different basis in each ⊗ factor."""
    ident, scaled = bases
    V = G.space("R3", 3)
    T = G.tensor(V, V)
    a, b = G.var("a", T), G.var("b", T)
    prog = G.compile_in(a + b, {a: G.tensor(ident, ident), b: G.tensor(ident, scaled)})
    av = np.zeros((3, 3))
    bv = np.eye(3)
    out = _run(prog, **{f"a_{a.node.uid}": av, f"b_{b.node.uid}": bv})
    # `a` contributes nothing, so the answer is `b` reconciled into a's frame: b's frame is
    # the identity one in factor 0 and the e₀-scaled one in factor 1.
    S0, S1 = np.eye(3), np.diag([2.0, 1.0, 1.0])
    assert np.allclose(out, S0 @ bv @ S1.T)


# ---------------------------------------------------------------------------
# Cluster B — an LLam/Lift realized in a frame that is not its natural tag.
# ---------------------------------------------------------------------------


def _alt_matrix(V: G.Expr, k: int, dom: G.Expr, cod: G.Expr) -> np.ndarray:
    """Represent ``alt(V, k)`` as a matrix with the given domain / codomain frames."""
    return np.asarray(
        G.compile_in(alt(V, k), target=G.tensor(G.dual(dom), cod)).run({})["result"], float
    )


@pytest.mark.parametrize("k", [2, 3])
def test_alt_matrix_is_basis_independent(k: int, bases) -> None:
    """``Alt``'s matrix is the same in any basis of ``V`` used in every tensor factor.

    ``P_σ`` commutes with ``S ⊗ … ⊗ S`` for every invertible ``S``, so a consistent change
    of basis cancels itself. This held before the fix and must keep holding.
    """
    ident, scaled = bases
    V = G.space("R3", 3)
    std = G.standard_basis(V)
    ref = _alt_matrix(V, k, G.tensor(*[std] * k), G.tensor(*[std] * k))
    for b in (ident, scaled):
        prod = G.tensor(*[b] * k)
        assert np.allclose(_alt_matrix(V, k, prod, prod), ref)


def test_alt_matrix_with_differing_domain_and_codomain_frames(bases) -> None:
    """``Alt`` realized with one frame on the domain and another on the codomain."""
    ident, scaled = bases
    V = G.space("R3", 3)
    ref = _alt_matrix(V, 2, G.tensor(ident, ident), G.tensor(ident, ident))
    got = _alt_matrix(V, 2, G.tensor(ident, ident), G.tensor(scaled, scaled))
    assert got.shape == ref.shape
    assert np.isfinite(got).all()
    # The codomain frame scales e₀ by 2 in both factors, so the codomain coordinates of
    # every column are divided by the corresponding diag(2,1,1)⊗diag(2,1,1) entry.
    d = np.kron(np.array([2.0, 1.0, 1.0]), np.array([2.0, 1.0, 1.0]))
    assert np.allclose(got, ref / d[:, None])


def test_alt_matrix_with_mixed_factor_frame(bases) -> None:
    """``Alt`` realized in a frame whose two ⊗ factors use different bases.

    Unlike the consistent case, the array here is NOT the standard one: ``P_σ`` does not
    commute with ``S ⊗ T`` for ``S ≠ T``, so the represented matrix stops being symmetric.
    """
    ident, scaled = bases
    V = G.space("R3", 3)
    ref = _alt_matrix(V, 2, G.tensor(ident, ident), G.tensor(ident, ident))
    mixed = G.tensor(ident, scaled)
    got = _alt_matrix(V, 2, mixed, mixed)
    assert got.shape == (9, 9)
    assert np.isfinite(got).all()
    assert np.allclose(ref, ref.T)          # the consistent representation is symmetric
    assert not np.allclose(got, got.T)      # the intermixed one is not
    # A map is stored (dom, cod) — the transpose of the usual matrix (see
    # ``tests/test_represent.py``) — so scaling the frame of ⊗-factor 1 on both sides
    # multiplies along the domain axis and divides along the codomain axis.
    d = np.kron(np.array([1.0, 1.0, 1.0]), np.array([2.0, 1.0, 1.0]))
    assert np.allclose(got, (ref * d[:, None]) / d[None, :])


def test_applied_alt_in_a_list_basis_frame(bases) -> None:
    """``alt(V,2)(x)`` realized in a list basis — an ``LLam`` under an ``Eval``."""
    ident, _scaled = bases
    V = G.space("R3", 3)
    T = G.tensor(V, V)
    x = G.var("x", T)
    frame = G.tensor(ident, ident)
    prog = G.compile_in(alt(V, 2)(x), {x: G.standard_basis(T)}, target=frame)
    xv = np.arange(9.0).reshape(3, 3)
    out = _run(prog, **{f"x_{x.node.uid}": xv})
    # The identity list basis is numerically the standard one, so this is just Alt(x).
    assert np.allclose(out, 0.5 * (xv - xv.T))
