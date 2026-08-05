"""Change-of-basis regressions: three correctness fixes (1/2/3) and three sub-optimality
fixes (4/5/6).

These paths were previously untested:

* **1** — the abstract-basis reject was TOP-node-only, so a structured/wrapped basis with a
  NESTED ``BVar`` leaf that is ``_basis_eq``-equal slipped to the identity short-circuit and
  failed with a confusing ``CobError`` instead of the clear "abstract basis" ``RepresentError``.
* **2** — ``Orth(Orth(WithInner(b,g)))`` dropped the metric (the peel stopped at the inner
  ``OrthonormalBasis``), orthonormalizing in the STANDARD product instead of ``g``.
* **3** — ``CompleteBasis(reduce=NONE)``'s over-complete guard was statically blind to a
  dynamic-column ``ProjectedBasis`` sub ⇒ a silent rectangular non-basis; a runtime assert now
  catches it.
* **4** — ``_is_standard_orthonormal`` didn't recurse ``BasisSum``/``BasisProd``/``BasisRep``/
  ``Dual``/``Primal`` of orthonormal components (so ``default_basis(ℝⁿ)``, a ``BasisRep``, was
  unrecognized and missed the cheap transpose).
* **5** — ``_norm_space`` only expanded an ``OPlusRep`` for a ``DimLit`` count, so a foldable
  non-literal count (``DimSum(1,1)``) wrongly failed the same-space guard.
* **6** — ``_basis_eq`` ignored the ``WithInnerBasis`` metric (a latent wrong-identity via
  the public ``cob()``).
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.core.space as S
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B
from grassmann.lower.space_basis import default_basis

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


def _roundtrips(a, b, feeds) -> bool:
    ab = _cob(a, b, feeds)
    ba = _cob(b, a, feeds)
    return np.allclose(ab @ ba, np.eye(ab.shape[0])) and np.allclose(ba @ ab, np.eye(ab.shape[0]))


def _invertible(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n)), -1) + np.diag(2.0 + np.arange(n))


# ===========================================================================
# 1 — nested abstract leaf is rejected with the clear "abstract basis" error
# ===========================================================================


@pytest.mark.parametrize(
    "make",
    [
        lambda: B.basis_sum(B.bvar("x"), B.bvar("x")),
        lambda: B.dual_basis(B.bvar("x")),
        lambda: B.orthonormal_basis(B.bvar("x")),
        lambda: B.basis_rep(2, B.basis_sum(B.bvar("x"), B.bvar("x"))),
        lambda: B.primal_basis(B.bvar("x")),
    ],
)
def test_1_nested_abstract_identity_cob_raises_clear_error(make) -> None:
    """An identity cob of a structured/wrapped basis with a nested ``BVar`` must raise the clear
    ``RepresentError`` ("abstract"), NOT a confusing downstream ``CobError``."""
    b = make()
    with pytest.raises(_RM.RepresentError, match="abstract"):
        _ctx_run(lambda ctx: _RM._basis_cob_rec(ctx, b, b), {})


def test_1_contains_abstract_predicate() -> None:
    assert _RM._contains_abstract(B.bvar("x"))
    assert _RM._contains_abstract(B.basis_sum(B.scalar_basis(), B.bvar("x")))
    assert _RM._contains_abstract(B.with_inner_basis(B.bvar("x"), C.var("g", C.linear_map_space(
        C.base("R2", 2), C.dual(C.base("R2", 2))))))
    # a fully-concrete structured basis is NOT abstract
    assert not _RM._contains_abstract(B.basis_rep(2, B.scalar_basis()))


# ===========================================================================
# 2 — Orth(Orth(WithInner(b,g))) stays g-orthonormal (idempotent), not standard
# ===========================================================================


def _withinner_setup():
    R2 = C.base("R2", 2)
    g = C.var("g", C.linear_map_space(R2, C.dual(R2)))
    b0, b1 = C.var("b0", R2), C.var("b1", R2)
    b = B.list_basis(b0, b1)
    G = np.array([[2.0, 0.5], [0.5, 1.0]])
    feeds = {_k(g): G, _k(b0): np.array([1.0, 1.0]), _k(b1): np.array([0.0, 2.0])}
    return b, g, G, feeds


def test_2_double_orth_withinner_is_g_orthonormal() -> None:
    b, g, G, feeds = _withinner_setup()
    oo = B.orthonormal_basis(B.orthonormal_basis(B.with_inner_basis(b, g)))
    Q = _realize(oo, feeds)
    assert np.allclose(Q.T @ G @ Q, np.eye(2)), "must be g-orthonormal (QᵀGQ=I), not standard"


def test_2_double_orth_not_standard_orthonormal_and_metric_consistent() -> None:
    b, g, G, feeds = _withinner_setup()
    single = B.orthonormal_basis(B.with_inner_basis(b, g))
    double = B.orthonormal_basis(single)
    # classified consistently with the single Orth(WithInner): both g-orthonormal, NOT standard.
    assert _RM._is_standard_orthonormal(double) is False
    assert _RM._is_standard_orthonormal(single) is False
    gm, parity = _RM._peel_orthonormal_metric(single)
    assert gm is not None and parity == 0


def test_2_double_orth_cob_roundtrips() -> None:
    b, g, _G, feeds = _withinner_setup()
    oo = B.orthonormal_basis(B.orthonormal_basis(B.with_inner_basis(b, g)))
    std = B.with_inner_basis(default_basis(C.base("R2", 2)), g)
    assert _roundtrips(oo, std, feeds)


# ===========================================================================
# 3 — CompleteBasis(reduce=NONE) over a dynamic-column ProjectedBasis sub
# ===========================================================================


def test_3_complete_none_overcomplete_projected_sub_runtime_error() -> None:
    """sub=image (dim 2 of ℝ³) + the full ℝ³ standard complement = 5 cols vs 3 rows ⇒ NONE is an
    over-complete rectangular non-basis; the dynamic-column sub defeats the static guard, so a
    RUNTIME assert must fire."""
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    f = C.var("Finj", C.linear_map_space(Rt, Re))
    feeds = {_k(f): (_invertible(3)[:3, :2]).T}
    sub = B.projected_basis(f, "image")
    cb = B.complete_basis(sub, default_basis(Re), reduce=B.Reduce.NONE)
    with pytest.raises(AssertionError, match="ambient dimension"):
        _cob(cb, default_basis(Re), feeds)


def test_3_complete_none_disjoint_projected_pair_roundtrips() -> None:
    """sub=image (dim 2) + complement=coker (dim 1), disjoint and spanning exactly ℝ³ ⇒ NONE is a
    genuine square basis whose realize is square-invertible and whose cob round-trips both-sided
    against the RREF completion of the same (sub, complement) pair."""
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    f = C.var("Finj", C.linear_map_space(Rt, Re))
    feeds = {_k(f): (_invertible(3)[:3, :2]).T}
    sub = B.projected_basis(f, "image")        # dim 2
    comp = B.projected_basis(f, "coker")       # dim 1, orthogonal complement of the image
    none = B.complete_basis(sub, comp, reduce=B.Reduce.NONE)
    rref = B.complete_basis(sub, comp, reduce=B.Reduce.RREF)
    m = _realize(none, feeds)
    assert m.shape == (3, 3) and abs(np.linalg.det(m)) > 1e-9
    assert _roundtrips(none, rref, feeds)


def test_3_complete_none_static_overcomplete_errors() -> None:
    """A STATICALLY over-complete NONE (two full ℝ² bases hstacked = 4 cols vs 2 rows) raises at
    lowering with the clear RepresentError."""
    R2 = C.base("R2", 2)
    cb = B.complete_basis(default_basis(R2), default_basis(R2), reduce=B.Reduce.NONE)
    with pytest.raises(_RM.RepresentError, match="NONE"):
        _cob(cb, default_basis(R2), {})


# ===========================================================================
# 4 — _is_standard_orthonormal recurses structured / variance wrappers
# ===========================================================================


def test_4_basis_rep_of_scalar_is_standard_orthonormal() -> None:
    assert _RM._is_standard_orthonormal(B.basis_rep(3, B.scalar_basis())) is True
    assert _RM._is_standard_orthonormal(default_basis(C.base("R3", 3))) is True
    # structured / variance recursion
    assert _RM._is_standard_orthonormal(
        B.basis_sum(B.scalar_basis(), B.basis_rep(2, B.scalar_basis()))) is True
    assert _RM._is_standard_orthonormal(B.dual_basis(default_basis(C.base("R3", 3)))) is True
    # NOT standard-orthonormal: a g-orthonormal WithInner-carrying Orth, and a plain ListBasis
    R2 = C.base("R2", 2)
    g = C.var("g", C.linear_map_space(R2, C.dual(R2)))
    assert _RM._is_standard_orthonormal(
        B.orthonormal_basis(B.with_inner_basis(default_basis(R2), g))) is False
    assert _RM._is_standard_orthonormal(
        B.list_basis(C.var("v0", R2), C.var("v1", R2))) is False


def test_4_cob_to_default_basis_uses_transpose_not_inv() -> None:
    """``cob(std → default_basis(ℝⁿ))`` — ``to`` is a ``BasisRep`` (of ``ScalarBasis``), now
    recognized standard-orthonormal ⇒ ``_inverse_cols`` takes the cheap TRANSPOSE branch instead
    of an ``InvOp``/``PinvOp``, and the result is still numerically correct."""
    R3 = C.base("R3", 3)
    std = default_basis(R3)
    v = [C.var(f"u{i}", R3) for i in range(3)]
    M = _invertible(3)
    feeds = {_k(v[i]): M[:, i] for i in range(3)}
    lb = B.list_basis(*v)

    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", _RM._cob_via_standard(ctx, lb, std))  # to = default_basis (BasisRep)
    kinds = {type(s.fn).__name__ for s in prog.statements}
    assert "TransposeOp" in kinds
    assert not (kinds & {"InvOp", "PinvOp"}), "default_basis to-leg must not need Inv/Pinv"
    out = np.asarray(prog.run(feeds).get("M"), float)
    assert np.allclose(out, M), "cob(list → default_basis) == realize(list) (the columns)"

    # Round-trip via the public matrix path too (dual side exercises the variance-recursion arm).
    a = B.dual_basis(std)
    other = B.dual_basis(lb)
    assert _roundtrips(a, other, feeds)


# ===========================================================================
# 5 — _norm_space static-folds a non-literal OPlusRep count
# ===========================================================================


def test_5_basis_rep_dimsum_count_same_space_roundtrips() -> None:
    """``cob(BasisRep(DimLit 2, b) → BasisRep(DimSum(1,1), b))`` — the two ⊕² spaces are equal
    once the foldable count is expanded; previously rejected as different spaces."""
    R1 = C.base("R1", 1)
    v = C.var("v", R1)
    b = B.list_basis(v)
    a = B.basis_rep(S.DimLit(2), b)
    bb = B.basis_rep(S.DimSum((S.DimLit(1), S.DimLit(1))), b)
    feeds = {_k(v): np.array([3.0])}
    # must not raise "different spaces" and must round-trip
    assert _roundtrips(a, bb, feeds)


def test_5_norm_space_folds_dimsum_oplusrep() -> None:
    rep = S.OPlusRep(S.DimSum((S.DimLit(1), S.DimLit(1))), S.Scalar())
    folded = _RM._norm_space(rep)
    assert isinstance(folded, S.OPlus) and len(folded.spaces) == 2


# ===========================================================================
# 6 — _basis_eq compares the WithInnerBasis metric (latent wrong-identity)
# ===========================================================================


def test_6_basis_eq_distinguishes_withinner_metric() -> None:
    R2 = C.base("R2", 2)
    g1 = C.var("g1", C.linear_map_space(R2, C.dual(R2)))
    g2 = C.var("g2", C.linear_map_space(R2, C.dual(R2)))
    b = B.list_basis(C.var("a", R2), C.var("b", R2))
    assert _RM._basis_eq(B.with_inner_basis(b, g1), B.with_inner_basis(b, g2)) is False
    assert _RM._basis_eq(B.with_inner_basis(b, g1), B.with_inner_basis(b, g1)) is True
