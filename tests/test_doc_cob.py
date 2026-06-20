"""Doc-tied tests: every **change-of-basis (cob) scenario** the docs (`docs/bases/`) describe.

One test (or family) per basis node, asserting the realization + conversion behavior its
`docs/bases/*.md` page states: how the basis realizes to ambient coordinate columns, and how
`cob` / `_basis_cob_matrix` converts coordinates between bases of the same space.  `cob` is
**total** — it goes between any two bases of the same space (the doc claim verified here).

The closed-form ground truth for `Img(J)` bases: `ImageBasis` columns are the literal `J`;
`ProjectedBasis("image")` columns are the `Q` of the QR; the correct cob from coords in basis 1
to coords in basis 2 is `pinv(C₂)·C₁` (a dual pair adds `(·)⁻ᵀ`; a product is a Kronecker).
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.lower._dims import basis_dim
from grassmann.lower.cob import _basis_eq
from grassmann.lower.space_basis import default_basis

_RM = sys.modules["grassmann.lower.represent"]

JMATH = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])  # injective 3x2
JSTORED = JMATH.T
Q, R = np.linalg.qr(JMATH)


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _ctx() -> object:
    return _RM._Ctx(prog=pa.Program("t"), basis_env={}, inputs={})


def _realize(basis, feeds=None) -> np.ndarray:
    ctx = _ctx()
    cols = _RM.realize_basis_cols(ctx, basis)
    ctx.prog.add_output("C", cols)
    return np.asarray(ctx.prog.run(feeds or {})["C"], float)


def _cob(b_from, b_to, feeds=None) -> np.ndarray:
    ctx = _ctx()
    m = _RM._basis_cob_matrix(ctx, b_from, b_to)
    ctx.prog.add_output("M", m)
    return np.asarray(ctx.prog.run(feeds or {})["M"], float)


def _img_setup():
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    return J, B.image_basis(J), B.projected_basis(J, "image")


# ===========================================================================
# Realization: each basis → its ambient coordinate columns.
# ===========================================================================


def test_scalar_basis_realizes_to_1x1_identity() -> None:
    """`ScalarBasis` — the canonical basis of ℝ, the 1×1 identity."""
    assert _realize(B.scalar_basis()) == pytest.approx(np.eye(1))


def test_standard_base_basis_realizes_to_identity() -> None:
    """`BaseBasis` (standard, self-sizing) — realizes to the ambient identity."""
    assert _realize(default_basis(C.base("V", 3))) == pytest.approx(np.eye(3))


def test_explicit_base_basis_realizes_to_its_columns() -> None:
    """`BaseBasis` with explicit numeric `vectors` — realizes to that matrix."""
    M = np.array([[2.0, 1.0], [0.0, 3.0]])
    assert _realize(B.base_basis("V", M)) == pytest.approx(M)


def test_base_basis_term_frame_realizes_stacked() -> None:
    """`BaseBasis` term frame — Term vectors realized and stacked as columns."""
    W = C.base("W", 3)
    a, b = C.var("a", W), C.var("b", W)
    frame = B.base_basis("W", vectors=(a, b), space=W)
    av, bv = np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])
    assert _realize(frame, {_k(a): av, _k(b): bv}) == pytest.approx(np.column_stack([av, bv]))


def test_base_basis_backup_completes_frame() -> None:
    """`BaseBasis` backup — a rank-deficient frame completed to a full basis."""
    W = C.base("W", 3)
    c = C.var("c", W)
    basis = B.base_basis("W", vectors=(c,), space=W, backup=B.base_basis("W", space=W))
    cols = _realize(basis, {_k(c): np.array([1.0, 2.0, 3.0])})
    assert cols.shape == (3, 3)
    assert np.linalg.matrix_rank(cols) == 3
    assert cols[:, 0] == pytest.approx([1.0, 2.0, 3.0])


def test_image_basis_realizes_to_literal_columns() -> None:
    """`ImageBasis` — the literal map-application columns `f(eᵢ)` = `J`."""
    J, imgb, _ = _img_setup()
    assert _realize(imgb, {_k(J): JSTORED}) == pytest.approx(JMATH)


def test_projected_basis_realizes_to_Q() -> None:
    """`ProjectedBasis("image")` — the orthonormal range `Q` of the QR."""
    J, _, projb = _img_setup()
    got = _realize(projb, {_k(J): JSTORED})
    assert got.T @ got == pytest.approx(np.eye(2))  # orthonormal
    assert np.abs(got) == pytest.approx(np.abs(Q))  # = Q up to column sign


def test_list_basis_realizes_stacked() -> None:
    """`ListBasis` — explicit Term vectors stacked as columns."""
    W = C.base("W", 3)
    a, b = C.var("a", W), C.var("b", W)
    av, bv = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert _realize(B.list_basis(a, b), {_k(a): av, _k(b): bv}) == pytest.approx(
        np.column_stack([av, bv])
    )


# ===========================================================================
# Sizing: basis_dim per the docs.
# ===========================================================================


def test_scalar_basis_dim() -> None:
    assert basis_dim(B.scalar_basis()) == 1


def test_self_sizing_base_basis_dim() -> None:
    """`BaseBasis` with a `space` is self-sizing (dim from the space)."""
    assert basis_dim(default_basis(C.base("V", 5))) == 5


def test_dual_basis_dim_matches_primal() -> None:
    assert basis_dim(B.dual_basis(default_basis(C.base("V", 4)))) == 4


def test_basis_sum_dim_is_sum() -> None:
    db = default_basis(C.oplus(C.base("V", 2), C.base("W", 3)))
    assert isinstance(db, B.BasisSum)
    assert basis_dim(db) == 5


def test_basis_prod_dim_is_product() -> None:
    db = default_basis(C.oprod(C.base("V", 2), C.base("W", 3)))
    assert isinstance(db, B.BasisProd)
    assert basis_dim(db) == 6


def test_basis_rep_dim_is_count_times() -> None:
    db = default_basis(C.oplus_rep(4, C.base("V", 2)))
    assert isinstance(db, B.BasisRep)
    assert basis_dim(db) == 8


# ===========================================================================
# Conversion (cob) — totality between any two bases of the same space.
# ===========================================================================


def test_cob_identical_basis_is_identity() -> None:
    """`cob(b, b)` short-circuits to the identity."""
    std = default_basis(C.base("V", 3))
    assert _cob(std, std) == pytest.approx(np.eye(3))


def test_cob_standard_to_explicit_is_inverse() -> None:
    """`cob(standard → rotated) = ROT⁻¹` (a self-sizing standard basis, no companion)."""
    ROT = np.array([[2.0, 1.0], [0.0, 3.0]])
    std = default_basis(C.base("V", 2))
    assert _cob(std, B.base_basis("V", ROT)) == pytest.approx(np.linalg.inv(ROT))


def test_cob_image_to_projected_is_R() -> None:
    """`cob(ImageBasis → ProjectedBasis) = pinv(Q)·J = R`."""
    J, imgb, projb = _img_setup()
    assert _cob(imgb, projb, {_k(J): JSTORED}) == pytest.approx(R)


def test_cob_projected_to_image_is_R_inv() -> None:
    """`cob(ProjectedBasis → ImageBasis) = R⁻¹` (the inverse direction)."""
    J, imgb, projb = _img_setup()
    assert _cob(projb, imgb, {_k(J): JSTORED}) == pytest.approx(np.linalg.inv(R))


def test_cob_ffs_pair_is_inverse() -> None:
    """The two FFS cobs compose to the identity (a genuine inverse pair)."""
    J, imgb, projb = _img_setup()
    f = {_k(J): JSTORED}
    assert _cob(imgb, projb, f) @ _cob(projb, imgb, f) == pytest.approx(np.eye(2))


def test_cob_dual_ffs_is_inverse_transpose() -> None:
    """`DualBasis` cob — `cob(dual b → dual b') = (cob(b→b'))⁻ᵀ`."""
    J, imgb, projb = _img_setup()
    f = {_k(J): JSTORED}
    primal = _cob(imgb, projb, f)
    dual = _cob(B.dual_basis(imgb), B.dual_basis(projb), f)
    assert dual == pytest.approx(np.linalg.inv(primal).T)


def test_cob_basis_prod_is_kron() -> None:
    """`BasisProd` cob — Kronecker product of the per-factor cobs (FFS leaf included)."""
    J, imgb, projb = _img_setup()
    base2 = B.base_basis("Re2", np.eye(2))
    from_b = B.basis_prod(B.dual_basis(imgb), base2)
    to_b = B.basis_prod(B.dual_basis(projb), base2)
    got = _cob(from_b, to_b, {_k(J): JSTORED})
    assert got == pytest.approx(np.kron(np.linalg.inv(R).T, np.eye(2)))


def test_cob_basis_sum_is_block_diagonal() -> None:
    """`BasisSum` cob — block-diagonal of the per-summand cobs."""
    A = np.array([[2.0, 0.0], [1.0, 1.0]])
    Bm = np.array([[3.0]])
    from_b = B.basis_sum(B.base_basis("V", A), B.base_basis("W", Bm))
    to_b = B.basis_sum(default_basis(C.base("V", 2)), default_basis(C.base("W", 1)))
    # cob to the standard basis on each block = the block's own columns; block-diagonal.
    got = _cob(from_b, to_b)
    expect = np.zeros((3, 3))
    expect[:2, :2] = A
    expect[2:, 2:] = Bm
    assert got == pytest.approx(expect)


def test_cob_applied_changes_coordinates() -> None:
    """The cob matrix actually maps ImageBasis coords `s` to ProjectedBasis coords `R·s`."""
    J, imgb, projb = _img_setup()
    M = _cob(imgb, projb, {_k(J): JSTORED})
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.standard_normal(2)
        assert M @ s == pytest.approx(R @ s)


def test_with_inner_basis_eq_strips_metric() -> None:
    """`WithInnerBasis` — the metric is metadata; cob strips it (basis identity unchanged)."""
    base = default_basis(C.base("V", 3))
    g = C.var("g", C.oprod(C.dual(C.base("V", 3)), C.dual(C.base("V", 3))))
    assert _basis_eq(B.with_inner_basis(base, g), base) or _cob(
        B.with_inner_basis(base, g), base
    ) == pytest.approx(np.eye(3))


def test_complete_basis_realizes_sub_then_complement() -> None:
    """`CompleteBasis` — realizes to `[ realize(sub) | complement ]` stacked."""
    sub = B.base_basis("V", np.array([[1.0], [0.0], [0.0]]))   # a 1-col subspace in R^3
    comp = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])      # the complement columns
    got = _realize(B.complete_basis(sub, comp))
    assert got == pytest.approx(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_app_space_has_no_default_basis() -> None:
    """A space-level binder/`BApp`/`BVar` only become concrete at instantiation — an
    unreduced space application has no default basis (mirrors the `App` space)."""
    from grassmann.core.kinds import SpaceSort
    from grassmann.lower.space_basis import space_basis

    sf = C.lam((("a", SpaceSort(C.base("A", 2))),), C.oprod(C.svar("a"), C.svar("a")))
    with pytest.raises(S.SpaceError):
        space_basis(C.space_app(sf, (0,), C.base("A", 2)))
