"""Audit Fix 1 — structured bases (``BasisSum``/``BasisProd``/``BasisRep``) with a SEAM
component (FFS / term ``ListBasis``) realize through ``realize_basis_cols`` and round-trip
their cob, instead of crashing in the pure ``realize_basis``.

Covers: a static-rank injective image, a dynamic-rank coimg, a term ``ListBasis``, a mixed
static + FFS sum, and the identity short-circuit; plus the static/dynamic ``BasisRep``-count
behaviour (a runtime-symbolic count raises a clear error, not an obscure crash).
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


def _invertible(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n)), -1) + np.diag(2.0 + np.arange(n))


def _injective_image():
    """An injective ℝ²↪ℝ³ ⇒ a STATIC-rank (dim 2) image."""
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    f = C.var("Finj", C.linear_map_space(Rt, Re))
    J = _invertible(3)[:3, :2]
    return f, {_k(f): J.T}


def _surjective_coimg():
    """A surjective ℝ³↠ℝ² ⇒ a DYNAMIC-rank (dim 2) coimg."""
    Rt, Re = C.base("Rt", 3), C.base("Re", 2)
    f = C.var("Fsurj", C.linear_map_space(Rt, Re))
    J = _invertible(3)[:2, :3]
    return f, {_k(f): J.T}


# --- BasisSum with a seam component ----------------------------------------


def test_basis_sum_with_ffs_image_realizes() -> None:
    """``BasisSum(ProjectedBasis(f,'image'), ScalarBasis())`` — the reported crash — realizes."""
    f, feeds = _injective_image()
    bs = B.basis_sum(B.projected_basis(f, "image"), B.scalar_basis())
    m = _realize(bs, feeds)
    assert m.shape == (3, 3)  # image dim 2 ⊕ scalar 1
    # block-diagonal: image block (in-space identity) ⊕ scalar 1
    assert np.allclose(m, np.eye(3))


def test_basis_sum_identity_cob_short_circuit() -> None:
    """The identity short-circuit ``cob(BasisSum(FFS,…) → same)`` no longer crashes."""
    f, feeds = _injective_image()
    bs = B.basis_sum(B.projected_basis(f, "image"), B.scalar_basis())
    M = _cob(bs, bs, feeds)
    assert M.shape == (3, 3)
    assert np.allclose(M, np.eye(3))


def test_basis_sum_dynamic_rank_component_realizes() -> None:
    """A DYNAMIC-rank FFS component (coimg of a surjection) in a ``BasisSum`` realizes."""
    f, feeds = _surjective_coimg()
    bs = B.basis_sum(B.projected_basis(f, "coimg"), B.scalar_basis())
    m = _realize(bs, feeds)
    assert m.shape == (3, 3)  # coimg dim 2 ⊕ scalar 1 (runtime-sized axis)
    M = _cob(bs, bs, feeds)
    assert np.allclose(M, np.eye(3))


def test_basis_sum_mixed_static_and_term() -> None:
    """A mixed sum: a term ``ListBasis`` (ℝ²) ⊕ a scalar realizes and round-trips."""
    R2 = C.base("R2", 2)
    v0, v1 = C.var("v0", R2), C.var("v1", R2)
    lb = B.list_basis(v0, v1)
    feeds = {_k(v0): np.array([1.0, 1.0]), _k(v1): np.array([0.0, 2.0])}
    bs = B.basis_sum(lb, B.scalar_basis())
    m = _realize(bs, feeds)
    assert m.shape == (3, 3)
    expected = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(m, expected)
    M = _cob(bs, bs, feeds)
    assert np.allclose(M, np.eye(3))


# --- BasisProd with a seam component ---------------------------------------


def test_basis_prod_with_ffs_image_realizes() -> None:
    """``BasisProd`` of two FFS images — Kron of in-space columns."""
    f, feeds = _injective_image()
    bp = B.basis_prod(B.projected_basis(f, "image"), B.projected_basis(f, "image"))
    m = _realize(bp, feeds)
    assert m.shape == (4, 4)  # 2·2 Kron
    assert np.allclose(m, np.eye(4))


# --- BasisRep with a seam component ----------------------------------------


def test_basis_rep_static_count_dynamic_component() -> None:
    """A static-count ``BasisRep`` of a DYNAMIC-rank FFS component realizes (block-repeat)."""
    f, feeds = _surjective_coimg()
    br = B.basis_rep(2, B.projected_basis(f, "coimg"))  # 2 copies of dim-2 coimg ⇒ 4×4
    m = _realize(br, feeds)
    assert m.shape == (4, 4)
    assert np.allclose(m, np.eye(4))
    M = _cob(br, br, feeds)
    assert np.allclose(M, np.eye(4))


def test_basis_rep_static_count_static_component() -> None:
    f, feeds = _injective_image()
    br = B.basis_rep(3, B.scalar_basis())
    m = _realize(br, feeds)
    assert m.shape == (3, 3)
    assert np.allclose(m, np.eye(3))


def test_basis_rep_symbolic_count_clean_error() -> None:
    """A runtime/symbolic ``BasisRep`` count raises a clear error, not an obscure crash."""
    br = B.basis_rep(S.DimVar("m"), B.scalar_basis())
    with pytest.raises(_RM.RepresentError, match="runtime/symbolic count"):
        _realize(br, {})
