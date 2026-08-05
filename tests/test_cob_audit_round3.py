"""Change-of-basis regressions A/B/C.

These three paths were previously untested:

* **A** — a short ``ListBasis(reduce=ORTHONORMALIZE)`` must realize a genuinely orthonormal
  full basis (its auto-completion is re-orthonormalized) and its cob must round-trip.
* **B** — a cob between two DISTINCT structured bases (``BasisSum`` / ``BasisProd`` /
  ``BasisRep``, and the ``BasisRep↔BasisSum`` cross) each carrying a dynamic-rank FFS
  component must round-trip (the dispatch arms used to call the pure, non-Dim-aware
  combinators and ``TypeError`` on a ``DimAtom`` axis).
* **C** — ``CompleteBasis(ProjectedBasis(f,"image"), default, reduce=ORTHONORMALIZE)`` must
  realize a full orthonormal ambient basis and round-trip (the reduced QR of the wide
  ``[sub|complement]`` stack returns fewer columns than the input).
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa

import grassmann.core as C
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
    """``cob(a→b)·cob(b→a) ≈ I``."""
    ab = _cob(a, b, feeds)
    ba = _cob(b, a, feeds)
    return np.allclose(ab @ ba, np.eye(ab.shape[0]))


def _invertible(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n)), -1) + np.diag(2.0 + np.arange(n))


# ===========================================================================
# A — short ORTHONORMALIZE ListBasis is genuinely orthonormal + round-trips
# ===========================================================================


def _ortho_list_realizes_full_orthonormal(n_vecs: int, with_backup: bool) -> None:
    R3 = C.base("R3", 3)
    vs = [C.var(f"u{i}", R3) for i in range(n_vecs)]
    # non-orthogonal, non-unit frame vectors so transpose != inverse unless re-orthonormalized.
    raw = [np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 2.0])][:n_vecs]
    feeds = {_k(vs[i]): raw[i] for i in range(n_vecs)}
    backup = default_basis(R3) if with_backup else None
    lb = B.list_basis(*vs, backup=backup, reduce=B.Reduce.ORTHONORMALIZE)

    m = _realize(lb, feeds)
    assert m.shape == (3, 3)
    assert np.allclose(m.T @ m, np.eye(3)), "completed basis must be orthonormal (QᵀQ=I)"

    std = default_basis(R3)
    assert _roundtrips(lb, std, feeds)
    # transpose branch must now be valid: cob(std→lb) == realize(lb)ᵀ
    cob_std_to_lb = _cob(std, lb, feeds)
    assert np.allclose(cob_std_to_lb, m.T)


def test_a_short_ortho_list_one_vector_no_backup() -> None:
    _ortho_list_realizes_full_orthonormal(1, with_backup=False)


def test_a_short_ortho_list_two_vectors_no_backup() -> None:
    _ortho_list_realizes_full_orthonormal(2, with_backup=False)


def test_a_short_ortho_list_one_vector_with_backup() -> None:
    _ortho_list_realizes_full_orthonormal(1, with_backup=True)


def test_a_short_ortho_list_two_vectors_with_backup() -> None:
    _ortho_list_realizes_full_orthonormal(2, with_backup=True)


# ===========================================================================
# B — cob between two DISTINCT structured bases w/ a dynamic-rank FFS component
# ===========================================================================


def _surjective_coimg():
    """A surjective ℝ³↠ℝ² ⇒ a DYNAMIC-rank (dim 2) coimg."""
    Rt, Re = C.base("Rt", 3), C.base("Re", 2)
    f = C.var("Fsurj", C.linear_map_space(Rt, Re))
    J = _invertible(3)[:2, :3]
    return f, {_k(f): J.T}


def test_b_basis_sum_distinct_dynamic_ffs_roundtrips() -> None:
    """``cob(BasisSum(FFS, Scalar) → BasisSum(FFS', Scalar))`` with a dynamic FFS summand.

    The two FFS summands are the same coimg space realized two ways: the orthonormal
    ``ProjectedBasis`` and its metricized (``target``/``source``) sibling — DISTINCT bases,
    so the dispatch takes the structured ⊕→⊕ block-diag arm (not the identity short-circuit).
    """
    f, feeds = _surjective_coimg()
    Rt = C.base("Rt", 3)
    # a source (dom) metric makes a coimg (dom-side) ProjectedBasis a genuinely different
    # realized frame ⇒ a distinct basis of the same coimg space.
    s = [C.var(f"s{i}", Rt) for i in range(3)]
    Sm = _invertible(3)
    feeds = {**feeds, **{_k(s[i]): Sm[:, i] for i in range(3)}}
    src = B.list_basis(*s)

    a = B.basis_sum(B.projected_basis(f, "coimg"), B.scalar_basis())
    b = B.basis_sum(B.projected_basis(f, "coimg", source=src), B.scalar_basis())
    assert _roundtrips(a, b, feeds)


def test_b_basis_prod_distinct_dynamic_ffs_roundtrips() -> None:
    """``cob(BasisProd(FFS, Scalar) → BasisProd(FFS', Scalar))`` (Kron arm) with a dynamic FFS."""
    f, feeds = _surjective_coimg()
    Rt = C.base("Rt", 3)
    s = [C.var(f"s{i}", Rt) for i in range(3)]
    Sm = _invertible(3)
    feeds = {**feeds, **{_k(s[i]): Sm[:, i] for i in range(3)}}
    src = B.list_basis(*s)
    a = B.basis_prod(B.projected_basis(f, "coimg"), B.scalar_basis())
    b = B.basis_prod(B.projected_basis(f, "coimg", source=src), B.scalar_basis())
    assert _roundtrips(a, b, feeds)


def test_b_basis_rep_distinct_dynamic_ffs_roundtrips() -> None:
    """``cob(BasisRep(2, FFS) → BasisRep(2, FFS'))`` (block-repeat arm) with a dynamic FFS."""
    f, feeds = _surjective_coimg()
    Rt = C.base("Rt", 3)
    s = [C.var(f"s{i}", Rt) for i in range(3)]
    Sm = _invertible(3)
    feeds = {**feeds, **{_k(s[i]): Sm[:, i] for i in range(3)}}
    src = B.list_basis(*s)
    a = B.basis_rep(2, B.projected_basis(f, "coimg"))
    b = B.basis_rep(2, B.projected_basis(f, "coimg", source=src))
    assert _roundtrips(a, b, feeds)


def test_b_basis_rep_to_basis_sum_cross_dynamic_ffs_roundtrips() -> None:
    """The ``BasisRep ↔ BasisSum`` cross arm with a dynamic FFS inner/summand."""
    f, feeds = _surjective_coimg()
    Rt = C.base("Rt", 3)
    s = [C.var(f"s{i}", Rt) for i in range(3)]
    Sm = _invertible(3)
    feeds = {**feeds, **{_k(s[i]): Sm[:, i] for i in range(3)}}
    src = B.list_basis(*s)
    rep = B.basis_rep(2, B.projected_basis(f, "coimg"))
    summ = B.basis_sum(
        B.projected_basis(f, "coimg", source=src),
        B.projected_basis(f, "coimg", source=src),
    )
    assert _roundtrips(rep, summ, feeds)


# ===========================================================================
# C — CompleteBasis(reduce=ORTHONORMALIZE) over a subspace sub realizes + round-trips
# ===========================================================================


def test_c_complete_basis_orthonormalize_realizes_and_roundtrips() -> None:
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    f = C.var("Finj", C.linear_map_space(Rt, Re))
    J = _invertible(3)[:3, :2]
    feeds = {_k(f): J.T}
    sub = B.projected_basis(f, "image")  # dim-2 subspace of ℝ³
    cod = Re
    cb = B.complete_basis(sub, default_basis(cod), reduce=B.Reduce.ORTHONORMALIZE)

    m = _realize(cb, feeds)
    assert m.shape == (3, 3)
    assert np.allclose(m.T @ m, np.eye(3)), "completed cod basis must be orthonormal"
    assert _roundtrips(cb, default_basis(cod), feeds)
