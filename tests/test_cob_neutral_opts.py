"""Correctness-neutral cob optimizations — each takes a cheaper/exacter route while keeping the
emitted *numbers* identical to the generic fallback (`_cob_via_standard`).

Every test here pins both the result (round-trip ⇒ identity AND/OR equal to the old via-standard
matrix) and, where it is the point of the change, the *emitted op* (a stmt note or an op type).
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B

_RM = sys.modules["grassmann.lower.represent"]


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _invertible(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n)), -1) + np.diag(2.0 + np.arange(n))


def _ctx_prog():
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    return ctx, prog


def _cob(a, b, feeds):
    ctx, prog = _ctx_prog()
    prog.add_output("M", _RM._basis_cob_matrix(ctx, a, b))
    return np.asarray(prog.run(feeds).get("M"), float), prog


def _cob_via_standard(a, b, feeds):
    """Force the generic standard route (the route the optimization replaces)."""
    ctx, prog = _ctx_prog()
    prog.add_output("M", _RM._cob_via_standard(ctx, a, b))
    return np.asarray(prog.run(feeds).get("M"), float)


def _listb(space, name, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    return B.list_basis(*vs), {_k(v): cols[:, i] for i, v in enumerate(vs)}


# --- Item A: Primal↔Dual mixed pairs take the exact inverse-transpose ----------------------


_A_CASES = []
for _n in (2, 3):
    _Vn = C.base("V", _n)
    _la, _fa = _listb(_Vn, f"A_a{_n}", _invertible(_n))
    _lb, _fb = _listb(_Vn, f"A_b{_n}", _invertible(_n) + np.eye(_n))
    _feeds = {**_fa, **_fb}
    # The two covariant wrappers are interchangeable space-wise.
    _A_CASES += [
        (f"Dual(a)->Primal(b) n={_n}", B.dual_basis(_la), B.primal_basis(_lb), _feeds),
        (f"Primal(a)->Dual(b) n={_n}", B.primal_basis(_la), B.dual_basis(_lb), _feeds),
    ]


@pytest.mark.parametrize("label,frm,to,feeds", _A_CASES, ids=[c[0] for c in _A_CASES])
def test_item_a_mixed_variance_exact_inverse_transpose(label, frm, to, feeds) -> None:
    """Dual↔Primal mixed pairs equal the generic via-standard result and round-trip to I."""
    got, _ = _cob(frm, to, feeds)
    via = _cob_via_standard(frm, to, feeds)
    assert got == pytest.approx(via), f"{label}: specialized cob != via-standard"
    back, _ = _cob(to, frm, feeds)
    n = got.shape[0]
    assert got @ back == pytest.approx(np.eye(n)), f"{label}: not invertible"


# --- Item B: ImageBasis -> ProjectedBasis(image) of the same map = the R factor, no InvOp(I) ---


_B_SHAPES = [(2, 2), (3, 3), (2, 3), (3, 5)]


@pytest.mark.parametrize("dom,cod", _B_SHAPES, ids=[f"{d}to{c}" for d, c in _B_SHAPES])
def test_item_b_image_to_projected_image_skips_inv_identity(dom, cod) -> None:
    """ImageBasis(f) → metric-free ProjectedBasis(f,'image') returns R directly (no InvOp(I))."""
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var(f"F{dom}{cod}", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = _invertible(max(dom, cod))[:cod, :dom]
    feeds = {_k(f): Jm.T}
    frm, to = B.image_basis(f), B.projected_basis(f, "image")

    got, prog = _cob(frm, to, feeds)
    via = _cob_via_standard(frm, to, feeds)
    assert got == pytest.approx(via), "specialized cob != via-standard"
    # The whole point: no InvOp at all (the old route InvOp'd the to-side identity).
    assert not any(isinstance(s.fn, pa.InvOp) for s in prog.statements), "InvOp(I) not skipped"
    back, _ = _cob(to, frm, feeds)
    n = got.shape[0]
    assert got @ back == pytest.approx(np.eye(n)), "not invertible"


def test_item_b_metric_target_still_generic() -> None:
    """A target-metric ProjectedBasis(image) target keeps the generic reorthonormalize route."""
    dom, cod = 2, 3
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var("Ft", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = _invertible(cod)[:, :dom]
    tgt, ft = _listb(Re, "tg", _invertible(cod))
    frm = B.image_basis(f)
    to = B.projected_basis(f, "image", target=tgt)
    feeds = {_k(f): Jm.T, **ft}
    got, _ = _cob(frm, to, feeds)
    via = _cob_via_standard(frm, to, feeds)
    assert got == pytest.approx(via)
    back, _ = _cob(to, frm, feeds)
    n = got.shape[0]
    assert got @ back == pytest.approx(np.eye(n))

# --- Item C: ProjectedBasis identity guard is which-aware (only own-side metric matters) --------


def _has_gsvd(prog) -> bool:
    return any("GSvd" in type(s.fn).__name__ for s in prog.statements)


def test_item_c_image_irrelevant_source_side_is_identity() -> None:
    """image: own side is target(cod); differing only source(dom) => identity (no GSVD)."""
    dom, cod = 2, 3
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var("Fimg", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = _invertible(cod)[:, :dom]
    s1, fs1 = _listb(Rt, "s1", _invertible(dom))
    s2, fs2 = _listb(Rt, "s2", _invertible(dom) + np.eye(dom))
    frm = B.projected_basis(f, "image", source=s1)
    to = B.projected_basis(f, "image", source=s2)
    feeds = {_k(f): Jm.T, **fs1, **fs2}
    got, prog = _cob(frm, to, feeds)
    assert not _has_gsvd(prog), "identity short-circuit did not fire (GSVD emitted)"
    assert got == pytest.approx(np.eye(got.shape[0])), "irrelevant-side-differ cob != I"
    assert got == pytest.approx(_cob_via_standard(frm, to, feeds))


def test_item_c_ker_irrelevant_target_side_is_identity() -> None:
    """ker: own side is source(dom); differing only target(cod) => identity (no GSVD)."""
    dom, cod = 3, 2
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    h = C.var("Hk", C.linear_map_space(Rt, Re), props=C.props.surj)
    Hm = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    t1, ft1 = _listb(Re, "kt1", _invertible(cod))
    t2, ft2 = _listb(Re, "kt2", _invertible(cod) + np.eye(cod))
    frm = B.projected_basis(h, "ker", target=t1)
    to = B.projected_basis(h, "ker", target=t2)
    feeds = {_k(h): Hm.T, **ft1, **ft2}
    got, prog = _cob(frm, to, feeds)
    assert not _has_gsvd(prog), "identity short-circuit did not fire (GSVD emitted)"
    assert got == pytest.approx(np.eye(got.shape[0]))
    assert got == pytest.approx(_cob_via_standard(frm, to, feeds))


def test_item_c_image_own_target_side_differs_still_reorthonormalizes() -> None:
    """image: differing target(cod) metric is a genuine reorthonormalize (NOT identity)."""
    dom, cod = 2, 3
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var("Fimg2", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = _invertible(cod)[:, :dom]
    t1, ft1 = _listb(Re, "t1", _invertible(cod))
    t2, ft2 = _listb(Re, "t2", _invertible(cod) + np.eye(cod))
    frm = B.projected_basis(f, "image", target=t1)
    to = B.projected_basis(f, "image", target=t2)
    feeds = {_k(f): Jm.T, **ft1, **ft2}
    got, _ = _cob(frm, to, feeds)
    assert got == pytest.approx(_cob_via_standard(frm, to, feeds))
    assert got != pytest.approx(np.eye(got.shape[0])), "own-side-differ wrongly collapsed to I"
    back, _ = _cob(to, frm, feeds)
    assert got @ back == pytest.approx(np.eye(got.shape[0]))

# --- Item D: dynamic-δ FFS `to` is square (same DimAtom both axes) => InvOp, not PinvOp ----------


def test_item_d_dynamic_ffs_to_emits_inv_not_pinv() -> None:
    """A dynamic-rank FFS `to` (no-props map, SVD δ) realizes square => exact InvOp, no PinvOp."""
    dom, cod = 2, 3
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var("Fdyn", C.linear_map_space(Rt, Re))   # NO props => dynamic/SVD rank => DimAtom δ
    Jm = _invertible(max(dom, cod))[:cod, :dom]
    tgt, ft = _listb(Re, "tgD", _invertible(cod))
    frm = B.image_basis(f)
    to = B.projected_basis(f, "image", target=tgt)  # dynamic δ on both axes (square in-space)
    feeds = {_k(f): Jm.T, **ft}

    ctx, prog = _ctx_prog()
    prog.add_output("M", _RM._cob_via_standard(ctx, frm, to))
    notes = [s.note or "" for s in prog.statements]
    assert any("C⁻¹" in n for n in notes), "dynamic-δ FFS `to` did not emit the square InvOp"
    assert not any("pinv(C)" in n for n in notes), "dynamic-δ FFS `to` still emitted PinvOp"
    got = np.asarray(prog.run(feeds).get("M"), float)
    assert np.all(np.isfinite(got))
    # results unchanged + round-trip
    back, _ = _cob(to, frm, feeds)
    fwd, _ = _cob(frm, to, feeds)
    assert fwd @ back == pytest.approx(np.eye(fwd.shape[0]))

# --- Item E: _identity_like on a dynamic FFS basis sizes the eye from the memoized seam cols -----


def test_item_e_dynamic_ffs_identity_skips_in_space_einsum() -> None:
    """A dynamic-rank FFS identity-cob is the correct δ×δ eye sized from the seam (no extra
    in-space U_dᵀ·imm einsum)."""
    dom, cod = 2, 3
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var("FdynE", C.linear_map_space(Rt, Re))   # no props => dynamic rank
    Jm = _invertible(max(dom, cod))[:cod, :dom]
    feeds = {_k(f): Jm.T}
    pb = B.projected_basis(f, "image")

    ctx, prog = _ctx_prog()
    prog.add_output("M", _RM._identity_like(ctx, pb))
    # No in-space immersion einsum is emitted just to size the eye.
    assert not any(isinstance(s.fn, pa.EinsumStmtOp) for s in prog.statements), \
        "_identity_like emitted an in-space einsum to size a dynamic FFS eye"
    got = np.asarray(prog.run(feeds).get("M"), float)
    assert got == pytest.approx(np.eye(got.shape[0])), "dynamic FFS identity != δ×δ eye"
    assert got.shape == (dom, dom)   # runtime rank = 2 for this full-rank injective feed


def test_item_e_static_ffs_identity_unchanged() -> None:
    """A static-dim FFS identity-cob is still the right eye (the static path is unchanged)."""
    dom, cod = 2, 3
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var("FstatE", C.linear_map_space(Rt, Re), props=C.props.inj)  # static rank = dom
    Jm = _invertible(cod)[:, :dom]
    feeds = {_k(f): Jm.T}
    pb = B.projected_basis(f, "image")
    got, _ = _cob(pb, pb, feeds)
    assert got == pytest.approx(np.eye(dom))


# --- Item F: dual/primal cob of orthonormal inners skips the InvOp (orthogonal ⇒ ⁻ᵀ = itself) ----


def test_item_f_orthonormal_dual_cob_skips_inv_transpose() -> None:
    """`cob(Dual(o₁) → Dual(o₂))` with standard-orthonormal inners is orthogonal, so the
    inverse-transpose IS the inner cob — no `InvTranspose` op is emitted; a non-orthonormal inner
    still uses it.  Results unchanged (round-trips)."""
    from grassmann.core import Reduce
    V = C.base("V", 3)
    av = [C.var(f"fa{i}", V) for i in range(3)]
    bv = [C.var(f"fb{i}", V) for i in range(3)]
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    Bm = np.array([[2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    feeds = {**{_k(av[i]): A[:, i] for i in range(3)}, **{_k(bv[i]): Bm[:, i] for i in range(3)}}
    o1 = B.dual_basis(B.list_basis(*av, reduce=Reduce.ORTHONORMALIZE))
    o2 = B.dual_basis(B.list_basis(*bv, reduce=Reduce.ORTHONORMALIZE))
    m, prog = _cob(o1, o2, feeds)
    assert not any("(cob)^-T" in (s.note or "") for s in prog.statements)  # InvOp skipped
    back, _ = _cob(o2, o1, feeds)
    assert m @ back == pytest.approx(np.eye(3))                            # results unchanged
    # a non-orthonormal inner still takes the inverse-transpose
    _, prog2 = _cob(B.dual_basis(B.list_basis(*av)), B.dual_basis(B.list_basis(*bv)), feeds)
    assert any("(cob)^-T" in (s.note or "") for s in prog2.statements)
