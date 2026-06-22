"""``ListBasis(reduce=RREF)`` is ALWAYS a full basis of its ambient.

A ``ListBasis`` is always a full basis of its ambient (the vectors' space).  Under ``RREF``
with no explicit backup, the frame's linearly-independent columns are kept and then the frame
is **completed** to the full ambient from the ambient's standard basis (the auto-backup).  So a
rank-deficient ℝ³ RREF frame realizes to a full 3×3 basis (its 2 independent columns plus a
standard completion), and a cob of such a basis is well-posed (no singular inverse).
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


def _rref_basis():
    R3 = C.base("R3", 3)
    v0, v1, v2 = C.var("v0", R3), C.var("v1", R3), C.var("v2", R3)
    return B.list_basis(v0, v1, v2, reduce=B.Reduce.RREF), (v0, v1, v2)


def test_rank_deficient_rref_completes_to_full_ambient() -> None:
    """A rank-2 frame (v2 = v0+v1) realizes to a FULL 3×3 ambient basis: its 2 independent
    columns plus a standard completion (auto-backup)."""
    lb, (v0, v1, v2) = _rref_basis()
    feeds = {
        _k(v0): np.array([1.0, 0.0, 0.0]),
        _k(v1): np.array([0.0, 1.0, 0.0]),
        _k(v2): np.array([1.0, 1.0, 0.0]),  # dependent ⇒ dropped, completed from standard
    }
    m = _realize(lb, feeds)
    assert m.shape == (3, 3)
    assert np.linalg.matrix_rank(m) == 3  # a genuine full basis
    # the first two kept columns are the independent explicit ones
    assert np.allclose(m[:, 0], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(m[:, 1], np.array([0.0, 1.0, 0.0]))


def test_full_rank_rref_unchanged() -> None:
    """A full-rank frame is returned unchanged (all columns kept)."""
    lb, (v0, v1, v2) = _rref_basis()
    feeds = {
        _k(v0): np.array([1.0, 0.0, 0.0]),
        _k(v1): np.array([0.0, 1.0, 0.0]),
        _k(v2): np.array([0.0, 0.0, 1.0]),
    }
    m = _realize(lb, feeds)
    assert m.shape == (3, 3)
    assert np.allclose(m, np.eye(3))


def test_rref_basis_cob_no_singular_inverse() -> None:
    """The identity cob of a rank-deficient RREF basis (now full-ambient) is well-posed."""
    lb, (v0, v1, v2) = _rref_basis()
    feeds = {
        _k(v0): np.array([1.0, 0.0, 0.0]),
        _k(v1): np.array([0.0, 1.0, 0.0]),
        _k(v2): np.array([1.0, 1.0, 0.0]),
    }
    M = _cob(lb, lb, feeds)
    assert M.shape == (3, 3)
    assert np.all(np.isfinite(M))
    assert np.allclose(M, np.eye(3))


def test_rref_basis_cob_round_trips_against_listbasis() -> None:
    """A cob between an RREF basis (now full ℝ³) and another full ℝ³ ListBasis round-trips."""
    lb, (v0, v1, v2) = _rref_basis()
    R3 = C.base("R3", 3)
    # another (short, hence auto-completed) full ℝ³ ListBasis
    w0, w1 = C.var("w0", R3), C.var("w1", R3)
    other = B.list_basis(w0, w1)
    feeds = {
        _k(v0): np.array([1.0, 0.0, 0.0]),
        _k(v1): np.array([0.0, 1.0, 0.0]),
        _k(v2): np.array([1.0, 1.0, 0.0]),
        _k(w0): np.array([1.0, 1.0, 0.0]),
        _k(w1): np.array([1.0, -1.0, 0.0]),
    }
    m = _cob(lb, other, feeds)
    back = _cob(other, lb, feeds)
    assert m.shape == (3, 3)
    assert np.allclose(m @ back, np.eye(3))


# ===========================================================================
# A ListBasis is ALWAYS a full basis of its ambient (Part A).
# ===========================================================================


def test_short_listbasis_realizes_full_ambient_and_space_is_ambient() -> None:
    """A short ℝ³ ``ListBasis`` (2 vectors) realizes to a FULL 3×3 ambient basis (2 explicit +
    standard completion), and its space is ℝ³ (the ambient), not a subspace."""
    R3 = C.base("R3", 3)
    v0, v1 = C.var("v0", R3), C.var("v1", R3)
    lb = B.list_basis(v0, v1)
    feeds = {_k(v0): np.array([1.0, 0.0, 0.0]), _k(v1): np.array([0.0, 1.0, 0.0])}
    m = _realize(lb, feeds)
    assert m.shape == (3, 3)
    assert np.linalg.matrix_rank(m) == 3
    np.testing.assert_allclose(m[:, 0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(m[:, 1], np.array([0.0, 1.0, 0.0]))
    # space is the ambient ℝ³ (no longer None for a short frame — audit #3 closed for term bases)
    prog = pa.Program("sp")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    assert _RM._space_of_basis(ctx, lb) == R3
    assert _RM.basis_dim(lb) == 3


def test_two_short_listbases_over_different_planes_cob_as_full_r3() -> None:
    """Two short ℝ³ ``ListBasis``es over DIFFERENT k-planes are now two FULL ℝ³ bases ⇒ a genuine
    same-space cob that round-trips (audit #3: no meaningless cross-subspace cob for term bases)."""
    R3 = C.base("R3", 3)
    a0, a1 = C.var("a0", R3), C.var("a1", R3)
    b0, b1 = C.var("b0", R3), C.var("b1", R3)
    ba = B.list_basis(a0, a1)  # spans the x,y-plane
    bb = B.list_basis(b0, b1)  # spans the x,z-plane (a DIFFERENT 2-plane)
    feeds = {
        _k(a0): np.array([1.0, 0.0, 0.0]),
        _k(a1): np.array([0.0, 1.0, 0.0]),
        _k(b0): np.array([1.0, 0.0, 0.0]),
        _k(b1): np.array([0.0, 0.0, 1.0]),
    }
    m = _cob(ba, bb, feeds)
    back = _cob(bb, ba, feeds)
    assert m.shape == (3, 3)
    assert np.allclose(m @ back, np.eye(3))


def test_complete_basis_projected_image_round_trips() -> None:
    """``CompleteBasis(ProjectedBasis(f,"image"), default_basis(cod), RREF)`` completes Img(f)
    (a proper subspace) to a full codomain basis and round-trips against the codomain basis."""
    from grassmann.lower.space_basis import default_basis

    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    f = C.var("F", C.linear_map_space(Rt, Re), props=C.props.inj)
    projb = B.projected_basis(f, "image")
    cod_std = default_basis(Re)
    complete = B.complete_basis(projb, cod_std, reduce=B.Reduce.RREF)
    feeds = {_k(f): np.array([[2.0, 1.0, 1.0], [0.0, 3.0, 1.0]])}  # stored = J.T (cod=3, dom=2)
    cols = _realize(complete, feeds)
    assert cols.shape == (3, 3)
    assert np.linalg.matrix_rank(cols) == 3  # a full codomain basis
    m = _cob(complete, cod_std, feeds)
    back = _cob(cod_std, complete, feeds)
    assert m.shape == (3, 3)
    assert np.allclose(m @ back, np.eye(3))
