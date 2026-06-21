"""Audit Fix 2 — ``ListBasis(reduce=RREF, backup=None)`` applies an RREF column SELECTION.

Without a ``backup`` the old code returned the raw (possibly rank-deficient ⇒ singular)
frame.  Now a rank-deficient frame realizes to its linearly-independent columns — a genuine
basis of the frame's span (column count = the frame's rank) — while a full-rank frame is
unchanged.  A cob of such a basis no longer takes a singular inverse.
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


def test_rank_deficient_rref_selects_independent_columns() -> None:
    """A rank-2 frame (v2 = v0+v1) realizes to a (3, 2) FULL-COLUMN-RANK basis of its span."""
    lb, (v0, v1, v2) = _rref_basis()
    feeds = {
        _k(v0): np.array([1.0, 0.0, 0.0]),
        _k(v1): np.array([0.0, 1.0, 0.0]),
        _k(v2): np.array([1.0, 1.0, 0.0]),  # dependent ⇒ dropped
    }
    m = _realize(lb, feeds)
    assert m.shape == (3, 2)
    assert np.linalg.matrix_rank(m) == 2
    # the kept columns are the first two independent ones
    assert np.allclose(m, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))


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
    """The identity cob of a rank-deficient RREF basis is well-posed (no singular inverse)."""
    lb, (v0, v1, v2) = _rref_basis()
    feeds = {
        _k(v0): np.array([1.0, 0.0, 0.0]),
        _k(v1): np.array([0.0, 1.0, 0.0]),
        _k(v2): np.array([1.0, 1.0, 0.0]),
    }
    M = _cob(lb, lb, feeds)
    assert M.shape == (2, 2)
    assert np.all(np.isfinite(M))
    assert np.allclose(M, np.eye(2))


def test_rref_basis_cob_round_trips_against_listbasis() -> None:
    """A cob between an RREF basis (rank 2) and a NONE ListBasis of the same span round-trips."""
    lb, (v0, v1, v2) = _rref_basis()
    R3 = C.base("R3", 3)
    # an explicit 2-D basis of the same {x,y}-plane span
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
    assert m.shape == (2, 2)
    assert np.allclose(m @ back, np.eye(2))
