"""The extended ``BaseBasis`` — the term-level "term basis" (`20 §7`).

``BaseBasis(name, vectors?, space?, backup?)``:

* ``space`` makes the basis **self-sizing** — a standard basis (no explicit vectors) recovers
  its dimension / ambient from the space, closing the old "standard basis carries no dim" gap.
* ``vectors`` may be a **Term frame** (a tuple of Term vectors realized at lowering, like
  ``ListBasis``) as well as a numeric matrix.
* ``backup`` **completes** a rank-deficient frame to a full basis (RREF / pivoted greedy
  selection: keep the frame's independent columns, fill the rest from the backup).
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann as G
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.lower._dims import DynamicDim, basis_dim
from grassmann.lower.cob import _basis_eq
from grassmann.lower.space_basis import default_basis

_RM = sys.modules["grassmann.lower.represent"]


def _realize(basis: B.Basis, feeds: dict | None = None) -> np.ndarray:
    prog = pa.Program("r")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    cols = _RM.realize_basis_cols(ctx, basis)
    prog.add_output("C", cols)
    return np.asarray(prog.run(feeds or {})["C"], float)


def _vkey(v: G.Expr) -> str:
    return f"{v.node.name}_{v.node.uid}"


# --- self-sizing standard basis --------------------------------------------


def test_default_standard_basis_carries_space() -> None:
    """``default_basis(Base)`` now carries the space so the standard basis self-sizes."""
    sp = S.Base("V", S.DimLit(3))
    db = default_basis(sp)
    assert isinstance(db, B.BaseBasis)
    assert db.vectors is None
    assert db.space == sp
    assert basis_dim(db) == 3  # previously raised DynamicDim


def test_self_sizing_standard_basis_realizes_to_identity() -> None:
    """A standard basis with a ``space`` realizes standalone to the ambient identity."""
    sp = S.Base("V", S.DimLit(3))
    assert _realize(default_basis(sp)) == pytest.approx(np.eye(3))


def test_standard_basis_without_space_is_still_dynamic() -> None:
    """A bare standard basis (no space, no vectors) still has no static dim of its own."""
    with pytest.raises(DynamicDim):
        basis_dim(B.base_basis("V"))


def test_space_is_identity_irrelevant_for_basis_eq() -> None:
    """``space`` is sizing metadata: ``_basis_eq`` ignores it (so cob short-circuits hold)."""
    sp = S.Base("V", S.DimLit(3))
    assert _basis_eq(B.base_basis("V", space=sp), B.base_basis("V"))


# --- term frame ------------------------------------------------------------


def test_term_frame_realizes_stacked_columns() -> None:
    """A ``vectors`` tuple of Term vectors realizes as the stacked coordinate columns."""
    W = G.space("W", 3)
    a = G.var("a", W)
    b = G.var("b", W)
    frame = B.base_basis("W", vectors=(a.node, b.node), space=W.node)
    av = np.array([1.0, 0.0, 1.0])
    bv = np.array([0.0, 1.0, 1.0])
    cols = _realize(frame, {_vkey(a): av, _vkey(b): bv})
    assert cols == pytest.approx(np.column_stack([av, bv]))


# --- backup completion -----------------------------------------------------


def test_backup_completes_rank_deficient_frame() -> None:
    """A 1-vector frame in ``R³`` completes to a full-rank basis via the standard backup."""
    W = G.space("W", 3)
    c = G.var("c", W)
    backup = B.base_basis("W", space=W.node)  # standard identity
    basis = B.base_basis("W", vectors=(c.node,), space=W.node, backup=backup)
    cv = np.array([1.0, 2.0, 3.0])
    cols = _realize(basis, {_vkey(c): cv})
    assert cols.shape == (3, 3)
    assert np.linalg.matrix_rank(cols) == 3          # a genuine basis
    assert cols[:, 0] == pytest.approx(cv)            # the frame vector is kept first


def test_backup_skips_dependent_frame_columns() -> None:
    """A degenerate frame (collinear vectors) keeps only independent columns, fills the rest."""
    W = G.space("W", 3)
    c = G.var("c", W)
    d = G.var("d", W)
    backup = B.base_basis("W", space=W.node)
    basis = B.base_basis("W", vectors=(c.node, d.node), space=W.node, backup=backup)
    cv = np.array([1.0, 0.0, 0.0])
    dv = np.array([2.0, 0.0, 0.0])  # collinear with c ⇒ contributes no new direction
    cols = _realize(basis, {_vkey(c): cv, _vkey(d): dv})
    assert cols.shape == (3, 3)
    assert np.linalg.matrix_rank(cols) == 3
    assert cols[:, 0] == pytest.approx(cv)  # first independent frame column kept


# --- cob with a self-sizing standard basis ---------------------------------


def test_cob_from_self_sizing_standard_basis() -> None:
    """``cob(standard, rotated)`` works standalone now that the standard basis self-sizes."""
    sp = S.Base("V", S.DimLit(2))
    rot = np.array([[2.0, 1.0], [0.0, 3.0]])
    std = default_basis(sp)                 # standard, carries space
    rotated = B.base_basis("V", rot)
    prog = pa.Program("cob")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    m = _RM._basis_cob_matrix(ctx, std, rotated)
    prog.add_output("M", m)
    got = np.asarray(prog.run({})["M"], float)
    assert got == pytest.approx(np.linalg.inv(rot))  # cob(std→rot) = pinv(rot)·I
