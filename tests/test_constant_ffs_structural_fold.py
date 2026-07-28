"""Structural (dependency-based, NOT sampling) constant-fold of an FFS realization.

``realize_ffs_basis`` now folds a frame whose operator ``f_matrix`` is a PROVABLE build-time
constant — its dependency cone is a closed constant subprogram (no live program-input ref,
only ``stmt_out`` generators) — to a constant ``FfsRealization`` instead of deferring an
``SvdOp`` with a runtime rank δ.  This is the fix for the Argyris ``∫u·v`` blow-up: ~480
realizations of just two structurally-constant operators (the 2×2 reference tangent-frame
identity and the 4×4 rank-1 Λ² ``alt`` antisymmetrizer) were each emitting an un-foldable
dynamic SVD, OOM-ing codegen.

The soundness contract (``_cone_is_structurally_constant``):

* a frame whose operator is a build-time constant FOLDS — no SvdOp/QrOp/GSvd/QR-sign Stmt is
  emitted, ``cols`` is the numeric range/complement basis matching ``np.linalg.svd``;
* a frame whose operator depends on a program INPUT still emits the SVD (structural check is
  ``False`` — the soundness / conservativeness guard);
* the folded ``cols`` numerically equals the SVD range basis of the same numeric ``M``.
"""

from __future__ import annotations

import numpy as np
import pytest

import polyarray as pa
from polyarray.ir import Provenance, SymInput

from grassmann.lower.space_basis import (
    FfsRealization,
    _cone_is_structurally_constant,
    realize_ffs_basis,
)


# The set of factorization ops the fold must AVOID emitting when it folds.
_FACTORIZATION_OP_TYPES = (
    pa.SvdOp,
    pa.QrOp,
    pa.GSvdFullOp,
    pa.GSvdOp,
)


def _emitted_op_types(program: pa.Program) -> list[type]:
    return [type(s.fn) for s in program.statements]


def _has_factorization(program: pa.Program) -> bool:
    from grassmann.lower.space_basis import _QrSignConventionOp

    for fn in (s.fn for s in program.statements):
        if isinstance(fn, _FACTORIZATION_OP_TYPES) or isinstance(fn, _QrSignConventionOp):
            return True
    return False


def _const_matrix(program: pa.Program, arr: np.ndarray, note: str) -> pa.SymArray:
    """Emit ``arr`` as a nullary ``ConstOp`` Stmt output — a structurally-constant f_matrix."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=float))
    op = pa.ConstOp(key=note, data_bytes=a.tobytes(), shape=a.shape, dtype=str(a.dtype))
    [out] = program.emit_stmt(op, [], [pa.OutSpec(note, a.shape)], note=note)
    return out


# The two operators from the Argyris kernel, as plain numeric matrices.
_EYE2 = np.eye(2)
# A 4×4 antisymmetric readout (the shape of the Λ² ``alt`` frame's operator): a rank-2
# block, so img is 2 columns and coker is 2 — exercises a nontrivial range/complement split.
_ANTISYM4 = np.zeros((4, 4))
_ANTISYM4[0, 3] = 1.0
_ANTISYM4[3, 0] = -1.0


def _svd_cols(M: np.ndarray, which: str) -> np.ndarray:
    """The reference constant basis: numpy full SVD + matrix_rank tolerance + which-slice."""
    u, sv, vh = np.linalg.svd(M, full_matrices=True)
    tol = float(sv.max()) * max(M.shape) * float(np.finfo(float).eps) if sv.size else 0.0
    r = int((sv > tol).sum())
    if which in ("img", "coker"):
        return u[:, :r] if which == "img" else u[:, r:]
    v = vh.T
    return v[:, :r] if which == "coimg" else v[:, r:]


# ---------------------------------------------------------------------------
# (a) a build-time-constant operator FOLDS: no factorization Stmt, cols == SVD basis.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M, dim, which",
    [
        (_EYE2, 2, "img"),          # reference tangent-frame identity (the 2×2 case)
        (_ANTISYM4, 4, "img"),      # Λ² alt antisymmetrizer range (the 4×4 case)
        (_ANTISYM4, 4, "coker"),    # its orthogonal complement
        (_ANTISYM4, 4, "coimg"),    # right-factor variants
        (_ANTISYM4, 4, "ker"),
    ],
)
def test_constant_operator_folds_no_svd(M, dim, which):
    program = pa.Program("t")
    f_matrix = _const_matrix(program, M, "const_op")

    assert _cone_is_structurally_constant(program, f_matrix) is True

    n_stmts_before = len(program.statements)
    real = realize_ffs_basis(program, which, f_matrix, dim, dim)

    # No factorization Stmt was emitted (and in fact NO Stmt at all — cols is a pure constant).
    assert not _has_factorization(program), _emitted_op_types(program)
    assert len(program.statements) == n_stmts_before
    assert real.op == "const"
    assert real.rank is None and real.rank_dim is None

    # (c) cols is an orthonormal basis of THE SAME SUBSPACE the SVD which-slice spans.
    #
    # Asserted as "same subspace", NOT "same columns as `_svd_cols`". The which-slice fixes the
    # subspace, but its individual columns are only determined up to a rotation INSIDE each
    # degenerate singular subspace — and `_ANTISYM4`'s nonzero singular values ARE degenerate, so
    # `u[:, :r]` is a discontinuous function of `M` there and pinning it would be pinning
    # arbitrary rounding. grassmann now realizes the frame canonically from the range PROJECTOR
    # (`_canonical_subspace_basis`), which is continuous in the subspace; the equal-projector test
    # below is the strongest statement that is actually well-posed, and
    # `test_constant_fold_frame_is_canonical_per_subspace` adds the property the old byte-identity
    # assertion was standing in for.
    got = np.asarray(real.cols.cells, dtype=float)
    expected = _svd_cols(M, which)
    assert got.shape == expected.shape
    assert np.allclose(got.T @ got, np.eye(got.shape[1]), atol=1e-12), "columns not orthonormal"
    assert np.allclose(got @ got.T, expected @ expected.T, atol=1e-12), (
        "folded frame spans a different subspace than the SVD which-slice")


def test_eye_via_eyeop_folds():
    """The identity built via the substrate's own ``EyeOp`` (nullary) also folds."""
    program = pa.Program("t")
    [f_matrix] = program.emit_stmt(pa.EyeOp(2), [], [pa.OutSpec("eye", (2, 2))], note="eye")
    assert _cone_is_structurally_constant(program, f_matrix) is True
    real = realize_ffs_basis(program, "img", f_matrix, 2, 2)
    assert not _has_factorization(program)
    assert np.allclose(np.asarray(real.cols.cells, dtype=float), np.eye(2))


# ---------------------------------------------------------------------------
# (b) an input-dependent operator is NOT folded: SVD is emitted (soundness guard).
# ---------------------------------------------------------------------------


def _input_matrix_program() -> tuple[pa.Program, pa.SymArray]:
    prov = Provenance(kind="point", origin="J", index=(), label="J")
    program = pa.Program("t", inputs=[SymInput(name="J", shape=(2, 2), provenance=prov)])
    return program, program.input("J")


def test_input_dependent_operator_not_folded_direct():
    program, J = _input_matrix_program()
    assert _cone_is_structurally_constant(program, J) is False
    real = realize_ffs_basis(program, "img", J, 2, 2)
    assert _has_factorization(program)
    assert isinstance(real.rank_dim, pa.DimAtom)  # the deferred runtime rank δ
    assert real.op == "SvdOp"


def test_input_dependent_operator_not_folded_through_stmt():
    """f_matrix derived from an input via a Stmt (its cone reads input generators)."""
    program, J = _input_matrix_program()
    [Jt] = program.emit_stmt(pa.TransposeOp(), [J], [pa.OutSpec("Jt", (2, 2))], note="Jt")
    assert _cone_is_structurally_constant(program, Jt) is False
    realize_ffs_basis(program, "img", Jt, 2, 2)
    assert _has_factorization(program)


def test_mixed_const_plus_input_not_folded():
    """A const combined with an input is NOT constant (the input generator taints the cone)."""
    program, J = _input_matrix_program()
    c = _const_matrix(program, _EYE2, "c")
    [s] = program.emit_stmt(pa.AddOp(2), [c, J], [pa.OutSpec("s", (2, 2))], note="sum")
    assert _cone_is_structurally_constant(program, s) is False
    realize_ffs_basis(program, "img", s, 2, 2)
    assert _has_factorization(program)


# ---------------------------------------------------------------------------
# A chain of constant Stmts (stmt_out generators only) still folds — proves the
# check accepts an in-cone OutputRef/stmt_out chain, not just a single nullary leaf.
# ---------------------------------------------------------------------------


def test_constant_chain_folds():
    program = pa.Program("t")
    c = _const_matrix(program, _ANTISYM4, "c")
    [ct] = program.emit_stmt(pa.TransposeOp(), [c], [pa.OutSpec("ct", (4, 4))], note="ct")
    assert _cone_is_structurally_constant(program, ct) is True
    real = realize_ffs_basis(program, "img", ct, 4, 4)
    assert not _has_factorization(program)
    got = np.asarray(real.cols.cells, dtype=float)
    expected = _svd_cols(_ANTISYM4.T, "img")            # same SUBSPACE, not the same columns
    assert np.allclose(got @ got.T, expected @ expected.T, atol=1e-12)


def test_constant_fold_frame_is_canonical_per_subspace():
    """ONE subspace ⇒ ONE realized frame, whatever operator presented it.

    This is the property the old ``cols == _svd_cols(M)`` assertion was standing in for, stated so
    that it is well-posed. Two operators with the SAME range must fold to the IDENTICAL frame —
    otherwise a change of basis composed across two ``Space`` nodes naming one subspace comes out a
    rotation, which is the σ defect recorded in ``.claude/NOTES-sigma-plumbing.md``: for a
    reoriented facet, ``Img(alt_canonical)`` and ``Img(alt_permuted)`` are different nodes for the
    same subspace, and no memo can tie them because their keys legitimately differ.

    ``_ANTISYM4`` is the operator that exposed it: its nonzero singular values are DEGENERATE, so
    ``u[:, :r]`` moves by O(1) under 1e-15 of rounding. Here the two presentations are ``M`` and
    ``M·G`` for an invertible ``G`` — same range by construction, unrelated SVD factors.
    """
    rng = np.random.default_rng(0)
    g = rng.standard_normal((4, 4))
    assert abs(np.linalg.det(g)) > 1e-6

    frames = []
    for name, op in (("plain", _ANTISYM4), ("regauged", _ANTISYM4 @ g)):
        program = pa.Program("t")
        real = realize_ffs_basis(program, "img", _const_matrix(program, op, name), 4, 4)
        frames.append(np.asarray(real.cols.cells, dtype=float))

    assert frames[0].shape == frames[1].shape
    assert np.allclose(frames[0], frames[1], atol=1e-9), (
        f"one subspace realized two frames:\n{frames[0]}\nvs\n{frames[1]}")
