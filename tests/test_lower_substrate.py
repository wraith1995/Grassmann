"""IW4-substrate tests: ``space_basis`` / ``cob`` / ``choose`` (`41 §1`/`§1b`/`§2`).

The first integration of the Grassmann side with the shipped polyarray additions
(`42`): ``cob``/FFS realization emit real ``SvdOp``/``QrOp``/``GSvdOp``/``DimAtom`` Stmts
and run through ``polyarray.Program.run`` on small numeric examples.
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.lower.represent  # noqa: F401 (load the module for _Ctx/_basis_cob_matrix)
from grassmann.core import props as P
from grassmann.lower import (
    candidate_bases,
    choose,
    default_basis,
    realize_ffs_basis,
    score_basis,
    space_basis,
)
from grassmann.lower.represent import _basis_eq

_RM = sys.modules["grassmann.lower.represent"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _listb(name: str, cols: np.ndarray):
    """An explicit ``ListBasis`` (the term basis) of ``ℝⁿ`` whose columns are ``cols``.

    Returns ``(basis, feeds)``: each basis vector is a Term var fed the matching column.
    """
    n_amb = cols.shape[0]
    sp = C.base(name, n_amb)
    vs = [C.var(f"{name}{i}", sp) for i in range(cols.shape[1])]
    feeds = {f"{v.name}_{v.uid}": cols[:, i] for i, v in enumerate(vs)}
    return C.list_basis(*vs), feeds


def _cob_run(b_from, b_to, feeds: dict) -> np.ndarray:
    """The change-of-basis matrix ``b_from → b_to`` via the represent-level (total) cob."""
    prog = pa.Program("cob")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    m = _RM._basis_cob_matrix(ctx, b_from, b_to)
    prog.add_output("M", m)
    return np.asarray(prog.run(feeds)["M"], float)


def _sym_matrix(prog: pa.Program, Mn: np.ndarray, prefix: str):
    cells = np.empty(Mn.shape, dtype=object)
    vals: dict[str, float] = {}
    for idx in np.ndindex(*Mn.shape):
        name = prefix + "_".join(map(str, idx))
        cells[idx] = pa.RationalFunction.atom(name)
        vals[name] = float(Mn[idx])
    return pa.SymArray(cells, program=prog), vals


def _block_diag(*ms: np.ndarray) -> np.ndarray:
    rows = sum(m.shape[0] for m in ms)
    cols = sum(m.shape[1] for m in ms)
    out = np.zeros((rows, cols))
    r = c = 0
    for m in ms:
        out[r:r + m.shape[0], c:c + m.shape[1]] = m
        r += m.shape[0]
        c += m.shape[1]
    return out


# ===========================================================================
# space_basis — default + candidates per Space node (`41 §1`)
# ===========================================================================


def test_default_basis_scalar() -> None:
    db, cands = space_basis(C.scalar())
    assert db == C.scalar_basis()
    assert cands == (C.scalar_basis(),)


def _rb(n: int):
    """The default basis of ``space(_, n)`` = ``⊕ⁿ Scalar`` ⇒ ``BasisRep(n, ScalarBasis)``."""
    return C.basis_rep(n, C.scalar_basis())


def test_default_basis_dual() -> None:
    db, _ = space_basis(C.dual(C.base("V", 3)))
    assert _basis_eq(db, C.dual_basis(_rb(3)))


def test_default_basis_oplus_oprod() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    db_sum, _ = space_basis(C.oplus(V, W))
    assert _basis_eq(db_sum, C.basis_sum(_rb(2), _rb(3)))
    db_prod, _ = space_basis(C.oprod(V, W))
    assert _basis_eq(db_prod, C.basis_prod(_rb(2), _rb(3)))


def test_default_basis_oplusrep() -> None:
    db, _ = space_basis(C.oplus_rep(4, C.base("V", 2)))
    assert _basis_eq(db, C.basis_rep(4, _rb(2)))


def test_default_basis_with_inner_is_g_orthonormal() -> None:
    V = C.base("V", 2)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))  # the metric is a Term (a Var), not a sentinel
    sp = C.with_inner(V, g)
    db, cands = space_basis(sp)
    # default is the g-orthonormal WithInnerBasis (metric becomes identity in coords).
    assert isinstance(db, C.WithInnerBasis)
    # candidate: reuse the underlying standard basis (no whitening cob).
    assert any(_basis_eq(c, _rb(2)) for c in cands)


def test_default_basis_linear_map_space() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    sp = C.linear_map_space(V, W)  # ~V * W
    db, _ = space_basis(sp)
    assert _basis_eq(db, C.basis_prod(C.dual_basis(_rb(2)), _rb(3)))


def test_default_basis_img_is_orthonormal_range() -> None:
    V, W = C.base("V", 3), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    db, cands = space_basis(C.img(f, source=V, target=W))
    # The natural basis of Img(f) is now the ORTHONORMAL range (Q/SVD-U) —
    # ProjectedBasis(f, "image") — not the literal map-application ImageBasis (`73 T2`).
    assert isinstance(db, C.ProjectedBasis) and db.which == "image"
    # the literal map-application ImageBasis(f) is offered as a candidate alternative.
    assert any(isinstance(c, C.ImageBasis) for c in cands)
    # candidate includes the codomain's standard basis (numeric-result-wins foil).
    assert any(_basis_eq(c, _rb(4)) for c in cands)


def test_default_basis_ffs_projected() -> None:
    V, W = C.base("V", 3), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    db_k, _ = space_basis(C.ker(f, source=V, target=W))
    assert isinstance(db_k, C.ProjectedBasis) and db_k.which == "ker"
    db_ci, _ = space_basis(C.coimg(f, source=V, target=W))
    assert db_ci.which == "coimg"
    db_ck, _ = space_basis(C.coker(f, source=V, target=W))
    assert db_ck.which == "coker"


def test_default_basis_svar_is_bvar() -> None:
    db, _ = space_basis(C.svar("α"))
    assert isinstance(db, C.BVar)


# ===========================================================================
# cob — change of basis, run through polyarray.Program.run (`41 §2`)
# ===========================================================================


# Explicit bases of ℝⁿ are now ``ListBasis`` (the term basis); cob through them runs at the
# represent level (their vectors are Terms realized via `_rep`).  ``A``/``Bm`` are the basis
# columns in standard coords; ``cob(A→B) = pinv(C_B)·C_A = Bm⁻¹·A`` for square invertible bases.
_A = np.array([[1.0, 1.0], [0.0, 2.0]])
_BM = np.array([[2.0, 0.0], [1.0, 1.0]])
_EXP = np.linalg.solve(_BM, _A)  # = Bm⁻¹·A


def test_cob_identical_basis_is_identity() -> None:
    b, feeds = _listb("R2", _A)
    np.testing.assert_allclose(_cob_run(b, b, feeds), np.eye(2))


def test_cob_base_to_base_matches_known_matrix() -> None:
    ba, fa = _listb("a", _A)
    bb, fb = _listb("b", _BM)
    np.testing.assert_allclose(_cob_run(ba, bb, {**fa, **fb}), _EXP, atol=1e-9)


def test_cob_to_standard_is_realize_matrix() -> None:
    ba, fa = _listb("a", _A)
    np.testing.assert_allclose(_cob_run(ba, _rb(2), fa), _A, atol=1e-9)


def test_cob_basisprod_is_kron() -> None:
    ba, fa = _listb("a", _A)
    bb, fb = _listb("b", _BM)
    bf = C.basis_prod(ba, _rb(2))
    bt = C.basis_prod(bb, _rb(2))
    got = _cob_run(bf, bt, {**fa, **fb})
    np.testing.assert_allclose(got, np.kron(_EXP, np.eye(2)), atol=1e-9)


def test_cob_basissum_is_block_diag() -> None:
    ba, fa = _listb("a", _A)
    bb, fb = _listb("b", _BM)
    bf = C.basis_sum(ba, _rb(2))
    bt = C.basis_sum(bb, _rb(2))
    got = _cob_run(bf, bt, {**fa, **fb})
    np.testing.assert_allclose(got, _block_diag(_EXP, np.eye(2)), atol=1e-9)


def test_cob_dualbasis_is_inv_transpose() -> None:
    ba, fa = _listb("a", _A)
    bb, fb = _listb("b", _BM)
    got = _cob_run(C.dual_basis(ba), C.dual_basis(bb), {**fa, **fb})
    np.testing.assert_allclose(got, np.linalg.inv(_EXP).T, atol=1e-9)


def test_cob_basisrep_is_block_repeat() -> None:
    ba, fa = _listb("a", _A)
    bb, fb = _listb("b", _BM)
    got = _cob_run(C.basis_rep(2, ba), C.basis_rep(2, bb), {**fa, **fb})
    np.testing.assert_allclose(got, _block_diag(_EXP, _EXP), atol=1e-9)


def test_cob_roundtrip_is_inverse() -> None:
    ba, fa = _listb("a", _A)
    bb, fb = _listb("b", _BM)
    feeds = {**fa, **fb}
    m = _cob_run(ba, bb, feeds)
    mback = _cob_run(bb, ba, feeds)
    np.testing.assert_allclose(m @ mback, np.eye(2), atol=1e-9)


def test_listbasis_with_backup_completes_to_ambient() -> None:
    # A short ℝ² term frame (e0) completed from a backup (e1): the new idiom — a `ListBasis` is
    # always a full basis of its ambient, so a short frame is completed from its backup.
    # cob to standard = [e0|e1] = I.
    sub, fs = _listb("sub", np.array([[1.0], [0.0]]))
    comp, fc = _listb("ce", np.array([[0.0], [1.0]]))
    cb = C.list_basis(*sub.vectors, backup=comp, reduce=C.Reduce.RREF)
    np.testing.assert_allclose(_cob_run(cb, _rb(2), {**fs, **fc}), np.eye(2), atol=1e-9)


# ===========================================================================
# FFS realization — injective ⇒ QrOp, metric ⇒ GSvdOp, generic ⇒ SvdOp (`41 §1`)
# ===========================================================================


def test_injective_image_basis_uses_qrop() -> None:
    rng = np.random.default_rng(0)
    fn = rng.standard_normal((4, 2))  # cod=4, dom=2, injective
    prog = pa.Program("qr")
    F, vals = _sym_matrix(prog, fn, "f")
    real = realize_ffs_basis(prog, "img", F, dom_dim=2, cod_dim=4, props=P.inj, label="f")
    assert real.op == "QrOp"
    assert real.rank_dim is None  # static rank = |dom| (no δ_f)
    # the emitted op really is a QrOp.
    assert any(isinstance(s.fn, pa.QrOp) for s in prog.statements)
    prog.add_output("cols", real.cols)
    Q = prog.run(vals)["cols"]
    assert Q.shape == (4, 2)
    np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-9)


def test_generic_image_basis_uses_svdop_dynamic_rank() -> None:
    rng = np.random.default_rng(1)
    M = rng.standard_normal((4, 2)) @ rng.standard_normal((2, 4))  # rank 2, 4x4
    prog = pa.Program("svd")
    F, vals = _sym_matrix(prog, M, "g")
    real = realize_ffs_basis(prog, "img", F, dom_dim=4, cod_dim=4, label="g")
    assert real.op == "SvdOp"
    assert isinstance(real.rank_dim, pa.DimAtom)  # runtime δ_f
    assert any(isinstance(s.fn, pa.SvdOp) for s in prog.statements)
    prog.add_output("img", real.cols)
    Im = prog.run(vals)["img"]
    assert Im.shape == (4, 2)  # δ_f resolved to 2 at run time
    np.testing.assert_allclose(Im.T @ Im, np.eye(2), atol=1e-9)


def test_generic_kernel_basis_in_kernel() -> None:
    rng = np.random.default_rng(2)
    M = rng.standard_normal((4, 2)) @ rng.standard_normal((2, 4))  # rank 2
    prog = pa.Program("svdker")
    F, vals = _sym_matrix(prog, M, "h")
    real = realize_ffs_basis(prog, "ker", F, dom_dim=4, cod_dim=4, label="h")
    prog.add_output("ker", real.cols)
    K = prog.run(vals)["ker"]
    assert K.shape == (4, 2)  # |V| − rank = 4 − 2
    np.testing.assert_allclose(M @ K, 0.0, atol=1e-8)  # columns are in ker f


def test_metric_image_basis_uses_gsvdop() -> None:
    rng = np.random.default_rng(3)
    M = rng.standard_normal((5, 2)) @ rng.standard_normal((2, 4))  # cod=5 dom=4 rank2

    def spd(k: int) -> np.ndarray:
        X = rng.standard_normal((k, k))
        return X @ X.T + k * np.eye(k)

    Mv, Mw = spd(4), spd(5)
    prog = pa.Program("gsvd")
    F, vf = _sym_matrix(prog, M, "a")
    MV, vv = _sym_matrix(prog, Mv, "mv")
    MW, vw = _sym_matrix(prog, Mw, "mw")
    real = realize_ffs_basis(
        prog, "img", F, dom_dim=4, cod_dim=5, metric_dom=MV, metric_cod=MW, label="a",
    )
    assert real.op == "GSvdOp"
    prog.add_output("img", real.cols)
    U = prog.run({**vf, **vv, **vw})["img"]
    assert U.shape == (5, 2)
    # orthonormal in the codomain metric M_W (the g-orthonormal payoff).
    np.testing.assert_allclose(U.T @ Mw @ U, np.eye(2), atol=1e-8)


def test_metric_kernel_basis_in_kernel() -> None:
    rng = np.random.default_rng(4)
    M = rng.standard_normal((5, 2)) @ rng.standard_normal((2, 4))

    def spd(k: int) -> np.ndarray:
        X = rng.standard_normal((k, k))
        return X @ X.T + k * np.eye(k)

    Mv, Mw = spd(4), spd(5)
    prog = pa.Program("gsvdker")
    F, vf = _sym_matrix(prog, M, "b")
    MV, vv = _sym_matrix(prog, Mv, "mv")
    MW, vw = _sym_matrix(prog, Mw, "mw")
    real = realize_ffs_basis(
        prog, "ker", F, dom_dim=4, cod_dim=5, metric_dom=MV, metric_cod=MW, label="b",
    )
    prog.add_output("ker", real.cols)
    K = prog.run({**vf, **vv, **vw})["ker"]
    assert K.shape == (4, 2)
    np.testing.assert_allclose(M @ K, 0.0, atol=1e-8)


def test_ffs_two_ranks_two_shapes() -> None:
    """Same FFS program, two inputs of different numerical rank ⇒ two output shapes."""

    def img_program() -> pa.Program:
        prog = pa.Program("dyn")
        cells = np.empty((4, 4), dtype=object)
        for idx in np.ndindex(4, 4):
            cells[idx] = pa.RationalFunction.atom("M" + "_".join(map(str, idx)))
        F = pa.SymArray(cells, program=prog)
        real = realize_ffs_basis(prog, "img", F, dom_dim=4, cod_dim=4, label="M")
        prog.add_output("img", real.cols)
        return prog

    def vals(M: np.ndarray) -> dict[str, float]:
        return {"M" + "_".join(map(str, idx)): float(M[idx]) for idx in np.ndindex(4, 4)}

    rng = np.random.default_rng(5)
    prog = img_program()
    M2 = rng.standard_normal((4, 2)) @ rng.standard_normal((2, 4))
    M3 = rng.standard_normal((4, 3)) @ rng.standard_normal((3, 4))
    s2 = prog.run(vals(M2))["img"].shape
    s3 = prog.run(vals(M3))["img"].shape
    assert s2 == (4, 2)
    assert s3 == (4, 3)
    assert s2 != s3


# ===========================================================================
# which="image" — the Q/orthonormal range owns the SVD/QR realization (`73 Topic 2`)
# ===========================================================================


def test_image_which_is_alias_of_img() -> None:
    """``ProjectedBasis``' ``which="image"`` realizes identically to the legacy ``"img"``."""
    rng = np.random.default_rng(7)
    fn = rng.standard_normal((4, 2))  # cod=4, dom=2, injective
    prog_a = pa.Program("a")
    Fa, va = _sym_matrix(prog_a, fn, "f")
    ra = realize_ffs_basis(prog_a, "image", Fa, dom_dim=2, cod_dim=4, props=P.inj, label="f")
    prog_a.add_output("cols", ra.cols)
    prog_b = pa.Program("b")
    Fb, vb = _sym_matrix(prog_b, fn, "f")
    rb = realize_ffs_basis(prog_b, "img", Fb, dom_dim=2, cod_dim=4, props=P.inj, label="f")
    prog_b.add_output("cols", rb.cols)
    assert ra.op == rb.op == "QrOp"
    np.testing.assert_allclose(prog_a.run(va)["cols"], prog_b.run(vb)["cols"], atol=1e-12)


def test_projected_image_injective_is_qrop_orthonormal_range() -> None:
    """``which="image"`` on an injective map ⇒ ``QrOp`` Q (the recovered IsoSpace)."""
    rng = np.random.default_rng(8)
    fn = rng.standard_normal((4, 2))
    prog = pa.Program("qr-image")
    F, vals = _sym_matrix(prog, fn, "f")
    real = realize_ffs_basis(prog, "image", F, dom_dim=2, cod_dim=4, props=P.inj, label="f")
    assert real.op == "QrOp" and real.rank_dim is None  # static rank = |dom|
    prog.add_output("Q", real.cols)
    Q = prog.run(vals)["Q"]
    assert Q.shape == (4, 2)
    np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-9)
    # Q spans the same range as the literal map columns f(eᵢ).
    span_q = Q @ Q.T
    span_f = fn @ np.linalg.pinv(fn)
    np.testing.assert_allclose(span_q, span_f, atol=1e-9)


def test_image_basis_literal_map_application_columns() -> None:
    """``ImageBasis(f)`` realizes the LITERAL ``f(eᵢ)`` columns — NOT orthonormalized."""
    from grassmann.lower.space_basis import realize_image_application

    rng = np.random.default_rng(9)
    fn_cd = rng.standard_normal((4, 2))  # (cod, dom): columns are f(e_i)
    prog = pa.Program("imgapp")
    F, vals = _sym_matrix(prog, fn_cd, "f")
    real = realize_image_application(prog, F, dom_dim=2, cod_dim=4, label="f")
    assert real.op == "MapApply" and real.rank is None and real.rank_dim is None
    prog.add_output("cols", real.cols)
    cols = prog.run(vals)["cols"]
    # columns ARE the (cod,dom) map matrix — the raw f(e_i), not orthonormal.
    np.testing.assert_allclose(cols, fn_cd, atol=1e-12)
    assert not np.allclose(cols.T @ cols, np.eye(2), atol=1e-6)


def test_image_basis_to_projected_image_cob_edge() -> None:
    """The ``ImageBasis ↔ ProjectedBasis(image)`` change of basis is realizable (`73 T2`).

    For an injective ``f`` the literal columns ``R_img = f(eᵢ)`` and the orthonormal range
    ``Q`` are related by ``Q = R_img · M`` for a square ``M`` (here the QR ``R⁻¹``); the
    realized columns must span the same subspace.
    """
    from grassmann.lower.space_basis import realize_image_application

    rng = np.random.default_rng(10)
    fn_cd = rng.standard_normal((4, 2))
    prog = pa.Program("edge")
    F, vals = _sym_matrix(prog, fn_cd, "f")
    img_app = realize_image_application(prog, F, dom_dim=2, cod_dim=4, label="f")
    q_real = realize_ffs_basis(prog, "image", F, dom_dim=2, cod_dim=4, props=P.inj, label="f")
    prog.add_output("R", img_app.cols)
    prog.add_output("Q", q_real.cols)
    out = prog.run(vals)
    R, Q = out["R"], out["Q"]
    # both span the range of f: projector R·R⁺ == Q·Qᵀ.
    np.testing.assert_allclose(R @ np.linalg.pinv(R), Q @ Q.T, atol=1e-9)


# ===========================================================================
# choose — symbolic-aware basis choice heuristic (`41 §1b`)
# ===========================================================================


def test_choose_absorbs_operator_structure() -> None:
    V, W = C.base("V", 3), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    sp = C.img(f, source=V, target=W)
    _, cands = space_basis(sp)
    chosen = choose(sp, cands, term=f)
    # the operator's own image (SVD/Q) basis wins — coordinates become numeric there.
    # That is now ProjectedBasis(f, "image") (the orthonormal range), `73 Topic 2`.
    assert isinstance(chosen, C.ProjectedBasis) and chosen.which == "image"
    assert chosen.f is f


def test_choose_prefers_rational_image_frame_under_propagated_lane() -> None:
    """F1 — the fold-preserving scorer, gated on the propagated rational-frame lane.

    At an un-pinned INJECTIVE ``Img(f)`` leaf: OFF (default) the orthonormalizing
    ``ProjectedBasis`` (QR/SVD) wins — byte-identical to history; ON (the ``literal_ffs_frames``
    lane the exact/certificate consumer propagates) the RATIONAL literal ``ImageBasis`` wins, so
    the algebraic-irrational QR ``R`` never enters and polyarray's exact lane survives.  A
    NON-injective image (rank-deficient literal columns) keeps the orthonormal default on BOTH
    lanes (nothing rational to prefer).
    """
    from grassmann.lower.represent import literal_ffs_frames

    V, W = C.base("V", 2), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W), P.inj)          # an INJECTIVE immersion
    sp = C.img(f, source=V, target=W)
    _, cands = space_basis(sp)
    off = choose(sp, cands, term=f)
    assert isinstance(off, C.ProjectedBasis) and off.which == "image"   # numeric default unmoved
    with literal_ffs_frames(True):
        on = choose(sp, cands, term=f)
    assert isinstance(on, C.ImageBasis) and (on.f is f)                 # rational frame, fold survives

    # a non-injective image has no rational alternative ⇒ orthonormal default on both lanes.
    g = C.var("g", C.linear_map_space(C.base("A", 3), C.base("B", 3)))  # no inj assertion
    spg = C.img(g)
    _, candsg = space_basis(spg)
    with literal_ffs_frames(True):
        assert isinstance(choose(spg, candsg, term=g), C.ProjectedBasis)


def test_propagated_literal_frames_drops_the_qr_radical() -> None:
    """F2 — the artifact: an injective-image lowering emits the QR radical by DEFAULT and NONE
    of it under the propagated rational lane, with the SAME physical result (`assert-on-the-artifact`).

    ``project(v, Img(J))`` on a live injective ``J`` realizes ``Img(J)``'s frame.  Default =
    ``QrOp`` (the fold-blocking radical); ``represent(literal_frames=True)`` = zero ``QrOp`` (the
    rational literal/Gram frame).  Both reconstruct the SAME ambient projection (basis-independent),
    so the change is a frame swap, not a value change.
    """
    import grassmann as G
    from grassmann.lower.represent import represent as g_represent

    rng = np.random.default_rng(3)
    Rt, Re = G.space("Rt", 2), G.space("Re", 4)
    J = G.var("J", Rt >> Re, props=G.inj)
    v = G.var("v", Re)
    term = G.project(v, G.img(J)).node

    def _qr(prog: pa.Program) -> int:
        return sum(1 for st in prog.statements if "Qr" in type(st.fn).__name__)

    p_off = g_represent(term)
    p_on = g_represent(term, literal_frames=True)
    assert _qr(p_off) >= 1, "the default injective-image frame should realize a QR"
    assert _qr(p_on) == 0, "the propagated rational lane must emit NO QR radical"

    Jv, vv = rng.standard_normal((2, 4)), rng.standard_normal(4)
    A = Jv.T                                                  # literal image columns (edim, tdim)
    amb_off = A @ np.linalg.pinv(A) @ vv                     # oracle: orthogonal projection onto Img(J)
    fk = {f"J_{J.node.uid}": Jv, f"v_{v.node.uid}": vv}
    c_off = np.asarray(p_off.run(fk)["result"], float).ravel()
    c_on = np.asarray(p_on.run(fk)["result"], float).ravel()
    from chartlib.covariant import qr_sign_fix
    Q, _ = qr_sign_fix(Jv.T)
    np.testing.assert_allclose(Q @ c_off, amb_off, atol=1e-9)   # orthonormal-coords reconstruction
    np.testing.assert_allclose(A @ c_on, amb_off, atol=1e-9)    # ImageBasis-coords reconstruction


def test_choose_parametric_in_out_same_basis() -> None:
    V = C.base("V", 3)
    sp = C.oprod(C.dual(V), V)  # an endomorphism's space
    db, cands = space_basis(sp)
    chosen = choose(sp, cands, in_basis=db)
    # same basis for both sides ⇒ no spurious cob (no spurious symbolics).
    assert _basis_eq(chosen, db)


def test_choose_defaults_to_orthonormal_standard() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    sp = C.oprod(V, W)
    db, cands = space_basis(sp)
    # absent operator structure / pinned input, the orthonormal default wins.
    assert _basis_eq(choose(sp, cands), db)


def test_score_orthonormal_default_positive() -> None:
    V = C.base("V", 2)
    sp = C.oprod(V, V)
    db = default_basis(sp)
    assert score_basis(sp, db) > 0


def test_choose_empty_candidates_raises() -> None:
    with pytest.raises(ValueError):
        choose(C.scalar(), ())


def test_candidate_bases_includes_default() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    sp = C.oprod(V, W)
    cands = candidate_bases(sp)
    assert any(_basis_eq(c, default_basis(sp)) for c in cands)
