"""IW4-substrate tests: ``space_basis`` / ``cob`` / ``choose`` (`41 §1`/`§1b`/`§2`).

The first integration of the Grassmann side with the shipped polyarray additions
(`42`): ``cob``/FFS realization emit real ``SvdOp``/``QrOp``/``GSvdOp``/``DimAtom`` Stmts
and run through ``polyarray.Program.run`` on small numeric examples.
"""

from __future__ import annotations

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
from grassmann.core import props as P
from grassmann.lower import (
    candidate_bases,
    choose,
    cob,
    default_basis,
    realize_ffs_basis,
    score_basis,
    space_basis,
)
from grassmann.lower.cob import _basis_eq

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


def test_default_basis_dual() -> None:
    # ``_basis_eq`` (not ``==``): the default standard basis now carries its space as sizing
    # metadata (`20 §7`), which ``_basis_eq`` treats as identity-irrelevant.
    db, _ = space_basis(C.dual(C.base("V", 3)))
    assert _basis_eq(db, C.dual_basis(C.base_basis("V")))


def test_default_basis_oplus_oprod() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    db_sum, _ = space_basis(C.oplus(V, W))
    assert _basis_eq(db_sum, C.basis_sum(C.base_basis("V"), C.base_basis("W")))
    db_prod, _ = space_basis(C.oprod(V, W))
    assert _basis_eq(db_prod, C.basis_prod(C.base_basis("V"), C.base_basis("W")))


def test_default_basis_oplusrep() -> None:
    db, _ = space_basis(C.oplus_rep(4, C.base("V", 2)))
    assert _basis_eq(db, C.basis_rep(4, C.base_basis("V")))


def test_default_basis_with_inner_is_g_orthonormal() -> None:
    g = object()
    sp = C.with_inner(C.base("V", 2), g)
    db, cands = space_basis(sp)
    # default is the g-orthonormal WithInnerBasis (metric becomes identity in coords).
    assert isinstance(db, C.WithInnerBasis)
    # candidate: reuse the underlying standard basis (no whitening cob).
    assert any(_basis_eq(c, C.base_basis("V")) for c in cands)


def test_default_basis_linear_map_space() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    sp = C.linear_map_space(V, W)  # ~V * W
    db, _ = space_basis(sp)
    assert _basis_eq(db, C.basis_prod(C.dual_basis(C.base_basis("V")), C.base_basis("W")))


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
    assert any(_basis_eq(c, C.base_basis("W")) for c in cands)


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


def test_cob_identical_basis_is_identity() -> None:
    A = np.array([[1.0, 1.0], [0.0, 2.0]])
    b = C.base_basis("R2", A)
    prog = pa.Program("id")
    M = cob(prog, b, b)
    prog.add_output("M", M)
    np.testing.assert_allclose(prog.run({})["M"], np.eye(2))


def test_cob_base_to_base_matches_known_matrix() -> None:
    A = np.array([[1.0, 1.0], [0.0, 2.0]])  # from-basis columns (in std coords)
    Bm = np.array([[2.0, 0.0], [1.0, 1.0]])  # to-basis columns
    prog = pa.Program("cob")
    M = cob(prog, C.base_basis("a", A), C.base_basis("b", Bm))
    prog.add_output("M", M)
    res = prog.run({})
    # std = A @ x = Bm @ y  ⇒  y = Bm⁻¹ A x, so cob = Bm⁻¹ A.
    np.testing.assert_allclose(res["M"], np.linalg.solve(Bm, A), atol=1e-12)


def test_cob_to_standard_is_realize_matrix() -> None:
    A = np.array([[1.0, 1.0], [0.0, 2.0]])
    prog = pa.Program("std")
    M = cob(prog, C.base_basis("a", A), C.base_basis("std", np.eye(2)))
    prog.add_output("M", M)
    np.testing.assert_allclose(prog.run({})["M"], A, atol=1e-12)


def test_cob_basisprod_is_kron() -> None:
    A = np.array([[1.0, 1.0], [0.0, 2.0]])
    Bm = np.array([[2.0, 0.0], [1.0, 1.0]])
    exp = np.linalg.solve(Bm, A)
    bf = C.basis_prod(C.base_basis("a", A), C.base_basis("s", np.eye(2)))
    bt = C.basis_prod(C.base_basis("b", Bm), C.base_basis("s", np.eye(2)))
    prog = pa.Program("prod")
    prog.add_output("M", cob(prog, bf, bt))
    np.testing.assert_allclose(prog.run({})["M"], np.kron(exp, np.eye(2)), atol=1e-12)


def test_cob_basissum_is_block_diag() -> None:
    A = np.array([[1.0, 1.0], [0.0, 2.0]])
    Bm = np.array([[2.0, 0.0], [1.0, 1.0]])
    exp = np.linalg.solve(Bm, A)
    bf = C.basis_sum(C.base_basis("a", A), C.base_basis("s", np.eye(2)))
    bt = C.basis_sum(C.base_basis("b", Bm), C.base_basis("s", np.eye(2)))
    prog = pa.Program("sum")
    prog.add_output("M", cob(prog, bf, bt))
    np.testing.assert_allclose(prog.run({})["M"], _block_diag(exp, np.eye(2)), atol=1e-12)


def test_cob_dualbasis_is_inv_transpose() -> None:
    A = np.array([[1.0, 1.0], [0.0, 2.0]])
    Bm = np.array([[2.0, 0.0], [1.0, 1.0]])
    exp = np.linalg.solve(Bm, A)
    bf = C.dual_basis(C.base_basis("a", A))
    bt = C.dual_basis(C.base_basis("b", Bm))
    prog = pa.Program("dual")
    prog.add_output("M", cob(prog, bf, bt))
    np.testing.assert_allclose(prog.run({})["M"], np.linalg.inv(exp).T, atol=1e-12)


def test_cob_basisrep_is_block_repeat() -> None:
    A = np.array([[1.0, 1.0], [0.0, 2.0]])
    Bm = np.array([[2.0, 0.0], [1.0, 1.0]])
    exp = np.linalg.solve(Bm, A)
    bf = C.basis_rep(2, C.base_basis("a", A))
    bt = C.basis_rep(2, C.base_basis("b", Bm))
    prog = pa.Program("rep")
    prog.add_output("M", cob(prog, bf, bt))
    np.testing.assert_allclose(prog.run({})["M"], _block_diag(exp, exp), atol=1e-12)


def test_cob_roundtrip_is_inverse() -> None:
    A = np.array([[1.0, 1.0], [0.0, 2.0]])
    Bm = np.array([[2.0, 0.0], [1.0, 1.0]])
    ba, bb = C.base_basis("a", A), C.base_basis("b", Bm)
    prog = pa.Program("rt")
    M = cob(prog, ba, bb)
    Mback = cob(prog, bb, ba)
    prog.add_output("M", M)
    prog.add_output("Mback", Mback)
    res = prog.run({})
    np.testing.assert_allclose(res["M"] @ res["Mback"], np.eye(2), atol=1e-12)


def test_cob_completebasis_stacks_sub_and_complement() -> None:
    # sub basis = e0; complement = e1; ambient = R^2.  cob to standard = [sub|comp] = I.
    sub = C.base_basis("sub", np.array([[1.0], [0.0]]))
    comp = np.array([[0.0], [1.0]])
    cb = C.complete_basis(sub, comp)
    prog = pa.Program("complete")
    prog.add_output("M", cob(prog, cb, C.base_basis("std", np.eye(2))))
    np.testing.assert_allclose(prog.run({})["M"], np.eye(2), atol=1e-12)


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
