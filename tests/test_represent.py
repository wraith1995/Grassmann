"""IW4 tests: the per-node ``representDown`` lowering (`41 §3`).

These are the seeds of acceptance gates G1/G2 (`41 §7`):

* a small matrix program (``Eval``/``Compose``/dot over concrete ``Base`` spaces) lowered
  → ``Program.run`` → matches hand-computed numpy;
* the **pullback** (`41 §7`): the body of ``λT.λb.λx,y. b(Tx,Ty)`` becomes the deferred
  contraction ``EinsumStmt("ab,ai,bj->ij", B, T, T)`` = ``TᵀBT``, verified vs ``T.T@B@T``;
* a ``Pinv``/``Solve`` program lowering to ``PinvOp``/``InvOp`` with the right
  ``AssertOp``, matching ``numpy.linalg.lstsq``/``solve``;
* provenance: emitted ``Stmt``s carry non-empty algebra-centric ``note``s.

**Storage convention (resolved, see report).** A map space ``~s0*s1`` mirrors its space
factors onto array axes ``(|s0|, |s1|) = (|dom|, |cod|)`` (`41 §0`). ``Eval`` contracts
the leading ``Dual`` (input) axis, so a stored map matrix is the transpose of the usual
``(cod, dom)`` matrix; results below are computed against that convention.
"""

from __future__ import annotations

import numpy as np
import polyarray as pa

import grassmann.core as C
import grassmann.core.props as P
from grassmann.lower import represent


def _var_key(v: C.Var) -> str:
    return f"{v.name}_{v.uid}"


def _notes(prog: pa.Program) -> list[str]:
    return [s.note for s in prog.statements]


def _einsum_specs(prog: pa.Program) -> list[str]:
    return [s.fn.spec for s in prog.statements if isinstance(s.fn, pa.EinsumStmtOp)]


# ===========================================================================
# G1 — a small matrix program over concrete Base spaces vs hand numpy.
# ===========================================================================


def test_eval_matvec_matches_numpy() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    prog = represent(C.eval_(f, (0,), x))

    fv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (|V|,|W|)
    xv = np.array([1.0, 1.0])
    res = prog.run({_var_key(f): fv, _var_key(x): xv})["result"]
    np.testing.assert_allclose(res, fv.T @ xv)


def test_compose_matches_numpy() -> None:
    V, W, U = C.base("V", 2), C.base("W", 3), C.base("U", 2)
    f = C.var("f", C.linear_map_space(W, U))
    g = C.var("g", C.linear_map_space(V, W))
    prog = represent(C.compose(f, (0,), g))

    rng = np.random.default_rng(3)
    Fm = rng.standard_normal((3, 2))  # f stored (|W|,|U|)
    Gm = rng.standard_normal((2, 3))  # g stored (|V|,|W|)
    res = prog.run({_var_key(f): Fm, _var_key(g): Gm})["result"]
    # composite stored (|V|,|U|) = Gm @ Fm in this convention.
    np.testing.assert_allclose(res, Gm @ Fm)
    assert "ab,pa->pb" in _einsum_specs(prog)


def test_trace_matches_numpy() -> None:
    V = C.base("V", 4)
    f = C.var("f", C.linear_map_space(V, V))  # ~V * V, an endomorphism
    # Trace contracts a primal index (1, the V factor) with a dual index (0, the ~V slot).
    prog = represent(C.trace(f, 1, 0))
    rng = np.random.default_rng(4)
    A = rng.standard_normal((4, 4))
    res = prog.run({_var_key(f): A})["result"]
    np.testing.assert_allclose(res, np.trace(A))


def test_inner_default_metric_is_identity() -> None:
    V = C.base("V", 3)
    prog = represent(C.inner(V))
    res = prog.run({})["result"]
    # orthonormal default ⇒ G_s = I (the major min-sym win, `41 §1`).
    np.testing.assert_allclose(res, np.eye(3))


def test_sum_and_scale_matches_numpy() -> None:
    V = C.base("V", 3)
    x = C.var("x", V)
    y = C.var("y", V)
    prog = represent(C.sum_(C.scale(2.0, x), y))
    rng = np.random.default_rng(5)
    xv, yv = rng.standard_normal(3), rng.standard_normal(3)
    res = prog.run({_var_key(x): xv, _var_key(y): yv})["result"]
    np.testing.assert_allclose(res, 2.0 * xv + yv)


# ===========================================================================
# G2 — the pullback: body ⇒ deferred EinsumStmt("ab,ai,bj->ij") = TᵀBT.
# ===========================================================================


def test_pullback_body_is_deferred_tbt_contraction() -> None:
    V, W = C.base("V", 3), C.base("W", 2)
    Tv = C.var("T", C.linear_map_space(V, W))            # T : V ⊸ W
    bv = C.var("b", C.oprod(C.dual(W), C.dual(W)))       # b : bilinear form on W
    xv, yv = C.var("x", V), C.var("y", V)
    Tx = C.eval_(Tv, (0,), xv)
    Ty = C.eval_(Tv, (0,), yv)
    body = C.eval_(bv, (0, 1), Tx, Ty)
    pullback_form = C.llam(xv, yv, body=body)            # the n×n form matrix

    prog = represent(pullback_form)

    # The body contraction is the single deferred TᵀBT einsum (`41 §7`).
    assert "ab,ai,bj->ij" in _einsum_specs(prog)

    rng = np.random.default_rng(0)
    Tm = rng.standard_normal((3, 2))  # T stored (|V|,|W|)
    Bm = rng.standard_normal((2, 2))
    res = prog.run({_var_key(Tv): Tm, _var_key(bv): Bm})["result"]
    assert res.shape == (3, 3)
    # In this storage convention TᵀBT = Tm @ Bm @ Tm.T (= the pullback form on V).
    np.testing.assert_allclose(res, Tm @ Bm @ Tm.T)


def test_pullback_only_T_and_b_are_inputs() -> None:
    V, W = C.base("V", 3), C.base("W", 2)
    Tv = C.var("T", C.linear_map_space(V, W))
    bv = C.var("b", C.oprod(C.dual(W), C.dual(W)))
    xv, yv = C.var("x", V), C.var("y", V)
    body = C.eval_(bv, (0, 1), C.eval_(Tv, (0,), xv), C.eval_(Tv, (0,), yv))
    prog = represent(C.llam(xv, yv, body=body))
    # the linear binder vars x,y do NOT leak as program inputs (`41 §3` LLam→matrix).
    names = {i.name for i in prog.inputs}
    assert names == {_var_key(Tv), _var_key(bv)}


# ===========================================================================
# G2 — Pinv / Solve programs vs numpy, with the right AssertOp.
# ===========================================================================


def test_pinv_apply_matches_lstsq() -> None:
    V, W = C.base("V", 2), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    v = C.var("v", W)
    prog = represent(C.pinv_apply(f, v))
    assert any(isinstance(s.fn, pa.PinvOp) for s in prog.statements)

    rng = np.random.default_rng(2)
    Fm = rng.standard_normal((2, 4))  # f stored (|V|,|W|); real (cod,dom) map = Fm.T
    vv = rng.standard_normal(4)
    res = prog.run({_var_key(f): Fm, _var_key(v): vv})["result"]
    xhat = np.linalg.lstsq(Fm.T, vv, rcond=None)[0]
    np.testing.assert_allclose(res, xhat, atol=1e-9)


def test_pinv_map_matches_numpy_pinv() -> None:
    V, W = C.base("V", 4), C.base("W", 2)
    g = C.var("g", C.linear_map_space(V, W))
    prog = represent(C.pinv(g))
    assert any(isinstance(s.fn, pa.PinvOp) for s in prog.statements)
    rng = np.random.default_rng(6)
    Gm = rng.standard_normal((4, 2))  # stored (|V|,|W|)
    res = prog.run({_var_key(g): Gm})["result"]
    np.testing.assert_allclose(res, np.linalg.pinv(Gm), atol=1e-9)


def test_solve_emits_invop_and_full_rank_assert() -> None:
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=P.bij)
    prog = represent(C.solve(f))
    assert any(isinstance(s.fn, pa.InvOp) for s in prog.statements)
    # the bij D-scope check (`41 §4`).
    assert any(
        isinstance(s.fn, pa.AssertOp) and s.fn.kind == "square_full_rank"
        for s in prog.statements
    )
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    res = prog.run({_var_key(f): A})["result"]
    np.testing.assert_allclose(res, np.linalg.inv(A))


def test_solve_apply_matches_solve() -> None:
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=P.bij)
    b = C.var("b", V)
    prog = represent(C.solve_apply(f, b))
    assert any(isinstance(s.fn, pa.SolveOp) for s in prog.statements)
    rng = np.random.default_rng(7)
    A = rng.standard_normal((3, 3))
    bv = rng.standard_normal(3)
    res = prog.run({_var_key(f): A, _var_key(b): bv})["result"]
    np.testing.assert_allclose(res, np.linalg.solve(A, bv))


# ===========================================================================
# FFS movement — Project onto img(f) via the QrOp seam (injective ⇒ static rank).
# ===========================================================================


def test_project_onto_injective_image_uses_qr_seam() -> None:
    V, W = C.base("V", 2), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W), props=P.inj)
    imgf = C.img(f, source=V, target=W)
    w = C.var("w", W)
    prog = represent(C.project(w, imgf))
    # the injective image basis is realized by QrOp (recovers IsoSpace, `41 §1`).
    assert any(isinstance(s.fn, pa.QrOp) for s in prog.statements)
    assert any(s.note.startswith("ffs-basis:ImageBasis") for s in prog.statements)

    rng = np.random.default_rng(0)
    Fm = rng.standard_normal((2, 4))  # f stored (|V|,|W|); real (cod,dom) map = Fm.T
    wv = rng.standard_normal(4)
    res = prog.run({_var_key(f): Fm, _var_key(w): wv})["result"]
    assert res.shape == (2,)  # static rank |dom| = 2 (no δ_f)
    Q, _ = np.linalg.qr(Fm.T)
    # projected coords are Qᵀw up to QR column-sign convention.
    np.testing.assert_allclose(np.abs(res), np.abs(Q.T @ wv), atol=1e-9)


# ===========================================================================
# Eval — the contraction-dim assert (D-scope) fires on a mismatch.
# ===========================================================================


def test_eval_dim_mismatch_raises_at_lowering() -> None:
    import pytest

    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    # x has the WRONG space (W, dim 3) for f's input slot (V, dim 2).
    bad = C.var("x", W)
    # synth would reject this; build the Eval node directly to exercise represent's check
    # by handing it a structurally-formed but dim-mismatched contraction.
    with pytest.raises(Exception):  # noqa: B017 - either InferError or RepresentError
        represent(C.eval_(f, (0,), bad))


# ===========================================================================
# Provenance — every emitted Stmt carries an algebra-centric note (`41 §6b`).
# ===========================================================================


def test_provenance_notes_are_algebra_centric() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    prog = represent(C.eval_(f, (0,), x))
    notes = _notes(prog)
    # every Stmt has a non-empty note.
    assert all(n for n in notes)
    # and the eval contraction note uses the algebra vocabulary (`41 §6b`).
    assert any(n.startswith("eval:contract") for n in notes)


def test_provenance_pullback_notes() -> None:
    V, W = C.base("V", 3), C.base("W", 2)
    Tv = C.var("T", C.linear_map_space(V, W))
    bv = C.var("b", C.oprod(C.dual(W), C.dual(W)))
    xv, yv = C.var("x", V), C.var("y", V)
    body = C.eval_(bv, (0, 1), C.eval_(Tv, (0,), xv), C.eval_(Tv, (0,), yv))
    prog = represent(C.llam(xv, yv, body=body))
    notes = _notes(prog)
    assert all(n for n in notes)
    assert any(n.startswith("llam:basis") for n in notes)
    assert any(n.startswith("eval:contract") for n in notes)
