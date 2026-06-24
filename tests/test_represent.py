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

import os

import numpy as np
import polyarray as pa

import grassmann.core as C
import grassmann.core.props as P
from grassmann.core.kinds import SpaceSort
from grassmann.lower import represent
from grassmann.lower.provenance import PROV_KINDS


def _llam_eye_mode() -> bool:
    """True when the LLam lowering is the default fused-``eye`` einsum (not ``vmap``, `41 §7`)."""
    return os.environ.get("GRASSMANN_LLAM_INDEX", "vmap") != "vmap"


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

    # The body contraction is the single deferred TᵀBT einsum (`41 §7`) — an EYE-mode IR shape.
    # Under the Path-B ``GRASSMANN_LLAM_INDEX=vmap`` lowering the body is a sub-Program + vmap call
    # (no fused free-axis einsum), so this structural check applies only to the default eye mode.
    if _llam_eye_mode():
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
    # ``llam:basis`` is the EYE-mode binding-const note; the vmap lowering tags its eye consts
    # ``llam:vmap:eye`` instead.  Accept either so the test is mode-agnostic.
    assert any(n.startswith("llam:basis") or n.startswith("llam:vmap") for n in notes)
    # The body contraction is a TOP-level stmt only under eye; under vmap it lives in the body
    # sub-Program (the top program just holds the eye consts + the vmap CallOp).
    if _llam_eye_mode():
        assert any(n.startswith("eval:contract") for n in notes)


# ===========================================================================
# (1) NEval → sub-Program / CallOp packaging (`41 §3` "Lam → sub-Program").
# ===========================================================================


def _eval_lam(V: C.Space, W: C.Space) -> tuple[C.Term, C.Var, C.Var]:
    """A Lam ``λ x:V. f(x)`` (f : V ⊸ W a free var) — a dependent function."""
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    lam = C.lam(((x.name, SpaceSort(V)),), C.eval_(f, (0,), x), (x,))
    return lam, f, x


def test_neval_lowers_via_subprogram_and_runs() -> None:
    # An NEval of a Lam now lowers via the §3 Lam→sub-Program path (a Stmt whose fn is the
    # body's own sub-Program), and runs to the right numbers.
    V, W = C.base("V", 2), C.base("W", 3)
    lam, f, _x = _eval_lam(V, W)
    a = C.var("a", V)
    prog = represent(C.neval(lam, (0,), a))

    # the body was packaged as a sub-Program invoked by a CallOp Stmt.
    call_stmts = [s for s in prog.statements if isinstance(s.fn, pa.CallOp)]
    assert len(call_stmts) == 1
    assert isinstance(call_stmts[0].fn.fn, pa.Program)
    assert any(s.note.startswith("neval:call-subprogram") for s in prog.statements)

    fv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # f stored (|V|,|W|)
    av = np.array([2.0, -1.0])
    res = prog.run({_var_key(f): fv, _var_key(a): av})["result"]
    np.testing.assert_allclose(res, fv.T @ av)


def test_nested_neval_lam_inside_lam_body_runs() -> None:
    # A nested NEval(Lam([y], …)) sitting INSIDE another Lam's body: the inner binder ``y`` must not
    # leak as a free var (and thus an un-fed input) of the outer sub-Program.  Regression for the
    # _free_term_vars binder-scoping fix (surfaced by differentiating f(g(x)) through value-Lams).
    V = C.base("V", 3)
    x = C.var("x", V)
    y = C.var("y", V)
    s = C.var("s", C.Scalar())
    inner = C.neval(C.lam(((y.name, SpaceSort(V)),), C.eval_(C.inner(V), (0, 1), y, y), (y,)), (0,), x)
    c = C.var("c", C.Scalar())
    outer = C.neval(
        C.lam(((s.name, SpaceSort(C.Scalar())),), C.sum_(s, inner), (s,)), (0,), c
    )  # NEval(λs. s + NEval(λy.⟨y,y⟩, x), c) = c + |x|²
    xv, cv = np.array([1.0, 2, 3]), np.array(5.0)
    res = represent(outer).run({_var_key(x): xv, _var_key(c): cv})["result"]
    np.testing.assert_allclose(res, cv + xv @ xv)


def test_reused_dependent_function_shares_one_subprogram() -> None:
    # A reused Lam (the "reused dependent function" case the inline path can't package
    # cleanly) is lowered ONCE into a shared sub-Program and *called* twice.
    V, W = C.base("V", 2), C.base("W", 3)
    lam, f, _x = _eval_lam(V, W)
    a, b = C.var("a", V), C.var("b", V)
    prog = represent(C.sum_(C.neval(lam, (0,), a), C.neval(lam, (0,), b)))

    call_stmts = [s for s in prog.statements if isinstance(s.fn, pa.CallOp)]
    assert len(call_stmts) == 2
    # both calls target the SAME sub-Program object (body lowered once, not duplicated).
    assert len({id(s.fn.fn) for s in call_stmts}) == 1

    fv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    av, bv = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    res = prog.run({_var_key(f): fv, _var_key(a): av, _var_key(b): bv})["result"]
    np.testing.assert_allclose(res, fv.T @ av + fv.T @ bv)


# ===========================================================================
# (2) Structured Provenance — every emitted Stmt carries node+basis+why (`41 §6b`).
# ===========================================================================


def test_stmts_carry_structured_provenance() -> None:
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    prog = represent(C.eval_(f, (0,), x))

    # every emitted Stmt carries a *structured* polyarray Provenance (not just a string).
    provs = [s.provenance for s in prog.statements]
    assert all(p is not None for p in provs)
    assert all(isinstance(p, pa.Provenance) for p in provs)
    # the algebra-centric kind vocabulary (`41 §6b`), NOT the FEM sampler vocabulary.
    assert all(p.kind in PROV_KINDS for p in provs)
    # the eval contraction's provenance records the node kind + the contraction reason.
    eval_provs = [p for p in provs if p.kind == "eval"]
    assert eval_provs
    ep = eval_provs[0]
    assert ep.origin == "Eval"
    assert "contract" in ep.label  # the "why" rides in the label.


def test_structured_provenance_records_basis_choice() -> None:
    # An FFS realization Stmt's provenance records the basis CHOICE (+ why), the `41 §6b`
    # auditable "which coordinate choice produced this code".
    V, W = C.base("V", 2), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W), props=P.inj)
    imgf = C.img(f, source=V, target=W)
    w = C.var("w", W)
    prog = represent(C.project(w, imgf))

    ffs = [s.provenance for s in prog.statements if s.provenance.kind == "ffs-basis"]
    assert ffs
    # the basis choice ("ImageBasis(...)") and why ("inj=>QrOp") are recorded in the label.
    assert any("ImageBasis" in p.label and "QrOp" in p.label for p in ffs)


def test_provenance_survives_post_pass_partial_eval() -> None:
    # Structured provenance must survive the `41 §5` partial_eval post-pass (which copies
    # the program); a regression here would silently drop traceability.
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=P.bij)
    prog = represent(C.solve(f))
    assert all(s.provenance is not None for s in prog.statements)
    assert any(s.provenance.kind == "solve" for s in prog.statements)


# ===========================================================================
# (3) Static-axis per-contraction runtime AssertOp (`41 §4`, belt-and-suspenders).
# ===========================================================================


def _axis_len_op():  # noqa: ANN202 - test helper
    from grassmann.lower.represent import _AxisLenOp

    return _AxisLenOp


def test_static_contraction_emits_runtime_rank_eq_assert() -> None:
    # A purely-STATIC Eval contraction now also emits a runtime rank_eq AssertOp (`41 §4`),
    # not only the lowering-time static check.
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    prog = represent(C.eval_(f, (0,), x))
    rank_asserts = [
        s for s in prog.statements
        if isinstance(s.fn, pa.AssertOp) and s.fn.kind == "rank_eq"
    ]
    assert rank_asserts, "static Eval should still emit a runtime rank_eq AssertOp"
    # the emitted AssertOp trips on mismatched axis lengths (deliberate mismatch).
    op = rank_asserts[0].fn
    op(np.zeros((2,)), np.asarray(2), np.asarray(2))  # match: passes, returns first input
    with np.testing.assert_raises(AssertionError):
        op(np.zeros((2,)), np.asarray(2), np.asarray(3))  # mismatch: trips


def test_runtime_contraction_assert_trips_through_run() -> None:
    # End-to-end: the same _AxisLenOp + rank_eq machinery emitted at a contraction trips at
    # Program.run time on a deliberately mismatched (hand-built) bulk input.
    AxisLenOp = _axis_len_op()
    da = pa.DimAtom.from_input("da", "a", 0)
    db = pa.DimAtom.from_input("db", "b", 0)
    prov = pa.Provenance(kind="param_dof", origin="x", index=(), label="x")
    sa = pa.SymInput(name="a", shape=(da,), provenance=prov)
    sb = pa.SymInput(name="b", shape=(db,), provenance=prov)
    p = pa.Program("contract", inputs=(sa, sb))
    A, B = p.input("a"), p.input("b")
    [la] = p.emit_stmt(AxisLenOp(0), [A], [pa.OutSpec("la", ())])
    [lb] = p.emit_stmt(AxisLenOp(0), [B], [pa.OutSpec("lb", ())])
    p.emit_stmt(
        pa.AssertOp(kind="rank_eq", msg="assert:eval-dim@0"),
        [A, la, lb], [pa.OutSpec("ok", (da,))],
    )
    p.add_output("out", p.statements[-1].out[0])

    # matched contracted axes: runs fine.
    np.testing.assert_allclose(p.run({"a": np.ones(3), "b": np.ones(3)})["out"], np.ones(3))
    # mismatched: the runtime AssertOp trips.
    with np.testing.assert_raises(AssertionError):
        p.run({"a": np.ones(3), "b": np.ones(4)})


# ===========================================================================
# ListBasis — explicit list-of-Term-vectors basis (`73 Topic 2`, B5)
# ===========================================================================


def test_list_basis_stacks_lowered_vectors_as_columns() -> None:
    """``ListBasis(v0, v1)`` of ℝ³ lowers to a FULL ℝ³ basis whose LEADING columns are
    ⟦v0⟧, ⟦v1⟧ — a short frame is completed to the ambient (`73 T2`)."""
    from grassmann.lower.represent import _Ctx, realize_basis_cols

    V = C.base("V", 3)
    x = C.var("x", V)
    y = C.var("y", V)
    lb = C.list_basis(x, y)
    prog = pa.Program("lb", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("B", realize_basis_cols(ctx, lb))
    res = prog.run({_var_key(x): np.array([1.0, 2.0, 3.0]),
                    _var_key(y): np.array([4.0, 5.0, 6.0])})["B"]
    # A short ℝ³ ListBasis now realizes to a full 3×3 ambient basis (the 2 explicit
    # vectors plus a standard completion).
    assert res.shape == (3, 3)
    np.testing.assert_allclose(res[:, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(res[:, 1], np.array([4.0, 5.0, 6.0]))
    assert abs(np.linalg.det(res)) > 1e-9  # a genuine full basis
    # the column-stack provenance note is present.
    assert any("ListBasis=stack-columns" in n for n in _notes(prog))


def test_list_basis_ravels_multi_axis_vectors() -> None:
    """``ListBasis`` ravels a multi-axis (``V⊗ᵏ``) vector into one ambient column, then
    completes to the full ambient (here ℝ⁴ = V⊗V)."""
    from grassmann.lower.represent import _Ctx, realize_basis_cols

    V = C.base("V", 2)
    x = C.var("x", C.oprod(V, V))  # V⊗V, stored (2,2)
    lb = C.list_basis(x)
    prog = pa.Program("lb2", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("B", realize_basis_cols(ctx, lb))
    xv = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = prog.run({_var_key(x): xv})["B"]
    # 1 explicit raveled column, completed to a full 4×4 ambient basis.
    assert res.shape == (4, 4)
    np.testing.assert_allclose(res[:, 0], xv.reshape(-1))
    assert abs(np.linalg.det(res)) > 1e-9


def test_list_basis_inconsistent_dim_raises() -> None:
    from grassmann.lower.represent import RepresentError, _Ctx, realize_basis_cols

    V, W = C.base("V", 3), C.base("W", 2)
    x = C.var("x", V)
    y = C.var("y", W)
    lb = C.list_basis(x, y)
    prog = pa.Program("lb3", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    with np.testing.assert_raises(RepresentError):
        realize_basis_cols(ctx, lb)


def test_image_basis_node_realizes_literal_map_application() -> None:
    """The LITERAL ``f(eᵢ)`` columns are the ambient immersion ``_ffs_immersion(ImageBasis)``;
    ``realize_basis_cols`` now realizes the FFS basis **in-space** (the square ``R`` of
    ``J = U_d·R``) (`73 T2`, `51 §FFS-δ`)."""
    from grassmann.lower.represent import _Ctx, _ffs_immersion, realize_basis_cols

    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))  # stored (|V|,|W|) = (2,3)
    fmat = np.arange(6.0).reshape(2, 3)  # stored (dom,cod)

    # The ambient immersion is the literal map application: f(eᵢ) = the (cod,dom) matrix = fmatᵀ.
    prog = pa.Program("ib", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("B", _ffs_immersion(ctx, C.image_basis(f)))
    res = prog.run({_var_key(f): fmat})["B"]
    assert res.shape == (3, 2)
    np.testing.assert_allclose(res, fmat.T)
    assert any("ImageBasis" in n and "map-application" in n for n in _notes(prog))

    # In-space realize is the square R (J = U_d·R), so U_d·R reconstructs the immersion J.
    prog2 = pa.Program("ib2", budget=pa.SymbolicBudget.force_stmts())
    ctx2 = _Ctx(prog=prog2, basis_env={}, inputs={})
    prog2.add_output("R", realize_basis_cols(ctx2, C.image_basis(f)))
    r = prog2.run({_var_key(f): fmat})["R"]
    assert r.shape == (2, 2)  # square in-space


# Permute — tensor axis reorder / (0,2)-form transpose (the reordering Lift/InvLift can't do).


def test_permute_transposes_a_form() -> None:
    M = C.base("M", 3)
    b = C.var("B", C.oprod(C.dual(M), C.dual(M)))
    Bv = np.arange(9.0).reshape(3, 3)
    out = represent(C.Permute(b, (1, 0))).run({_var_key(b): Bv})["result"]
    np.testing.assert_allclose(out, Bv.T)


def test_permute_reorders_three_factors() -> None:
    M = C.base("M", 3)
    w = C.var("W", C.oprod(C.dual(M), C.dual(M), M))
    Wv = np.arange(27.0).reshape(3, 3, 3)
    out = represent(C.Permute(w, (2, 0, 1))).run({_var_key(w): Wv})["result"]
    np.testing.assert_allclose(out, Wv.transpose(2, 0, 1))
