"""2.3 match decomposition — a node realizes DIRECTLY in a structure-matching target (no cob).

Correctness of `target` is covered by ``test_target_correctness``; here we assert the
*optimization*: when the concrete/STD target matches the node's structure and the operands are
free (unpinned ⇒ declared directly in the sub-target), NO conversion cob is emitted — the result
is built straight in the target.  AUTO stays byte-identical.
"""

import numpy as np
import polyarray as pa

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent
from grassmann.lower.space_basis import default_basis


def _notes(prog):
    return [st.note for sub in pa.forward.iter_programs(prog) for st in sub.statements]


def _no_target_cob(prog):
    return not any(n.startswith("cob:target") for n in _notes(prog))


def _nstmt(prog):
    return sum(1 for sub in pa.forward.iter_programs(prog) for _ in sub.statements)


def _run(prog, feeds):
    return np.asarray(prog.run(feeds)["result"], float)


def _vk(v):
    return f"{v.name}_{v.uid}"


# ---------------------------------------------------------------------------
# Structural split — OProdV / OPlusV build each factor in its sub-target.
# ---------------------------------------------------------------------------


def test_oprod_splits_a_basisprod_target_no_cob():
    V = C.base("V", 3)
    u, w = C.var("u", V), C.var("w", V)  # unpinned ⇒ declared directly in the sub-target
    op = C.oprod_v(u, w)
    uc, wc = np.array([1.0, 0, 2]), np.array([0.0, 1, 3])
    feeds = {_vk(u): uc, _vk(w): wc}
    prog = represent(op, target=B.STD)  # default_basis(OProd) is BasisProd(std, std)
    assert _no_target_cob(prog)
    assert np.allclose(_run(prog, feeds), np.outer(uc, wc))


def test_oplus_splits_a_basissum_target_no_cob():
    V, W = C.base("V", 3), C.base("W", 2)
    x, y = C.var("x", W), C.var("y", V)
    op = C.oplus_v(x, y)
    xc, yc = np.array([1.0, 2]), np.array([3.0, 4, 5])
    feeds = {_vk(x): xc, _vk(y): yc}
    prog = represent(op, target=B.STD)
    assert _no_target_cob(prog)
    assert np.allclose(_run(prog, feeds), np.concatenate([xc, yc]))


# ---------------------------------------------------------------------------
# Direct realize — Zero (any basis), Identity (same-basis endo pair).
# ---------------------------------------------------------------------------


def test_zero_is_direct_in_any_target():
    V = C.base("V", 3)
    a = [C.var(f"a{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    Am = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    feeds = {f"a{i}_{a[i].uid}": Am[:, i] for i in range(3)}
    prog = represent(C.zero(V), target=lb)
    assert _no_target_cob(prog)
    assert np.allclose(_run(prog, feeds), 0.0)


def test_identity_is_direct_in_a_same_basis_endo_pair():
    V = C.base("V", 3)
    a = [C.var(f"a{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    Am = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    feeds = {f"a{i}_{a[i].uid}": Am[:, i] for i in range(3)}
    prog = represent(C.identity(V), target=B.basis_prod(B.dual_basis(lb), lb))
    assert _no_target_cob(prog)
    assert np.allclose(_run(prog, feeds), np.eye(3))


# ---------------------------------------------------------------------------
# Push-down — Remetric realizes the operand in a WithInner target's underlying basis.
# ---------------------------------------------------------------------------


def test_remetric_pushes_a_withinner_target_down():
    V = C.base("V", 3)
    u = C.var("u", V)  # unpinned
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    prog = represent(C.remetric(u, g), target=B.with_inner_basis(default_basis(V), g))
    assert _no_target_cob(prog)


# ---------------------------------------------------------------------------
# AUTO byte-identical: the decomposition arms are gated on a concrete/STD target.
# ---------------------------------------------------------------------------


def test_auto_unchanged_by_the_decomposition_arms():
    V, W = C.base("V", 3), C.base("W", 2)
    u, w = C.var("u", V), C.var("w", V)
    x, y = C.var("x", W), C.var("y", V)
    for t in (C.oprod_v(u, w), C.oplus_v(x, y), C.zero(V), C.identity(V)):
        assert _nstmt(represent(t)) == _nstmt(represent(t, target=B.AUTO))


# ---------------------------------------------------------------------------
# Operator push-down: Trace steers f so the surviving slots land in the target.
# ---------------------------------------------------------------------------


def test_trace_pushes_target_into_surviving_slot():
    V, W = C.base("V", 3), C.base("W", 2)
    a, b, c = C.var("a", V), C.var("b", C.dual(V)), C.var("c", W)
    t = C.trace(C.oprod_v(a, b, c), 0, 1)  # contract V⊗V* ⇒ ⟨a,b⟩·c, surviving slot = W
    ac, bc, cc = np.array([1.0, 2, 0]), np.array([0.0, 1, 3]), np.array([2.0, 1])
    std_result = (ac @ bc) * cc
    # operands pinned to std (input contract fixed); a non-default W target steers the surviving slot
    env = {a.uid: default_basis(V), b.uid: default_basis(C.dual(V)), c.uid: default_basis(W)}
    e = [C.var(f"e{i}", W) for i in range(2)]
    lbW = B.list_basis(*e)
    Wm = np.array([[1.0, 1], [0, 1.0]])
    feeds = {_vk(a): ac, _vk(b): bc, _vk(c): cc, **{f"e{i}_{e[i].uid}": Wm[:, i] for i in range(2)}}
    assert np.allclose(_run(represent(t, basis_env=env, target=lbW), feeds),
                       np.linalg.solve(Wm, std_result))  # result re-expressed into lbW
    assert np.allclose(_run(represent(t, basis_env=env, target=B.STD), feeds), std_result)
    # with default operands + STD target the surviving slot is steered ⇒ no result-level cob
    assert _no_target_cob(represent(t, target=B.STD))


# ---------------------------------------------------------------------------
# Metric-in-target: Inner/Flat build the Gram directly in a concrete target basis.
# ---------------------------------------------------------------------------


def test_inner_flat_build_gram_in_a_concrete_target():
    V = C.base("V", 3)
    a = [C.var(f"e{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    Am = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    feeds = {f"e{i}_{a[i].uid}": Am[:, i] for i in range(3)}
    gram = Am.T @ Am  # the Gram of lb under the standard metric (G = I)
    want = B.basis_prod(B.dual_basis(lb), B.dual_basis(lb))  # the metric's two covariant slots in lb
    for t in (C.inner(V), C.flat(V)):
        prog = represent(t, target=want)
        assert _no_target_cob(prog)  # built directly in `want` — no conversion cob
        assert np.allclose(_run(prog, feeds), gram)  # G_s in lb = lbᵀ·G·lb
    # AUTO and STD are unchanged (the steer fires only for a concrete Basis target)
    assert _nstmt(represent(C.inner(V))) == _nstmt(represent(C.inner(V), target=B.AUTO))
    assert _nstmt(represent(C.inner(V), target=B.STD)) == _nstmt(represent(C.inner(V)))


# ---------------------------------------------------------------------------
# Permute push-down: a structural target is pushed into the operand (operand axis j carries want's
# axis inv[j]); the transpose then lands directly in `want` — no post-cob.
# ---------------------------------------------------------------------------


def test_permute_pushes_a_structural_target_into_the_operand():
    V, W = C.base("V", 3), C.base("W", 2)
    a = [C.var(f"e{i}", V) for i in range(3)]
    b = [C.var(f"w{i}", W) for i in range(2)]
    lbV, lbW = B.list_basis(*a), B.list_basis(*b)
    Av = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    Aw = np.array([[1.0, 1], [0, 1.0]])
    frame = {f"e{i}_{a[i].uid}": Av[:, i] for i in range(3)}
    frame |= {f"w{i}_{b[i].uid}": Aw[:, i] for i in range(2)}
    M = np.arange(6.0).reshape(3, 2)  # the std operand V⊗W
    op = C.var("op", C.oprod(V, W))
    t = C.permute(op, (1, 0))  # V⊗W ⇒ W⊗V (transpose)
    dflt_in = default_basis(C.oprod(V, W))
    # STD: two-route via machinery-derived operand coords (pin op to (lbV,lbW), feed M in that basis)
    opc = _run(represent(op, basis_env={op.uid: dflt_in}, target=B.basis_prod(lbV, lbW)),
               {**frame, "op_" + str(op.uid): M})
    std = represent(t, basis_env={op.uid: B.basis_prod(lbV, lbW)}, target=B.STD)
    assert not any(n.startswith("cob:target-std") for n in _notes(std))  # pushed down, no post-cob
    assert np.allclose(_run(std, {**frame, "op_" + str(op.uid): opc}), M.T)
    # concrete target (lbW,lbV): == M.T cob'd there, no post-cob
    want = B.basis_prod(lbW, lbV)
    permvar = C.var("pv", C.oprod(W, V))
    expc = _run(represent(permvar, basis_env={permvar.uid: default_basis(C.oprod(W, V))}, target=want),
                {**frame, "pv_" + str(permvar.uid): M.T})
    got = represent(t, basis_env={op.uid: dflt_in}, target=want)
    assert np.allclose(_run(got, {**frame, "op_" + str(op.uid): M}), expc)
    # AUTO byte-identical
    assert _nstmt(represent(t)) == _nstmt(represent(t, target=B.AUTO))


# ---------------------------------------------------------------------------
# Operator arm-C — operand steering: the inverse/adjoint math is reused unchanged; only the
# basis ``f`` is realized in changes, so the natural result lands directly in ``want`` (no cob).
# ---------------------------------------------------------------------------


def _endo_map_target(V):
    """A structure-matching map target ``BasisProd(Dual(lb), lb)`` of the endo space V*⊗V,
    plus its frame-var feeds.  ``_slot_swap_operand_target`` returns a non-None ``f_target``
    for it, so the steered arm fires."""
    a = [C.var(f"e{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    Am = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    feeds = {f"e{i}_{a[i].uid}": Am[:, i] for i in range(3)}
    return B.basis_prod(B.dual_basis(lb), lb), lb, feeds


def test_solve_steers_into_a_slot_aligned_target_no_cob():
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), C.bij)  # UNPINNED
    want, _lb, _fr = _endo_map_target(V)
    prog = represent(C.solve(f), target=want)
    assert _no_target_cob(prog)


def test_pinv_steers_into_a_slot_aligned_target_no_cob():
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V))  # identity metric ⇒ PinvOp branch; UNPINNED
    want, _lb, _fr = _endo_map_target(V)
    prog = represent(C.pinv(f), target=want)
    assert _no_target_cob(prog)


def test_pinv_apply_steers_into_a_vector_target_no_cob():
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V))  # PinvOp branch; both operands UNPINNED
    v = C.var("v", V)
    a = [C.var(f"pe{i}", V) for i in range(3)]
    lb = B.list_basis(*a)  # a concrete vector basis of the result space V
    prog = represent(C.pinv_apply(f, v), target=lb)
    assert _no_target_cob(prog)


def test_arm_c_auto_byte_identical():
    """AUTO (and no-target) byte-identical — arm-C is gated on a concrete/STD target.

    (``Dagger`` is included for completeness — it bailed, but AUTO must still be byte-identical.)
    """
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), C.bij)
    fp = C.var("fp", C.linear_map_space(V, V))
    v = C.var("v", V)
    for t in (C.dagger(f), C.solve(f), C.pinv(fp), C.pinv_apply(fp, v)):
        assert _nstmt(represent(t)) == _nstmt(represent(t, target=B.AUTO))


# ---------------------------------------------------------------------------
# Structured movement: TupleEmbed/TupleProject emit their CONSTANT eye-block
# directly in a structure-matching per-block target — the matrix is byte-identical there
# (verified: an ⊕ basis block-diagonal in [c₀…] with the lone block slot == cᵢ keeps the
# off-block selected columns zero), so it lands in `want` with no conversion cob.
# ---------------------------------------------------------------------------


def _bail_route(node, want, feeds):
    """The always-correct reference: build the natural (default eye-block) Rep, then cob it
    into `want` with the generic per-axis machinery (exactly the epilogue bail) and run it."""
    prog = pa.forward.Program()
    ctx = represent.__globals__["_Ctx"](prog=prog, basis_env={}, inputs={})
    rep = represent.__globals__["_rep"](node, ctx)
    rep2 = represent.__globals__["_coerce_rep_to_target"](ctx, rep, want, note="ref")
    prog.add_output("result", rep2.array)
    return np.asarray(prog.run(feeds)["result"], float)


def _oplus_block_frames():
    """Two per-block list bases (V:3, W:2) of the ⊕ space, plus their frame-var feeds."""
    V, W = C.base("V", 3), C.base("W", 2)
    ev = [C.var(f"ev{i}", V) for i in range(3)]
    ew = [C.var(f"ew{i}", W) for i in range(2)]
    lbV, lbW = B.list_basis(*ev), B.list_basis(*ew)
    Vm = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    Wm = np.array([[1.0, 1], [0, 1.0]])
    feeds = {f"ev{i}_{ev[i].uid}": Vm[:, i] for i in range(3)}
    feeds |= {f"ew{i}_{ew[i].uid}": Wm[:, i] for i in range(2)}
    return C.oplus(V, W), lbV, lbW, feeds


def test_tuple_embed_steers_into_a_structural_target_no_cob():
    sp, lbV, lbW, feeds = _oplus_block_frames()
    cod = B.basis_sum(lbV, lbW)  # the ⊕ codomain in per-block list bases
    for i, block_lb in ((0, lbV), (1, lbW)):
        te = C.tuple_embed(sp, i)
        want = B.basis_prod(B.dual_basis(block_lb), cod)  # block_i ⊸ s: dual dom, primal cod
        prog = represent(te, target=want)
        assert _no_target_cob(prog)  # eye-block built straight in `want`
        # two-route: the steered constant == the always-correct (default eye-block + cob) bail.
        assert np.allclose(_run(prog, feeds), _bail_route(te, want, feeds))


def test_tuple_project_steers_into_a_structural_target_no_cob():
    sp, lbV, lbW, feeds = _oplus_block_frames()
    dom = B.dual_basis(B.basis_sum(lbV, lbW))  # the ⊕ domain (dual) in per-block list bases
    for i, block_lb in ((0, lbV), (1, lbW)):
        tp = C.tuple_project(sp, i)
        want = B.basis_prod(dom, block_lb)  # s ⊸ block_i: dual dom, primal cod
        prog = represent(tp, target=want)
        assert _no_target_cob(prog)
        assert np.allclose(_run(prog, feeds), _bail_route(tp, want, feeds))


def test_tuple_embed_project_bail_on_mismatch():
    """A block slot that disagrees with the ⊕ split (or a non-structural ⊕ slot) BAILS to the
    cob — the eye-block is no longer byte-identical there, so the result is cob'd (still correct)."""
    sp, lbV, lbW, feeds = _oplus_block_frames()
    other = B.list_basis(*[C.var(f"z{i}", C.base("V", 3)) for i in range(3)])
    te = C.tuple_embed(sp, 0)
    # block slot lbV but the ⊕ V-block split is `other` ⇒ mismatch ⇒ bail (cob present).
    mism = B.basis_prod(B.dual_basis(lbV), B.basis_sum(other, lbW))
    assert not _no_target_cob(represent(te, target=mism))
    # a non-structural ⊕ slot (a flat list_basis of the whole ⊕) ⇒ no split ⇒ bail.
    flat = B.list_basis(*[C.var(f"q{i}", sp) for i in range(5)])
    nonstruct = B.basis_prod(B.dual_basis(lbV), flat)
    assert not _no_target_cob(represent(te, target=nonstruct))


def test_tuple_embed_project_auto_byte_identical():
    """AUTO/STD unchanged — the #3 steer is gated on a CONCRETE structural Basis target.

    (STD is byte-identical because the default eye-block IS already in the standard basis ⇒
    the STD epilogue is a no-op, never a cob — same statement count as AUTO.)"""
    sp, _lbV, _lbW, _feeds = _oplus_block_frames()
    for i in range(2):
        for node in (C.tuple_embed(sp, i), C.tuple_project(sp, i)):
            assert _nstmt(represent(node)) == _nstmt(represent(node, target=B.AUTO))
            assert _nstmt(represent(node, target=B.STD)) == _nstmt(represent(node, target=B.AUTO))


def test_external_equivariant_pushes_target_into_operands():
    """An EQUIVARIANT external (commutes with cob) pushes a concrete/STD target into its operands so
    the elementwise result is born in `want` — no post-cob.  STANDARD kernels are unaffected."""
    from grassmann.lower.represent import BasisReq, ExternalOp

    V = C.base("V", 3)
    neg = ExternalOp(name="neg", fn=np.negative, basis_req=BasisReq.EQUIVARIANT)
    v = C.var("v", V)
    t = C.external(V, neg, v)
    vc = np.array([1.0, 2, 3])
    a = [C.var(f"e{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    Am = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    frame = {f"e{i}_{a[i].uid}": Am[:, i] for i in range(3)}
    # two-route: operand pinned to lb (fed lb-coords) + STD == operand std (fed vc) + STD == -vc
    ref = _run(represent(t, basis_env={v.uid: default_basis(V)}, target=B.STD), {_vk(v): vc})
    got = _run(represent(t, basis_env={v.uid: lb}, target=B.STD),
               {**frame, _vk(v): np.linalg.solve(Am, vc)})
    assert np.allclose(ref, -vc) and np.allclose(got, -vc)
    # steered into a concrete target with an UNPINNED (leaf) operand ⇒ declared in `want`, no cob
    assert _no_target_cob(represent(t, target=lb))
    # AUTO byte-identical; a STANDARD kernel is unchanged
    assert _nstmt(represent(t)) == _nstmt(represent(t, target=B.AUTO))
    sq = ExternalOp(name="sqrt", fn=np.sqrt, basis_req=BasisReq.STANDARD)
    assert np.allclose(_run(represent(C.external(V, sq, v), target=B.STD), {_vk(v): vc}), np.sqrt(vc))
