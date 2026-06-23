"""Basis-tag honesty: the ``target`` feature must give CORRECT results even when an
operand lives in a non-default basis.

The bug being guarded: ~11 ``_rep_X`` helpers tagged their result ``default_basis(space)``
while the coordinates actually lived in the OPERAND's basis.  Inert under ``AUTO`` (the
lowering never cob'd), but ``target=STD``/concrete activates the (now-emitted) cob FROM the
(dishonest) tag → wrong numbers whenever an operand is in a non-default basis.

The check is **independent of the implementation**: for each affected helper we build a term
whose operand(s) are ``Var``s, pick a std operand value ``V`` (numpy), and run two routes that
must agree by construction:

  route_STD : represent(t, basis_env={op: lb}, target=B.STD)  fed  lb-COORDS of ``V``
  route_REF : represent(t, basis_env={op: std}, target=B.AUTO)  fed  ``V``  (all std)

``route_REF`` is the ground-truth std output (everything std, no cob); ``route_STD`` feeds the
same underlying value expressed in a non-default basis ``lb`` and asks for the std result —
which must match.  We also round-trip ``target=lb`` then cob back == AUTO.

``lb`` is a non-default ``B.list_basis(...)`` of the operand's space realized by a known
invertible matrix ``Lb`` (its own frame Vars).  Two operand shapes appear:

* a **vector** of ``V`` (dim n): ``lb`` realized by ``Lb``; lb-coords = ``solve(Lb, value)``.
* a **map** of ``V*⊗V`` stored ``(dom, cod)``: basis ``BasisProd(dual(lb), lb)``;
  lb-coords = ``Lb.T @ value @ inv(Lb).T`` (verified against the cob the lowering emits).
"""

import numpy as np
import polyarray as pa

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent
from grassmann.lower.represent import default_basis


def _vk(v):
    return f"{v.name}_{v.uid}"


def _nstmts(prog):
    """Total statement count across the program and all its sub-programs."""
    return sum(len(s.statements) for s in pa.forward.iter_programs(prog))


def _notes(prog):
    """All statement notes across the program and its sub-programs."""
    return [st.note for s in pa.forward.iter_programs(prog) for st in s.statements]


def _run(prog, feeds):
    return np.asarray(prog.run(feeds)["result"], float)


# A fixed non-default, invertible 3×3 basis matrix for V=ℝ³ (columns = basis vectors).
_LB = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
_LBI = np.linalg.inv(_LB)


def _list_basis(V, tag, Lb=_LB):
    """A ``list_basis`` of ``V`` realized by ``Lb``'s columns, + its frame-var feeds."""
    n = Lb.shape[1]
    a = [C.var(f"{tag}{i}", V) for i in range(n)]
    return B.list_basis(*a), {_vk(a[i]): Lb[:, i] for i in range(n)}


def _vec_coords(value, Lb=_LB):
    """lb-coords of a vector ``value`` (std) in the basis realized by ``Lb``."""
    return np.linalg.solve(Lb, value)


def _map_coords(value, Lb=_LB):
    """lb-coords of a map ``value`` (std, stored dom,cod) in ``BasisProd(dual(lb), lb)``."""
    return Lb.T @ value @ np.linalg.inv(Lb).T


# ---------------------------------------------------------------------------
# The two-route checker.
# ---------------------------------------------------------------------------


def _two_route(term, op_specs):
    """Assert route_STD == route_REF for ``term``.

    ``op_specs`` maps each operand ``Var`` -> ``(lb, lb_feeds, std_value, coords_fn)``:
      lb         the non-default basis to pin the operand to (route_STD)
      lb_feeds   that basis's frame-var feeds (needed in BOTH routes)
      std_value  the operand's std (numpy) value (route_REF feed)
      coords_fn  std_value -> lb-coords (route_STD feed)
    Returns ``(out_std, out_ref)`` for any extra assertions.
    """
    env_lb, env_std = {}, {}
    feeds_std, feeds_ref = {}, {}
    for op, (lb, lb_feeds, value, coords_fn) in op_specs.items():
        env_lb[op.uid] = lb
        env_std[op.uid] = default_basis(op.space)
        feeds_std[_vk(op)] = coords_fn(value)
        feeds_ref[_vk(op)] = value
        feeds_std.update(lb_feeds)  # frame vars must be fed in route_STD (lb is realized)
        # route_REF pins to std, so the frame vars are unused there; harmless to omit.
    out_std = _run(represent(term, basis_env=env_lb, target=B.STD), feeds_std)
    out_ref = _run(represent(term, basis_env=env_std, target=B.AUTO), feeds_ref)
    assert np.allclose(out_std, out_ref), (
        f"route mismatch:\nSTD={out_std}\nREF={out_ref}"
    )
    return out_std, out_ref


# ---------------------------------------------------------------------------
# Per-helper terms.
# ---------------------------------------------------------------------------


def test_permute():
    """_rep_permute: result tag must be the operand's BasisProd reordered by the perm."""
    V = C.base("V", 3)
    VV = C.oprod(V, V)
    # operand is a value of V⊗V (a 3×3 array); use a list-basis on each factor via a prod.
    op = C.var("m", VV)
    lb1, f1 = _list_basis(V, "p")
    lbV = B.basis_prod(lb1, lb1)
    value = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    # coords of a V⊗V tensor in basis_prod(lb,lb): cob each axis lb->std applies Lb;
    # so std = Lb @ C @ Lb.T  ⇒  C = inv(Lb) @ std @ inv(Lb).T.
    def coords(v):
        return _LBI @ v @ _LBI.T
    t = C.permute(op, (1, 0))
    _two_route(t, {op: (lbV, f1, value, coords)})


def test_lift_and_invlift():
    """_rep_lift / InvLift: reshape preserves coords; tag = fused/unfused operand basis."""
    V = C.base("V", 3)
    VV = C.oprod(V, V)
    op = C.var("m", VV)
    lb1, f1 = _list_basis(V, "l")
    lbV = B.basis_prod(lb1, lb1)
    value = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
    def coords(v):
        return _LBI @ v @ _LBI.T
    # Lift fuses the two V axes into one V⊗V axis (dim 9).
    t = C.lift(op, (0, 1))
    _two_route(t, {op: (lbV, f1, value, coords)})


def test_invlift():
    """_rep_lift InvLift branch: unfuse a fused operand (reshape preserves coords)."""
    V = C.base("V", 3)
    VV = C.oprod(V, V)
    op = C.var("mf", VV)
    lb1, f1 = _list_basis(V, "u")
    lbV = B.basis_prod(lb1, lb1)
    value = np.array([[1.0, 0, 2], [3, 1, 0], [0, 4, 1.0]])
    def coords(v):
        return _LBI @ v @ _LBI.T
    # Lift fuses V⊗V into one Dual(V⊗V) axis; InvLift then reassociates it — exercises the
    # InvLift branch of ``_rep_lift`` over an operand carried in ``lb``.
    t = C.inv_lift(C.lift(op, (0, 1)), (0,))
    _two_route(t, {op: (lbV, f1, value, coords)})


def test_kron():
    """_rep_kron: tag = basis_prod(*f-slot bases, *g-slot bases)."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)  # V*⊗V, stored (dom,cod)
    f = C.var("f", MS)
    g = C.var("g", MS)
    lb, fr = _list_basis(V, "k")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 1, 0], [0, 3, 1]])
    Mg = np.array([[1.0, 2, 0], [0, 1, 1], [1, 0, 2]])
    t = C.kron(f, g)
    _two_route(t, {
        f: (opb, fr, Mf, _map_coords),
        g: (opb, fr, Mg, _map_coords),
    })


def test_compose():
    """_rep_compose: tag = surviving f-slot bases + each g's domain-slot basis."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    g = C.var("g", MS)
    lb, fr = _list_basis(V, "c")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 1, 0], [0, 3, 1]])
    Mg = np.array([[1.0, 2, 0], [0, 1, 1], [1, 0, 2]])
    t = C.compose(f, (0,), g)  # chain g's codomain into f's slot 0
    _two_route(t, {
        f: (opb, fr, Mf, _map_coords),
        g: (opb, fr, Mg, _map_coords),
    })


def test_trace():
    """_rep_trace: tag = the non-traced f-slot bases.

    A rank-3 tensor ``V ⊗ V* ⊗ V``: ``trace(0,1)`` contracts the primal slot 0 with the dual
    slot 1, leaving the surviving slot 2 (a ``V``).
    """
    V = C.base("V", 3)
    T3 = C.oprod(V, C.dual(V), V)
    op = C.var("t3", T3)
    lb, f1 = _list_basis(V, "tr")
    dlb = B.dual_basis(lb)
    lbV = B.basis_prod(lb, dlb, lb)
    value = np.arange(27.0).reshape(3, 3, 3)
    def coords(v):
        # realize lb->std: primal axis applies Lb, dual axis applies inv(Lb).T.
        # coords = inverse: primal axis inv(Lb), dual axis Lb.T.
        return np.einsum("ia,jb,kc,abc->ijk", _LBI, _LB.T, _LBI, v)
    t = C.trace(op, 0, 1)
    _two_route(t, {op: (lbV, f1, value, coords)})


def test_remetric():
    """_rep_remetric: tag = operand basis re-homed onto the WithInner space (coords unchanged)."""
    V = C.base("V", 3)
    op = C.var("v", V)
    lb, fr = _list_basis(V, "rm")
    # a metric term: a Var of V*⊗V used as the metric (SPD), fed an SPD matrix.
    g_metric = np.array([[2.0, 0, 0], [0, 3, 0], [0, 0, 1]])
    gvar = C.var("g", C.linear_map_space(V, V))
    value = np.array([1.0, 2, 3])
    # remetric only swaps metric metadata; the metric var must be fed in BOTH routes.
    Wt = C.remetric(op, gvar)

    env_lb = {op.uid: lb, gvar.uid: default_basis(gvar.space)}
    env_std = {op.uid: default_basis(V), gvar.uid: default_basis(gvar.space)}
    feeds_std = {_vk(op): _vec_coords(value), _vk(gvar): g_metric, **fr}
    feeds_ref = {_vk(op): value, _vk(gvar): g_metric}
    out_std = _run(represent(Wt, basis_env=env_lb, target=B.STD), feeds_std)
    out_ref = _run(represent(Wt, basis_env=env_std, target=B.AUTO), feeds_ref)
    assert np.allclose(out_std, out_ref), f"remetric STD={out_std} REF={out_ref}"


def test_solve():
    """_rep_solve: tag = the dual-swapped operand slot bases (inverse maps cod<->dom)."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS, C.bij)
    lb, fr = _list_basis(V, "s")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2]])  # invertible
    t = C.solve(f)
    _two_route(t, {f: (opb, fr, Mf, _map_coords)})


def test_pinv():
    """_rep_pinv (PinvOp branch): tag = dual-swapped operand slot bases."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)  # identity metric ⇒ plain PinvOp branch
    f = C.var("f", MS)
    lb, fr = _list_basis(V, "pv")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2]])
    t = C.pinv(f)
    _two_route(t, {f: (opb, fr, Mf, _map_coords)})


def test_solve_apply():
    """_rep_solve_apply: tag = dual-swapped operand slot basis applied to v."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS, C.bij)
    v = C.var("v", V)
    lb, fr = _list_basis(V, "sa")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2]])
    vv = np.array([1.0, 0, 2])
    t = C.solve_apply(f, v)
    _two_route(t, {
        f: (opb, fr, Mf, _map_coords),
        v: (lb, fr, vv, _vec_coords),
    })


def test_pinv_apply():
    """_rep_pinv_apply: tag = dual-swapped operand slot basis applied to v (PinvOp branch)."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    v = C.var("v", V)
    lb, fr = _list_basis(V, "pa")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2]])
    vv = np.array([1.0, 0, 2])
    t = C.pinv_apply(f, v)
    _two_route(t, {
        f: (opb, fr, Mf, _map_coords),
        v: (lb, fr, vv, _vec_coords),
    })


def test_neval_lam():
    """_rep_neval_lam: tag = the call output's operand-derived basis.

    Build a Lam ``λx. f·x`` (``f`` a std map W←V) applied (NEval) to a term-arg operand ``a``
    pinned to ``lb``.  The output (in W) is the operand-derived value ``f @ a``; the helper
    must tag it honestly so ``target=STD`` re-expresses correctly.
    """
    from grassmann.core.kinds import SpaceSort

    V = C.base("V", 3)
    W = C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))  # stored (dom,cod) = (V,W)
    x = C.var("x", V)  # the Lam's bound var
    lam = C.lam(((x.name, SpaceSort(V)),), C.eval_(f, (0,), x), (x,))
    op = C.var("a", V)
    lb, fr = _list_basis(V, "nl")
    value = np.array([2.0, 3, 5])
    Fm = np.array([[1.0, 0, 2], [0, 1, 1], [3, 0, 1]])
    t = C.neval(lam, (0,), op)

    # ``f`` is std-pinned in both routes; only ``op`` switches basis.
    env_lb = {op.uid: lb, f.uid: default_basis(f.space)}
    env_std = {op.uid: default_basis(V), f.uid: default_basis(f.space)}
    feeds_std = {_vk(op): _vec_coords(value), _vk(f): Fm, **fr}
    feeds_ref = {_vk(op): value, _vk(f): Fm}
    out_std = _run(represent(t, basis_env=env_lb, target=B.STD), feeds_std)
    out_ref = _run(represent(t, basis_env=env_std, target=B.AUTO), feeds_ref)
    assert np.allclose(out_std, out_ref), f"neval_lam STD={out_std} REF={out_ref}"


# ---------------------------------------------------------------------------
# Concrete-basis round-trip: represent(t, target=concrete lb_out) then cob back == AUTO.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Operator arm-C correctness: a STRUCTURE-MATCHING concrete map ``want`` (so steering fires)
# must still give the correct numbers — route to ``want`` (operand pinned), realize back, == REF.
# ---------------------------------------------------------------------------


def _realize_map_via_lowering(map_space, want, coords, want_feeds):
    """Realize a map ``want``-coords back to std using the LOWERING's own cob (convention-proof).

    Feeds ``coords`` into a fresh ``Var`` pinned to ``want`` and lowers it with ``target=STD`` —
    so the realize matches exactly what the lowering's ``cob:target`` would do, independent of any
    hand-derived realize formula.
    """
    m = C.var("m_realize", map_space)
    prog = represent(m, basis_env={m.uid: want}, target=B.STD)
    return _run(prog, {_vk(m): coords, **want_feeds})


def _two_route_map_target(term, op_specs, want, want_feeds, map_space):
    """route_TGT (concrete map ``want``, operand pinned) realized back == route_REF (std AUTO)."""
    env_lb, env_std = {}, {}
    feeds_tgt, feeds_ref = {}, {}
    for op, (lb, lb_feeds, value, coords_fn) in op_specs.items():
        env_lb[op.uid] = lb
        env_std[op.uid] = default_basis(op.space)
        feeds_tgt[_vk(op)] = coords_fn(value)
        feeds_ref[_vk(op)] = value
        feeds_tgt.update(lb_feeds)
    feeds_tgt.update(want_feeds)
    out_tgt = _run(represent(term, basis_env=env_lb, target=want), feeds_tgt)
    out_ref = _run(represent(term, basis_env=env_std, target=B.AUTO), feeds_ref)
    realized = _realize_map_via_lowering(map_space, want, out_tgt, want_feeds)
    assert np.allclose(realized, out_ref), (
        f"map-target route mismatch:\nTGT(realized)={realized}\nREF={out_ref}"
    )


def _endo_map_want():
    """A structure-matching map target ``BasisProd(dual(lb_out), lb_out)`` of an endo V*⊗V,
    realized by ``_LB``, + its frame-var feeds (so ``_slot_swap_operand_target`` fires)."""
    V = C.base("V", 3)
    lb_out, fr = _list_basis(V, "wt")
    return B.basis_prod(B.dual_basis(lb_out), lb_out), fr


# NOTE: ``_rep_dagger`` is NOT arm-C steered.  Operand-steering is unsound for its transpose
# fast path: ``_is_immersed_metric_basis`` treats a ``list_basis`` as orthonormal (Gram skipped),
# but a non-orthonormal list basis has Gram ≠ I, where the adjoint is ``G·fᵀ·G⁻¹`` ≠ ``fᵀ``
# (``transpose(map_coords(M)) ≠ map_coords(Mᵀ)``).  Unlike Solve/Pinv (inversion IS
# basis-covariant), transpose is not.  (A non-orthonormal ``list_basis`` is immersed-metric, so it
# routes through the metric formula — honest tag.  The fast path now ALSO tags the honest operand
# basis for an orthonormal-but-NON-STANDARD operand, fixing a former silent-wrong STD cob; see
# ``test_dagger_orthonormal_nonstandard_operand_std``.)


def test_solve_concrete_target():
    """_rep_solve arm-C: a slot-aligned concrete map ``want`` is correct (STD covered above)."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS, C.bij)
    lb, fr = _list_basis(V, "sc")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2]])
    want, wfr = _endo_map_want()
    _two_route_map_target(C.solve(f), {f: (opb, fr, Mf, _map_coords)}, want, wfr, MS)


def test_pinv_concrete_target():
    """_rep_pinv arm-C (PinvOp branch): a slot-aligned concrete map ``want`` is correct."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    lb, fr = _list_basis(V, "pc")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2]])
    want, wfr = _endo_map_want()
    _two_route_map_target(C.pinv(f), {f: (opb, fr, Mf, _map_coords)}, want, wfr, MS)


def test_pinv_apply_concrete_target():
    """_rep_pinv_apply arm-C (PinvOp branch): a concrete VECTOR ``want`` is correct.

    The result vector is f's domain primal; steering f's dom slot to ``Dual(want)`` lands it
    natively in ``want``.  Realize the want-coords back (``Lb_out @ coords``) == std AUTO.
    """
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    v = C.var("v", V)
    lb, fr = _list_basis(V, "pac")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2]])
    vv = np.array([1.0, 0, 2])
    want, wfr = _list_basis(V, "pawt")
    env_lb = {f.uid: opb, v.uid: lb}
    env_std = {f.uid: default_basis(MS), v.uid: default_basis(V)}
    feeds_tgt = {_vk(f): _map_coords(Mf), _vk(v): _vec_coords(vv), **fr, **wfr}
    feeds_ref = {_vk(f): Mf, _vk(v): vv}
    out_tgt = _run(represent(C.pinv_apply(f, v), basis_env=env_lb, target=want), feeds_tgt)
    out_ref = _run(represent(C.pinv_apply(f, v), basis_env=env_std, target=B.AUTO), feeds_ref)
    assert np.allclose(_LB @ out_tgt, out_ref), f"{_LB @ out_tgt} vs {out_ref}"


def test_concrete_basis_round_trip():
    """``represent(t, target=lb_out)`` then realize back to std == ``AUTO`` (a vector result).

    Exercises the concrete-``Basis`` epilogue (``_match_result``) end-to-end: with std operands
    the AUTO result is std; targeting a concrete ``lb_out`` re-expresses it into ``lb_out``-coords,
    and realizing those columns back (``Lb_out @ coords``) recovers the std AUTO result.  Uses a
    vector-result term whose helper is one of the fixed ones (``pinv_apply``).
    """
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    v = C.var("v", V)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2]])
    vv = np.array([1.0, 0, 2])
    t = C.pinv_apply(f, v)

    # operands std-pinned in BOTH runs (so AUTO is the ground-truth std result).
    env = {f.uid: default_basis(MS), v.uid: default_basis(V)}
    lb_out, fr = _list_basis(V, "rt")  # a concrete basis of the result space V
    feeds = {_vk(f): Mf, _vk(v): vv, **fr}

    out_auto = _run(represent(t, basis_env=env, target=B.AUTO), feeds)
    out_lb = _run(represent(t, basis_env=env, target=lb_out), feeds)
    # ``out_lb`` are lb_out-coords; realizing the columns (``Lb @ coords``) recovers std == AUTO.
    assert np.allclose(_LB @ out_lb, out_auto), f"round-trip: {_LB @ out_lb} vs {out_auto}"


# ---------------------------------------------------------------------------
# Operator concrete-target PUSH-DOWN (roadmap-#1): _rep_eval / _rep_compose steer their
# operand bases so the result is born directly in ``want`` (no result cob).  The two-route check
# proves correctness; AUTO must stay byte-identical (stmt count); a steered case with default/leaf
# operands must emit NO ``cob:target`` note (the steer realized natively).
#
# NOTE: ``_rep_neval_lam`` is NOT steered — backed off.  Threading ``target`` into the body
# lowering requires it to enter the sub-Program memo key (``_build_lam_subprogram``, keyed on
# ``(id(Lam), input uids)`` only), else two targets of the same Lam collide.  A ``Basis`` is not
# guaranteed hashable in the seam (an embedded-array term basis is not — see
# ``_operand_metric_basis``), so it cannot safely key the memo.  neval_lam keeps its bail.
# ---------------------------------------------------------------------------


def test_eval_pushdown():
    """_rep_eval push-down: f·v with a SURVIVING slot — operands pinned, result steered to want."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)  # V*⊗V, stored (dom,cod); slot 1 (cod) survives
    f = C.var("f", MS)
    v = C.var("v", V)
    lb, fr = _list_basis(V, "ep")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 1, 0], [0, 3, 1]])
    vv = np.array([1.0, 0, 2])
    t = C.eval_(f, (0,), v)  # contract slot 0 (dom) with v; slot 1 (cod, a V) survives
    # target=STD: two-route correctness.
    _two_route(t, {f: (opb, fr, Mf, _map_coords), v: (lb, fr, vv, _vec_coords)})
    # target=concrete vector want: realize back to std == REF.
    want, wfr = _list_basis(V, "ewt")
    env_lb = {f.uid: opb, v.uid: lb}
    env_std = {f.uid: default_basis(MS), v.uid: default_basis(V)}
    feeds_tgt = {_vk(f): _map_coords(Mf), _vk(v): _vec_coords(vv), **fr, **wfr}
    feeds_ref = {_vk(f): Mf, _vk(v): vv}
    out_tgt = _run(represent(t, basis_env=env_lb, target=want), feeds_tgt)
    out_ref = _run(represent(t, basis_env=env_std, target=B.AUTO), feeds_ref)
    assert np.allclose(_LB @ out_tgt, out_ref), f"want route: {_LB @ out_tgt} vs {out_ref}"
    # AUTO byte-identical (stmt count target=AUTO == no-target).
    assert _nstmts(represent(t, basis_env=env_std, target=B.AUTO)) == \
        _nstmts(represent(t, basis_env=env_std))
    # A steered case with DEFAULT operands emits NO cob:target note (native realize).
    notes = _notes(represent(t, basis_env=env_std, target=B.STD))
    assert not any((n or "").startswith("cob:target") for n in notes), notes


def test_compose_pushdown():
    """_rep_compose push-down: g into f's slot — operands pinned, result steered to want."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    g = C.var("g", MS)
    lb, fr = _list_basis(V, "cp")
    opb = B.basis_prod(B.dual_basis(lb), lb)
    Mf = np.array([[2.0, 0, 1], [1, 1, 0], [0, 3, 1]])
    Mg = np.array([[1.0, 2, 0], [0, 1, 1], [1, 0, 2]])
    t = C.compose(f, (0,), g)
    # target=STD: two-route correctness.
    _two_route(t, {f: (opb, fr, Mf, _map_coords), g: (opb, fr, Mg, _map_coords)})
    # target=concrete map want: realize back via the lowering's own cob == REF.
    want, wfr = _endo_map_want()
    _two_route_map_target(
        t, {f: (opb, fr, Mf, _map_coords), g: (opb, fr, Mg, _map_coords)}, want, wfr, MS,
    )
    env_std = {f.uid: default_basis(MS), g.uid: default_basis(MS)}
    # AUTO byte-identical (stmt count target=AUTO == no-target).
    assert _nstmts(represent(t, basis_env=env_std, target=B.AUTO)) == \
        _nstmts(represent(t, basis_env=env_std))
    # A steered case with DEFAULT operands emits NO cob:target note (native realize).
    notes = _notes(represent(t, basis_env=env_std, target=B.STD))
    assert not any((n or "").startswith("cob:target") for n in notes), notes


def test_dagger_orthonormal_nonstandard_operand_std():
    """_rep_dagger fast-path tag honesty: an ORTHONORMALIZE (Gram=I) but non-standard operand fires
    the transpose fast path; its result must be tagged in the operand basis (not default) so the STD
    cob fires.  Two-route via machinery-derived operand coords (==-by-construction)."""
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])  # std coords
    qsrc = np.array([[1.0, 2, 0], [0, 1, 1], [1, 0, 2.0]])  # non-standard frame (ORTHONORMALIZE'd → Q)
    qv = [C.var(f"q{i}", V) for i in range(3)]
    qb = B.list_basis(*qv, reduce=C.Reduce.ORTHONORMALIZE)
    frame = {f"q{i}_{qv[i].uid}": qsrc[:, i] for i in range(3)}
    opb = B.basis_prod(B.dual_basis(qb), qb)
    std_env = {f.uid: default_basis(MS)}
    ref = _run(represent(C.dagger(f), basis_env=std_env, target=B.STD), {_vk(f): Mf})
    assert np.allclose(ref, Mf.T)  # std adjoint is the transpose
    # Mf expressed in the orthonormal-non-standard map basis, via the lowering's own cob
    mf_q = _run(represent(f, basis_env=std_env, target=opb), {**frame, _vk(f): Mf})
    got = _run(represent(C.dagger(f), basis_env={f.uid: opb}, target=B.STD), {**frame, _vk(f): mf_q})
    assert np.allclose(got, ref)  # was WRONG before the fast-path tag fix (returned the Q-basis adjoint)


def test_llam_nondefault_body_basis_fails_loudly():
    """_rep_llam tag honesty: a body landing in a non-default basis (e.g. a pinned operand) can't be
    honestly tagged yet (the map tensor's domain free-label axes aren't basis-tracked), so it must
    RAISE rather than silently mislabel.  A default-operand body is correct (AUTO == STD == Mfᵀ)."""
    from grassmann.lower.represent import RepresentError

    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])
    x = C.var("x", V)
    llam = C.llam(x, body=C.eval_(f, (0,), x))  # LLam([x]. f·x) — the matrix of f
    std_env = {f.uid: default_basis(MS)}
    assert np.allclose(_run(represent(llam, basis_env=std_env), {_vk(f): Mf}), Mf.T)
    assert np.allclose(_run(represent(llam, basis_env=std_env, target=B.STD), {_vk(f): Mf}), Mf.T)
    # f pinned to a non-default map basis ⇒ body lands in `lb` ⇒ must raise (was silent-wrong)
    a = [C.var(f"e{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    Am = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    frame = {f"e{i}_{a[i].uid}": Am[:, i] for i in range(3)}
    opb = B.basis_prod(B.dual_basis(lb), lb)
    mf_lb = _run(represent(f, basis_env=std_env, target=opb), {**frame, _vk(f): Mf})
    try:
        _run(represent(llam, basis_env={f.uid: opb}, target=B.STD), {**frame, _vk(f): mf_lb})
        raise AssertionError("expected RepresentError for a non-default llam body basis")
    except RepresentError as ex:
        assert "non-default basis" in str(ex)
    # domain-only case (cod stays default, a body eval cob's the var axis to a pinned covector):
    # LLam([x]. ⟨a, x⟩) with `a` pinned non-default must also raise (caught by the pin pre-check).
    a = C.var("a", C.dual(V))
    d = [C.var(f"d{i}", C.dual(V)) for i in range(3)]
    lbd = B.list_basis(*d)
    dfr = {f"d{i}_{d[i].uid}": Am[:, i] for i in range(3)}
    lin = C.llam(x, body=C.eval_(a, (0,), x))
    try:
        _run(represent(lin, basis_env={a.uid: lbd}, target=B.STD), {**dfr, _vk(a): np.array([1.0, 2, 3])})
        raise AssertionError("expected RepresentError for a domain-only non-default llam pin")
    except RepresentError as ex:
        assert "non-default basis" in str(ex)


def test_pinv_gsvd_default_operand_correct():
    """_rep_pinv / _rep_pinv_apply GSVD (metric) branch, DEFAULT operand: square invertible ``f`` ⇒
    the metric-pinv collapses to the inverse, so the result == ``inv(Mf)`` — an independent ground
    truth.  Also guards byte-identity (STD == AUTO stmt count: a default-operand result is already in
    default, so the STD epilogue cob is a no-op)."""
    V0 = C.base("V", 3)
    g = C.var("gm", C.linear_map_space(V0, V0))
    Vg = C.with_inner(V0, g)
    MS = C.linear_map_space(Vg, Vg)
    f = C.var("f", MS)
    Gm = np.diag([2.0, 3.0, 1.0])
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])  # square invertible
    out = _run(represent(C.pinv(f), basis_env={f.uid: default_basis(MS)}, target=B.STD),
               {_vk(f): Mf, _vk(g): Gm})
    assert np.allclose(out, np.linalg.inv(Mf))
    # default operand: STD epilogue is a no-op ⇒ byte-identical stmt count to AUTO.
    p_std = represent(C.pinv(f), basis_env={f.uid: default_basis(MS)}, target=B.STD)
    p_auto = represent(C.pinv(f), basis_env={f.uid: default_basis(MS)}, target=B.AUTO)
    assert _nstmts(p_std) == _nstmts(p_auto)


# The GSVD metric-pinv TWO-ROUTE correctness check (non-default operand): the metric is now built in
# the OPERAND's per-slot bases, so a non-default pin is correct.  An in-process two-route check
# accumulates free einsum labels across the many ``represent()`` calls and trips a SPURIOUS numpy
# "does not contain the correct number of indices" ValueError (a probe artifact, NOT a real bug), so
# each route is built+run in a fresh subprocess sharing data via a temp ``.npy`` (`see _gsvd_two_route`).


def _gsvd_two_route(kind):
    """Run the two GSVD-pinv routes (``std`` vs ``ref``) for ``kind`` in SEPARATE subprocesses and
    return ``(got, ref)``.  Each subprocess builds exactly ONE ``represent()`` so the in-process
    einsum-symbol accumulation artifact never bites.  Operand ``f`` (square invertible) is pinned to a
    non-default ``BasisProd(dual(lb), lb)`` over a ``WithInner`` space (and ``v`` to ``lb`` for apply);
    ``target=STD`` (operand pinned) must equal ``target=AUTO`` (default operand) — same value, two
    expressions."""
    import os
    import subprocess
    import sys
    import tempfile

    src = os.path.join(os.path.dirname(__file__), "_gsvd_two_route_helper.py")
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for route in ("std", "ref"):
            p = os.path.join(d, f"{route}.npy")
            subprocess.run(
                [sys.executable, src, route, kind, p], check=True,
                capture_output=True, text=True,
            )
            out[route] = np.load(p)
    return out["std"], out["ref"]


def test_pinv_gsvd_nondefault_operand_two_route():
    """_rep_pinv GSVD branch, NON-DEFAULT operand: the metric built in the operand basis makes the pin
    correct — ``target=STD`` (operand pinned to ``lb``) == ``target=AUTO`` (default operand)."""
    got, ref = _gsvd_two_route("pinv")
    assert np.allclose(got, ref), f"pinv GSVD two-route:\nGOT={got}\nREF={ref}"


def test_pinv_apply_gsvd_nondefault_operand_two_route():
    """_rep_pinv_apply GSVD branch, NON-DEFAULT operand: the metric built in f's per-slot bases plus a
    cob of ``v`` into f's cod slot makes the pin correct — STD (pinned) == AUTO (default)."""
    got, ref = _gsvd_two_route("pinvapply")
    assert np.allclose(got, ref), f"pinv_apply GSVD two-route:\nGOT={got}\nREF={ref}"
