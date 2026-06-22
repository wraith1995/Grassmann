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

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent
from grassmann.lower.represent import default_basis


def _vk(v):
    return f"{v.name}_{v.uid}"


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
