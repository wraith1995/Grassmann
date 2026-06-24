"""Systematic basis-honesty sweep — the permanent guard against the SILENT-WRONG class.

This session fixed five bugs of one shape: an arm tagged its result in basis A while the coordinates
actually lived in basis B, so a ``target=STD`` cob was a no-op (or wrong) and the program returned
WRONG NUMBERS with no error (dagger fast path, llam non-default body, pinv-GSVD, eval rank-1 covector,
embed subspace-list).  Each was found by a hand-written two-route probe; this module turns that probe
into one parametrized property test over EVERY arm.

The invariant (per arm × non-default operand basis):

    represent(t, operand pinned to a NON-default basis B, fed B-coords, target=STD)
        either  ==  represent(t, operand default, fed std value, target=AUTO)      [honest]
        or      raises RepresentError                                              [loud bail — OK]
    but NEVER  succeeds with different numbers                                     [SILENT-WRONG — fail]

Two robustness rules learned the hard way:
  * coords in basis B are derived VIA THE MACHINERY (``represent(op, target=B)``), never by a
    hand-written transform — hand-transforms were the source of every false probe result this session.
  * a loud ``RepresentError`` is an ACCEPTABLE outcome (a deliberate bail, e.g. eye-mode llam on a
    cod-non-default body); only a *silent* mismatch is a bug.

To add an arm: append a ``Case`` to ``CASES``.  A ``Case`` is ``(name, make_term, var_specs)`` where
``var_specs`` maps each pinned operand ``Var`` to ``(space, std_value)``; the sweep auto-applies each
non-default basis flavor valid for that space.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent
from grassmann.lower.represent import RepresentError, default_basis


def _vk(v):
    return f"{v.name}_{v.uid}"


def _run(prog, feeds):
    return np.asarray(prog.run(feeds)["result"], float)


# A fixed non-orthonormal invertible 3×3 / 2×2 frame (columns = basis vectors).
_LB3 = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
_LB2 = np.array([[1.0, 1], [0, 1.0]])


def _frame_feeds(frame_vars, M):
    return {_vk(v): M[:, i] for i, v in enumerate(frame_vars)}


# ---------------------------------------------------------------------------
# Basis flavors per space — each returns (basis, frame_feeds) for a NON-default basis of `space`.
# ---------------------------------------------------------------------------


def _flavors(space, tag):
    """Non-default basis flavors valid for ``space``: a non-orthonormal list, an ORTHONORMALIZE list
    (vector/covector spaces), and a BasisProd(dual(lb), lb) for a rank-2 map space."""
    out = []
    factors = represent.__globals__["_factors"](space)
    if len(factors) == 1:  # rank-1: a vector or covector space
        n = represent.__globals__["_static_dim"](space)
        M = _LB3 if n == 3 else _LB2
        fv = [C.var(f"{tag}l{i}", space) for i in range(n)]
        out.append((B.list_basis(*fv), _frame_feeds(fv, M)))
        ov = [C.var(f"{tag}o{i}", space) for i in range(n)]
        out.append((B.list_basis(*ov, reduce=C.Reduce.ORTHONORMALIZE), _frame_feeds(ov, M)))
    elif len(factors) == 2:  # rank-2 map: dom ⊗ cod
        dom, cod = factors
        nd = represent.__globals__["_static_dim"](dom)
        nc = represent.__globals__["_static_dim"](cod)
        # non-orthonormal map basis BasisProd(dual(lb), lb)
        dv = [C.var(f"{tag}d{i}", _primal_space(dom)) for i in range(nd)]
        cv = [C.var(f"{tag}c{i}", cod) for i in range(nc)]
        feeds = {**_frame_feeds(dv, _LB3 if nd == 3 else _LB2),
                 **_frame_feeds(cv, _LB3 if nc == 3 else _LB2)}
        out.append((B.basis_prod(B.dual_basis(B.list_basis(*dv)), B.list_basis(*cv)), feeds))
        # ORTHONORMALIZE (non-standard orthonormal) map basis — Gram=I but ≠ default
        odv = [C.var(f"{tag}od{i}", _primal_space(dom)) for i in range(nd)]
        ocv = [C.var(f"{tag}oc{i}", cod) for i in range(nc)]
        ofeeds = {**_frame_feeds(odv, _LB3 if nd == 3 else _LB2),
                  **_frame_feeds(ocv, _LB3 if nc == 3 else _LB2)}
        oref = C.Reduce.ORTHONORMALIZE
        out.append((B.basis_prod(B.dual_basis(B.list_basis(*odv, reduce=oref)),
                                 B.list_basis(*ocv, reduce=oref)), ofeeds))
    return out


def _primal_space(space):
    """The primal of a (possibly Dual) space — so a Dual(V) dom slot's list basis is built over V."""
    import grassmann.core.space as S
    return space.space if isinstance(space, S.Dual) else space


def _machine_coords(v, space, basis, frame_feeds, std_value):
    """``std_value`` expressed in ``basis`` — derived by the lowering's OWN cob (no hand-transform):
    represent the bare var pinned to default, target=``basis``, run on the std value."""
    prog = represent(v, basis_env={v.uid: default_basis(space)}, target=basis)
    return _run(prog, {_vk(v): std_value, **frame_feeds})


# ---------------------------------------------------------------------------
# The sweep: two-route match-OR-raise per (case × flavor-combination).
# ---------------------------------------------------------------------------


def _sweep_one(make_term, var_specs, flavor_choice):
    """One non-default flavor per operand (``flavor_choice``: var -> (basis, feeds)).  Assert
    two-route match OR a loud RepresentError; FAIL only on a silent mismatch."""
    term = make_term()
    # route_REF: every operand default, AUTO.
    ref_env = {v.uid: default_basis(sp) for v, (sp, _val) in var_specs.items()}
    ref_feeds = {_vk(v): val for v, (sp, val) in var_specs.items()}
    try:
        ref = _run(represent(term, basis_env=ref_env, target=B.AUTO), ref_feeds)
    except RepresentError:
        return  # the default case itself bailed — nothing to compare (rare; tolerate)
    # route_STD: operands pinned to their chosen non-default basis, fed machine-derived coords.
    std_env, std_feeds = {}, {}
    for v, (sp, val) in var_specs.items():
        basis, feeds = flavor_choice[v]
        std_env[v.uid] = basis
        std_feeds[_vk(v)] = _machine_coords(v, sp, basis, feeds, val)
        std_feeds.update(feeds)
    try:
        got = _run(represent(term, basis_env=std_env, target=B.STD), std_feeds)
    except RepresentError:
        return  # a LOUD bail is acceptable — not silent-wrong.
    assert np.allclose(got, ref), (
        f"SILENT-WRONG basis tag: target=STD gave {got} but the default-operand reference is {ref}"
    )


def _sweep(make_term, var_specs):
    """Sweep every combination of one-non-default-flavor-per-operand."""
    import itertools
    per_var = {v: _flavors(sp, f"{_vk(v)}_") for v, (sp, _val) in var_specs.items()}
    if any(not flavs for flavs in per_var.values()):
        pytest.skip("a flavor could not be built for some operand space (e.g. FFS) — covered elsewhere")
    vars_ = list(var_specs)
    for combo in itertools.product(*(per_var[v] for v in vars_)):
        _sweep_one(make_term, var_specs, dict(zip(vars_, combo)))


# ---------------------------------------------------------------------------
# The case registry — one entry per arm.  (make_term re-builds fresh vars each call to avoid any
# cross-call object reuse; var_specs pins the operands by their builder-returned Var.)
# ---------------------------------------------------------------------------

V = C.base("V", 3)
W = C.base("W", 2)
DV = C.dual(V)
MS = C.linear_map_space(V, V)


def _vec(name, sp=V):
    return C.var(name, sp)


# Each entry: (id, builder) where builder() -> (term, {var: (space, std_value)}).
def _c_scale():
    u = _vec("u")
    return C.scale(2.0, u), {u: (V, np.array([1.0, 2, 3]))}


def _c_sum():
    u, w = _vec("u"), _vec("w")
    return C.sum_(u, w), {u: (V, np.array([1.0, 2, 3])), w: (V, np.array([0.0, 1, 4]))}


def _c_oplus():
    u, w = _vec("u"), _vec("w2", V)
    return C.oplus_v(u, w), {u: (V, np.array([1.0, 2, 3])), w: (V, np.array([0.0, 1, 4]))}


def _c_oprod():
    u, w = _vec("u"), _vec("w")
    return C.oprod_v(u, w), {u: (V, np.array([1.0, 0, 2])), w: (V, np.array([0.0, 1, 3]))}


def _c_eval_covector():
    a, v = C.var("a", DV), _vec("v")
    return C.eval_(a, (0,), v), {a: (DV, np.array([1.0, 2, 3])), v: (V, np.array([2.0, 0, 1]))}


def _c_eval_map():
    f, v = C.var("f", MS), _vec("v")
    return C.eval_(f, (0,), v), {f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])),
                                 v: (V, np.array([1.0, 2, 3]))}


def _c_compose():
    f, g = C.var("f", MS), C.var("g", MS)
    Mf = np.array([[2.0, 0, 1], [1, 1, 0], [0, 3, 1]])
    Mg = np.array([[1.0, 2, 0], [0, 1, 1], [1, 0, 2]])
    return C.compose(f, (0,), g), {f: (MS, Mf), g: (MS, Mg)}


def _c_trace():
    a, b, c = _vec("a"), C.var("b", DV), _vec("c")
    return (C.trace(C.oprod_v(a, b, c), 0, 1),
            {a: (V, np.array([1.0, 2, 0])), b: (DV, np.array([0.0, 1, 3])), c: (V, np.array([2.0, 1, 1]))})


def _c_dagger():
    f = C.var("f", MS)
    return C.dagger(f), {f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]]))}


def _c_solve():
    f = C.var("f", MS, C.bij)
    return C.solve(f), {f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]]))}


def _c_pinv():
    f = C.var("f", MS)
    return C.pinv(f), {f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]]))}


def _c_pinv_apply():
    f, v = C.var("f", MS), _vec("v")
    return C.pinv_apply(f, v), {f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])),
                                v: (V, np.array([1.0, 0, 2]))}


def _c_permute():
    op = C.var("m", C.oprod(V, V))
    return C.permute(op, (1, 0)), {op: (C.oprod(V, V), np.arange(9.0).reshape(3, 3))}


def _c_kron():
    f, g = C.var("f", MS), C.var("g", MS)
    Mf = np.array([[2.0, 0, 1], [1, 1, 0], [0, 3, 1]])
    Mg = np.array([[1.0, 2, 0], [0, 1, 1], [1, 0, 2]])
    return C.kron(f, g), {f: (MS, Mf), g: (MS, Mg)}


def _c_lift():
    op = C.var("m", C.oprod(V, V))
    return C.lift(op, (0, 1)), {op: (C.oprod(V, V), np.arange(9.0).reshape(3, 3))}


def _c_llam_map():
    f, x = C.var("f", MS), _vec("x")
    return C.llam(x, body=C.eval_(f, (0,), x)), {f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]]))}


def _c_llam_covector():
    a, x = C.var("a", DV), _vec("x")
    return C.llam(x, body=C.eval_(a, (0,), x)), {a: (DV, np.array([1.0, 2, 3]))}


def _c_dot():  # metric full contraction — exercises _reconcile_metric_operands (UNSTEERED, scalar)
    a, b = _vec("a"), _vec("b")
    return C.eval_(C.inner(V), (0, 1), a, b), {a: (V, np.array([1.0, 2, 3])), b: (V, np.array([2.0, 0, 1]))}


def _c_flat_apply():
    v = _vec("v")
    return C.eval_(C.flat(V), (0,), v), {v: (V, np.array([1.0, 2, 3]))}


def _c_sharp_apply():
    co = C.var("co", DV)
    return C.eval_(C.sharp(V), (0,), co), {co: (DV, np.array([1.0, 2, 3]))}


# Scalar-wrapped variants ⟨w, …⟩: a scalar result can't be steered, so these exercise the UNSTEERED
# contraction/reconcile path that `target=STD` steering masks for eval-map/compose/pinv_apply.
def _c_eval_map_scalar():
    w, f, v = C.var("w", DV), C.var("f", MS), _vec("v")
    return (C.eval_(w, (0,), C.eval_(f, (0,), v)),
            {w: (DV, np.array([1.0, 2, 3])), f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])),
             v: (V, np.array([1.0, 0, 2]))})


def _c_compose_scalar():
    w, f, g, v = C.var("w", DV), C.var("f", MS), C.var("g", MS), _vec("v")
    Mf = np.array([[2.0, 0, 1], [1, 1, 0], [0, 3, 1]])
    Mg = np.array([[1.0, 2, 0], [0, 1, 1], [1, 0, 2]])
    return (C.eval_(w, (0,), C.eval_(C.compose(f, (0,), g), (0,), v)),
            {w: (DV, np.array([1.0, 2, 3])), f: (MS, Mf), g: (MS, Mg), v: (V, np.array([1.0, 0, 2]))})


def _c_pinv_apply_scalar():
    w, f, v = C.var("w", DV), C.var("f", MS), _vec("v")
    return (C.eval_(w, (0,), C.pinv_apply(f, v)),
            {w: (DV, np.array([1.0, 2, 3])), f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])),
             v: (V, np.array([1.0, 0, 2]))})


def _c_var():
    u = _vec("u")
    return u, {u: (V, np.array([1.0, 2, 3]))}


def _c_external():
    from grassmann.lower.represent import BasisReq, ExternalOp
    neg = ExternalOp(name="neg", fn=np.negative, basis_req=BasisReq.EQUIVARIANT)
    v = _vec("v")
    return C.external(V, neg, v), {v: (V, np.array([1.0, 2, 3]))}


def _c_solve_apply():
    f, v = C.var("f", MS, C.bij), _vec("v")
    return C.solve_apply(f, v), {f: (MS, np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])),
                                 v: (V, np.array([1.0, 0, 2]))}


def _c_assert():
    v = _vec("v")
    return C.assert_(v, V), {v: (V, np.array([1.0, 2, 3]))}


def _c_project():
    v = _vec("v")
    return C.project(v, V), {v: (V, np.array([1.0, 2, 3]))}


def _c_embed():
    v = _vec("v")
    return C.embed(v, V), {v: (V, np.array([1.0, 2, 3]))}


def _case(name, builder, xfail=None):
    marks = [pytest.mark.xfail(reason=xfail, strict=True)] if xfail else []
    return pytest.param(name, builder, marks=marks, id=name)


CASES = [
    _case("scale", _c_scale), _case("sum", _c_sum), _case("oplus", _c_oplus), _case("oprod", _c_oprod),
    _case("eval_covector", _c_eval_covector), _case("eval_map", _c_eval_map), _case("compose", _c_compose),
    _case("trace", _c_trace), _case("dagger", _c_dagger), _case("solve", _c_solve), _case("pinv", _c_pinv),
    _case("pinv_apply", _c_pinv_apply), _case("permute", _c_permute), _case("kron", _c_kron),
    _case("lift", _c_lift), _case("llam_map", _c_llam_map), _case("llam_covector", _c_llam_covector),
    _case("dot", _c_dot), _case("project", _c_project), _case("embed", _c_embed),
    _case("flat_apply", _c_flat_apply),  # metric Eval builds + tags in the operand basis
    _case("sharp_apply", _c_sharp_apply),  # + covector-operand variance (_covector_primal)
    _case("var", _c_var), _case("external", _c_external), _case("solve_apply", _c_solve_apply),
    _case("assert", _c_assert),
    # scalar-wrapped (UNSTEERED) variants — exercise the contraction/reconcile that STD steering masks
    _case("eval_map_scalar", _c_eval_map_scalar), _case("compose_scalar", _c_compose_scalar),
    _case("pinv_apply_scalar", _c_pinv_apply_scalar),
]


@pytest.mark.parametrize("name,builder", CASES)
def test_basis_honesty_sweep(name, builder):
    """No arm is silent-wrong: a non-default operand basis either matches the std reference or raises."""
    term, var_specs = builder()
    _sweep(lambda: term, var_specs)
