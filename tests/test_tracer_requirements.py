"""Task-3 acceptance: unlimited-derivative tracers + requirements enumeration.

pointwise inserts each field/geometry as a TRACER variable (``deps[uid] = None``) with unlimited
derivatives available, ``diff`` the whole compound term (possibly more than once, to arbitrary
order), then a requirements pass reads off WHICH derivatives were actually referenced — so it can
sample exactly those from chartlib and prune the rest.

These tests pin:
* ``deps=None`` allocates a fresh symbolic derivative-map input ``∂source/∂x : X ⊸ space(source)``;
* :func:`enumerate_derivative_inputs` reads the referenced inputs with structured provenance
  ``(base tracer var, derivative order)`` — by var identity, NOT by parsing the ``d_…`` name;
* it is composable to arbitrary order (nested ``diff`` ⇒ order-2 inputs, reported as ``order=2``);
* the allocation is DEDUPED — the derivative of a given ``(source, x)`` is one input reused across
  repeated/nested ``diff`` calls, so the enumerated requirements are a true set;
* pruning unreferenced tracers leaves a valid program (feeding only the enumerated inputs runs).
"""

from __future__ import annotations

import numpy as np

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.transform import (
    clear_deriv_inputs,
    diff,
    enumerate_derivative_inputs,
)


def _prog_input_names(term: T.Term) -> set[str]:
    prog = G.compile(term)
    return {getattr(i, "name", "") for i in prog.inputs}


def test_tracer_allocates_provenance_tracked_input() -> None:
    clear_deriv_inputs()
    Rt, W = S.base("Rt", 2), S.base("W", 3)
    x = T.var("x", Rt)
    u, v = T.var("u", W), T.var("v", W)
    e = G.dot(u, v).node  # ⟨u, v⟩

    jd = diff(e, x, deps={u.uid: None, v.uid: None})
    recs = enumerate_derivative_inputs(jd)

    # one first-order input per tracer, with correct provenance + map space X ⊸ space(source).
    by_source = {r.source.uid: r for r in recs}
    assert set(by_source) == {u.uid, v.uid}
    for src in (u, v):
        r = by_source[src.uid]
        assert r.order == 1
        assert r.var.space == S.linear_map_space(Rt, W)
        assert r.source is src

    # the enumerated vars are exactly the derivative inputs the lowered program asks for.
    names = _prog_input_names(jd)
    for r in recs:
        assert f"{r.var.name}_{r.var.uid}" in names


def test_enumeration_is_arbitrary_order_and_composable() -> None:
    clear_deriv_inputs()
    Rt, W = S.base("Rt", 2), S.base("W", 3)
    x = T.var("x", Rt)
    u, v = T.var("u", W), T.var("v", W)
    e = G.dot(u, v).node

    j1 = diff(e, x, deps={u.uid: None, v.uid: None})
    order1 = enumerate_derivative_inputs(j1)

    # differentiate AGAIN, marking the tracers and their order-1 inputs as further unknowns.
    deps2 = {u.uid: None, v.uid: None, **{r.var.uid: None for r in order1}}
    j2 = diff(j1.body, x, deps=deps2)
    recs2 = enumerate_derivative_inputs(j2)

    got = sorted((r.source.name, r.order) for r in recs2)
    assert got == [("u", 1), ("u", 2), ("v", 1), ("v", 2)]
    # the order-1 inputs are the SAME objects (deduped, not re-allocated by the 2nd diff).
    o1_uids = {r.var.uid for r in order1}
    assert o1_uids <= {r.var.uid for r in recs2}


def test_derivative_inputs_are_deduped_across_calls() -> None:
    clear_deriv_inputs()
    Rt, W = S.base("Rt", 2), S.base("W", 3)
    x = T.var("x", Rt)
    u, v = T.var("u", W), T.var("v", W)
    e = G.dot(u, v).node

    a = enumerate_derivative_inputs(diff(e, x, deps={u.uid: None, v.uid: None}))
    b = enumerate_derivative_inputs(diff(e, x, deps={u.uid: None, v.uid: None}))
    assert {r.var.uid for r in a} == {r.var.uid for r in b}  # a true set, no duplicate inputs


def test_diff_twice_in_two_directions_then_evaluate() -> None:
    """The single Jacobian map, evaluated at two different seed directions, gives the two
    directional derivatives — feeding exactly the enumerated tracer-derivative inputs."""
    clear_deriv_inputs()
    Rt, W = S.base("Rt", 2), S.base("W", 3)
    x = T.var("x", Rt)
    u, v = T.var("u", W), T.var("v", W)
    e = G.dot(u, v).node

    jd = diff(e, x, deps={u.uid: None, v.uid: None})
    recs = enumerate_derivative_inputs(jd)
    seed = jd.vars[0]

    rng = np.random.default_rng(7)
    uc, vc = rng.standard_normal(3), rng.standard_normal(3)
    Du = rng.standard_normal((2, 3))   # ∂u/∂x sampled
    Dv = rng.standard_normal((2, 3))   # ∂v/∂x sampled
    src_val = {u.uid: Du, v.uid: Dv}
    feed = {f"u_{u.uid}": uc, f"v_{v.uid}": vc}
    for r in recs:  # sample only the enumerated derivative inputs
        feed[f"{r.var.name}_{r.var.uid}"] = src_val[r.source.uid]

    body = G.compile(jd.body)
    for direction in (np.array([1.0, 0.0]), np.array([0.3, -0.7])):
        got = float(np.asarray(body.run({**feed, f"{seed.name}_{seed.uid}": direction})["result"]))
        # d⟨u,v⟩·h = ⟨Du·h, v⟩ + ⟨u, Dv·h⟩
        want = (Du.T @ direction) @ vc + uc @ (Dv.T @ direction)
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_pruning_unreferenced_tracers_leaves_valid_program() -> None:
    """A tracer that does not appear in the differentiated term allocates NO derivative input;
    enumeration reflects only what is referenced, and the program runs feeding only those."""
    clear_deriv_inputs()
    Rt, W = S.base("Rt", 2), S.base("W", 3)
    x = T.var("x", Rt)
    u, v = T.var("u", W), T.var("v", W)
    e = T.Eval(T.inner(W), (0, 1), (u, u))  # ⟨u, u⟩ — references u only, NOT v

    jd = diff(e, x, deps={u.uid: None, v.uid: None})  # v is a tracer but unused
    recs = enumerate_derivative_inputs(jd)
    assert [(r.source.name, r.order) for r in recs] == [("u", 1)]  # v pruned

    # the program runs with only u + its derivative fed (v / d_v never required).
    seed = jd.vars[0]
    rng = np.random.default_rng(3)
    uc, Du, direction = rng.standard_normal(3), rng.standard_normal((2, 3)), rng.standard_normal(2)
    feed = {f"u_{u.uid}": uc, f"{seed.name}_{seed.uid}": direction}
    feed[f"{recs[0].var.name}_{recs[0].var.uid}"] = Du
    got = float(np.asarray(G.compile(jd.body).run(feed)["result"]))
    np.testing.assert_allclose(got, 2.0 * (uc @ (Du.T @ direction)), atol=1e-9)
