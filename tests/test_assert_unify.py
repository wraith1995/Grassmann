"""Asserted facts aiding unification / inference (A3).

``Assert(v, s)`` and the asserted map properties (``inj``/``surj``/``bij``/``rank``)
carried by a map's ``Var`` feed information into inference:

- structural unification of ``Assert``'s space **recovers implicit space metavars**;
- a map's asserted **bijectivity** forces ``|dom| ≡ |cod|`` (an isomorphism), so an
  assertion / ``Solve`` between unequal-dimension spaces is rejected;
- an asserted **rank** pins ``δ_f`` and cross-checks the inj/surj/bij identities.

These tests exercise both the *aid* (metavar recovery, dim facts) and the *soundness*
(rejecting an assertion that contradicts an asserted property).
"""

from __future__ import annotations

import pytest

import grassmann.core as G
from grassmann.infer import (
    AssertFactError,
    InferError,
    assert_property_dims,
    check_assert_property_consistency,
    consistent_dim_facts,
    dim_eq,
    map_facts_for_unify,
    synth_space,
)
from grassmann.infer.synth import Ctx, map_domain_codomain
from grassmann.infer.unify import fresh_meta

n = G.DimVar("n")
m = G.DimVar("m")
V = G.base("V", n)
W = G.base("W", m)
U = G.base("U", n)  # |U| = |V| = n  (same dim, different space)
R = G.scalar()


# ---------------------------------------------------------------------------
# Assert recovers implicit space metavars (the basic unification aid).
# ---------------------------------------------------------------------------


def test_assert_pins_open_metavar():
    """``Assert(x, V)`` with ``x : ?a`` solves ``?a := V`` in the store."""
    ctx = Ctx()
    meta = fresh_meta("a")
    x = G.var("x", meta)
    assert synth_space(G.assert_(x, V), ctx) == V
    assert ctx.store.resolve(meta) == V


def test_assert_into_map_space_recovers_factors():
    """Asserting a partially-open map space pins the open factor."""
    ctx = Ctx()
    meta = fresh_meta("cod")
    f = G.var("f", G.linear_map_space(V, meta))
    res = synth_space(G.assert_(f, G.linear_map_space(V, W)), ctx)
    assert res == G.linear_map_space(V, W)
    assert ctx.store.resolve(meta) == W


# ---------------------------------------------------------------------------
# Bijective ⇒ |dom| ≡ |cod| (isomorphism); inconsistent assert rejected.
# ---------------------------------------------------------------------------


def test_bij_iso_forces_dim_equality_solve_ok():
    """A genuine iso ``V ⊸ V`` (bij) gives ``Solve : V ⊸ V`` with no complaint."""
    f = G.var("f", G.linear_map_space(V, V), props=G.bij)
    assert synth_space(G.solve(f)) == G.linear_map_space(V, V)


def test_bij_between_unequal_dims_rejected_in_solve():
    """``Solve`` of a ``bij`` map asserted ``V ⊸ W`` with ``n ≢ m`` is rejected (A3)."""
    f = G.var("f", G.linear_map_space(V, W), props=G.bij)
    with pytest.raises(AssertFactError):
        synth_space(G.solve(f))


def test_bij_between_unequal_dims_rejected_in_assert():
    f = G.var("f", G.linear_map_space(V, W), props=G.bij)
    with pytest.raises(AssertFactError):
        synth_space(G.assert_(f, G.linear_map_space(V, W)))


def test_assert_fact_error_is_infer_error():
    """``AssertFactError`` is an ``InferError`` (caught by generic handling)."""
    f = G.var("f", G.linear_map_space(V, W), props=G.bij)
    with pytest.raises(InferError):
        synth_space(G.solve(f))


def test_bij_equal_dim_distinct_spaces_ok():
    """``V ⊸ U`` with ``|V| = |U| = n`` is a legitimate iso ⇒ accepted (dim, not space)."""
    f = G.var("f", G.linear_map_space(V, U), props=G.bij)
    # dims agree (both n); the assertion is consistent even though V ≠ U structurally.
    res = synth_space(G.assert_(f, G.linear_map_space(V, U)))
    assert res == G.linear_map_space(V, U)


# ---------------------------------------------------------------------------
# Property-implied dimension facts (the data the unifier path reads).
# ---------------------------------------------------------------------------


def test_bij_dim_fact_is_dom_eq_cod():
    f = G.var("f", G.linear_map_space(V, U), props=G.bij)
    dom, cod = map_domain_codomain(f)
    facts = assert_property_dims(f, dom, cod)
    assert any("bij" in fact.why for fact in facts)
    # The bij fact is |dom| ≡ |cod|; here both are n ⇒ no violation.
    assert consistent_dim_facts(facts) == []


def test_bij_dim_fact_violation_detected():
    f = G.var("f", G.linear_map_space(V, W), props=G.bij)
    dom, cod = map_domain_codomain(f)
    bad = consistent_dim_facts(assert_property_dims(f, dom, cod))
    assert len(bad) == 1
    assert not dim_eq(bad[0].lhs, bad[0].rhs)


def test_asserted_rank_cross_checks_bij_identities():
    """A ``bij`` map also given ``rank n`` agrees with ``|dom| = |cod| = n``."""
    f = G.var("f", G.linear_map_space(V, U), props=G.props.props(bij=True, rank=n))
    dom, cod = map_domain_codomain(f)
    facts = assert_property_dims(f, dom, cod)
    # rank ≡ |dom| and rank ≡ |cod| both hold.
    assert consistent_dim_facts(facts) == []


def test_asserted_rank_inconsistent_with_bij_rejected():
    """``bij`` ⇒ rank = |dom|; asserting a different rank is a contradiction."""
    bad_rank = G.DimVar("k")  # k ≢ n
    f = G.var("f", G.linear_map_space(V, V), props=G.props.props(bij=True, rank=bad_rank))
    dom, cod = map_domain_codomain(f)
    bad = consistent_dim_facts(assert_property_dims(f, dom, cod))
    assert bad  # rank k ≢ |dom| n
    with pytest.raises(AssertFactError):
        check_assert_property_consistency(f, dom, cod)


def test_inj_and_surj_facts():
    inj_f = G.var("h", G.linear_map_space(V, W), props=G.inj)
    surj_f = G.var("g", G.linear_map_space(V, W), props=G.surj)
    d1 = map_domain_codomain(inj_f)
    d2 = map_domain_codomain(surj_f)
    assert any("inj" in f.why for f in assert_property_dims(inj_f, *d1))
    assert any("surj" in f.why for f in assert_property_dims(surj_f, *d2))


def test_no_props_no_facts():
    """An unannotated map (F3 runtime-rank path) contributes no static dim facts."""
    f = G.var("f", G.linear_map_space(V, W))  # no props
    assert assert_property_dims(f, *map_domain_codomain(f)) == []
    assert map_facts_for_unify(f) == []


def test_map_facts_for_unify_decodes_self():
    f = G.var("f", G.linear_map_space(V, V), props=G.bij)
    facts = map_facts_for_unify(f)
    assert facts and all(dim_eq(fact.lhs, fact.rhs) for fact in facts)
