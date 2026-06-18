"""Resource-based linearity checking for ``LLam`` (C1+C2).

The inference layer enforces a real **usage / resource** discipline on the linear binder
``LLam``: each linear-bound variable must be consumed *exactly once* along every
evaluation path. These tests cover:

- the exactly-once discipline (under-use, over-use rejected);
- the **additive** ``Sum`` rule (each summand must consume the same resources — so
  ``Σ_σ`` of permutations is fine, but mismatched branches are not);
- the **C2** requirement that ``x ↦ x ⊕ 0`` (a ``Zero`` summand) is a single ordinary use
  of ``x`` and typechecks as linear-identity-like;
- that already-valid programs (the §26 antisymmetrizer / pullback) still pass.
"""

from __future__ import annotations

import math

import pytest

import grassmann.core as G
from grassmann.infer import (
    LinearityError,
    check_llam_resources,
    synth_space,
)
from grassmann.infer.linearity import Usage, check_linearity

n = G.DimVar("n")
V = G.base("V", n)
W = G.base("W", G.DimVar("m"))
R = G.scalar()


# ---------------------------------------------------------------------------
# C2 — x ↦ x ⊕ 0 is a single use (Zero consumes nothing).
# ---------------------------------------------------------------------------


def test_inject_into_oplus_with_zero_is_single_use():
    """``x ↦ x ⊕ 0``: the ``Zero`` summand consumes nothing, so ``x`` is used once."""
    x = G.var("x", V)
    body = G.oplus_v(x, G.zero(W))  # x ⊕ 0_W
    llam = G.llam(x, body=body)
    # No LinearityError; the space is the linear-map space into V⊕W.
    s = synth_space(llam)
    assert s == G.linear_map_space(V, G.oplus(V, W))


def test_zero_first_component_also_single_use():
    """``x ↦ 0 ⊕ x`` is equally fine (ι₁ injection)."""
    x = G.var("x", V)
    llam = G.llam(x, body=G.oplus_v(G.zero(W), x))
    assert synth_space(llam) == G.linear_map_space(V, G.oplus(W, V))


def test_usage_of_zero_is_empty():
    z = G.zero(V)
    # A bare Zero consumes nothing of any tracked var.
    from grassmann.infer.linearity import _usage

    assert _usage(z, {123}, {123: "x"}) == Usage.empty()


# ---------------------------------------------------------------------------
# Exactly-once: under-use and over-use.
# ---------------------------------------------------------------------------


def test_identity_uses_var_once():
    x = G.var("x", V)
    check_llam_resources(G.llam(x, body=x))  # no raise


def test_multiplicative_double_use_rejected():
    """``x ↦ x ⊗ x`` consumes ``x`` twice (multiplicative) ⇒ rejected."""
    x = G.var("x", V)
    llam = G.llam(x, body=G.oprod_v(x, x))
    with pytest.raises(LinearityError):
        synth_space(llam)


def test_double_use_via_eval_rejected():
    """A map applied to ``x`` and a tensor with ``x`` again ⇒ two uses."""
    # f : V ⊸ V ; body = f(x) ⊗ x  uses x twice.
    f = G.var("f", G.linear_map_space(V, V))
    x = G.var("x", V)
    body = G.oprod_v(G.eval_(f, (0,), x), x)
    with pytest.raises(LinearityError):
        synth_space(G.llam(x, body=body))


# ---------------------------------------------------------------------------
# Additive Sum rule.
# ---------------------------------------------------------------------------


def test_sum_of_uses_is_one_use():
    """``x ↦ x + x``: additive branches each use ``x`` once ⇒ a single linear use."""
    x = G.var("x", V)
    # Both summands are in space V; Sum unifies them.
    llam = G.llam(x, body=G.sum_(x, x))
    assert synth_space(llam) == G.linear_map_space(V, V)


def test_sum_branch_disagreement_rejected():
    """Branches that consume the linear vars differently are rejected (additive rule)."""
    x = G.var("x", V)
    y = G.var("y", V)
    # (x ⊗ y) + (x ⊗ x): the second branch uses x twice / y zero ⇒ disagreement.
    bad = G.sum_(G.oprod_v(x, y), G.oprod_v(x, x))
    with pytest.raises(LinearityError):
        synth_space(G.llam(x, y, body=bad))


# ---------------------------------------------------------------------------
# The §26 worked examples still pass (no regression; precision, not rejection).
# ---------------------------------------------------------------------------


def test_antisymmetrizer_multivar_sum_of_perms_ok():
    """``Σ_σ sgn(σ)·(v_{σ0}⊗…⊗v_{σ(k-1)})``: every vᵢ used once per summand ⇒ linear."""
    k = 3
    vs = [G.var(f"v{i}", V) for i in range(k)]

    def sign(perm: tuple[int, ...]) -> int:
        inv = sum(
            1
            for i in range(len(perm))
            for j in range(i + 1, len(perm))
            if perm[i] > perm[j]
        )
        return 1 if inv % 2 == 0 else -1

    import itertools

    summands = [
        G.scale(sign(sigma), G.oprod_v(*[vs[sigma[j]] for j in range(k)]))
        for sigma in itertools.permutations(range(k))
    ]
    alt = G.llam(*vs, body=G.sum_(*summands))
    s = synth_space(alt)
    # ⊗(Dual V)*3 ⊗ (V⊗V⊗V)
    assert s == G.oprod(
        G.dual(V), G.dual(V), G.dual(V), G.oprod(V, V, V)
    )
    assert math.factorial(k) == 6  # sanity: 6 summands, each linear in every vᵢ


def test_pullback_inner_llam_ok():
    """The §26.1 inner ``LLam([x,y]. b(T x, T y))`` passes the resource check."""
    formW = G.oprod(G.dual(W), G.dual(W), R)
    T = G.var("T", G.linear_map_space(V, W))
    b = G.var("b", formW)
    x = G.var("x", V)
    y = G.var("y", V)
    Tx = G.eval_(T, (0,), x)
    Ty = G.eval_(T, (0,), y)
    body = G.eval_(b, (0, 1), Tx, Ty)
    inner = G.llam(x, y, body=body)
    check_llam_resources(inner)  # no raise
    # x and y each used exactly once.
    s = synth_space(inner)
    assert s == G.oprod(G.dual(V), G.dual(V), R)


def test_check_linearity_walks_nested_llams():
    """``check_linearity`` recurses and validates every nested ``LLam``."""
    # outer ⊸ over b, inner ⊸ over x,y — both linear-correct.
    formW = G.oprod(G.dual(W), G.dual(W), R)
    T = G.var("T", G.linear_map_space(V, W))
    b = G.var("b", formW)
    x = G.var("x", V)
    y = G.var("y", V)
    body = G.eval_(b, (0, 1), G.eval_(T, (0,), x), G.eval_(T, (0,), y))
    nested = G.llam(b, body=G.llam(x, y, body=body))
    check_linearity(nested)  # no raise
