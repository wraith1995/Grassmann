"""The resource calculus that gives :class:`~grassmann.core.term.LLam` its linear discipline.

A linear binder ``λ⊸(x). e`` promises that ``e`` consumes each bound variable **exactly
once**. This module decides that promise: it computes, per linear-bound variable, how many
times the variable is consumed along every evaluation path of the body, and rejects the
body unless that count is one. The core's own ``LLam`` constructor keeps a coarser
structural guard alongside it — a linear var may not appear in a dependence position (a
``Lam`` binder-domain annotation, or an explicit space/nat argument of an ``NEval``), and
must occur at least once — but "occurs at least once" lets a var used twice, or used in
one branch and not another, through. The counting here is what makes the discipline real.

The two combinators
-------------------
Every node is one of two kinds, and the kind decides how its children's usage combines::

    multiplicative (⊗-like)             additive (branches)

         node                                node
        ╱  |  ╲                             ╱  |  ╲
       a   b   c                           a   b   c

    counts add                          counts must agree
    usage = usage(a) + usage(b)         usage = usage(a) = usage(b)
                     + usage(c)                         = usage(c)

A **multiplicative** node's children all lie on the *same* evaluation path, so their usage
counts add: ``Eval`` (the map and every argument), ``Compose``, ``Scale``, ``Kron``,
``OProdV``, the inverses, and every other node not named below. An **additive** node's
children are alternative branches of one value, so each branch must consume the tracked
variables identically and the node's usage is that shared count.

Two nodes are additive. A ``Sum`` is additive because its summands are alternatives:
``Σ_σ sgn(σ)·(v_{σ0}⊗…)`` uses every ``vᵢ`` exactly once *in each* summand, so the whole
sum uses each ``vᵢ`` exactly once — this is precisely what lets the antisymmetrizer
``Alt`` typecheck as a linear map. An ``OPlusV`` is additive because its slots are the
parallel, independent components of ``⊕`` rather than a shared path: the diagonal
``h ↦ h ⊕ h`` consumes ``h`` once, not twice.

``Zero`` is the unit of the calculus: it consumes nothing, an empty :class:`Usage`. That
is what makes ``x ↦ x ⊕ 0`` well formed — the ``Zero`` slot imposes no constraint on the
other slots, so the usage is the single ordinary use of ``x``. Nonlinearity inside a slot
is still caught: ``h ⊗ h`` is multiplicative within the slot, counts two, and fails the
outer exactly-once check.

A nested binder (``LLam`` or ``Lam``) binds its own vars, each with a fresh ``uid``, so it
cannot shadow an outer linear var. The value it denotes captures the outer var once, so a
binder contributes the usage of its single body child multiplicatively.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from grassmann.core.space import is_space
from grassmann.core.term import (
    LLam,
    OPlusV,
    Sum,
    Var,
    Zero,
    is_term,
)


class LinearityError(Exception):
    """Raised when an ``LLam`` body consumes a linear variable other than exactly once."""


@dataclass(frozen=True)
class Usage:
    """How many times each tracked linear variable is consumed, keyed by ``uid``.

    A multiset of consumptions. The two combinators of the calculus act on it: a
    multiplicative node adds two usages, an additive node requires them to be equal.
    """

    counts: Counter[int]

    @staticmethod
    def empty() -> Usage:
        """Give the empty usage — nothing consumed."""
        return Usage(Counter())

    @staticmethod
    def single(uid: int) -> Usage:
        """Give the usage consuming the var ``uid`` exactly once."""
        return Usage(Counter({uid: 1}))


def _mul(a: Usage, b: Usage) -> Usage:
    """Combine two usages multiplicatively: the counts add, since both lie on one path."""
    out = Counter(a.counts)
    out.update(b.counts)
    return Usage(out)


def _mul_all(us: list[Usage]) -> Usage:
    out: Counter[int] = Counter()
    for u in us:
        out.update(u.counts)
    return Usage(out)


def _add_merge(branches: list[Usage], names: dict[int, str]) -> Usage:
    """Combine branch usages additively, requiring every branch to agree.

    Returns the usage the branches share. A disagreement means a linear variable is
    consumed in one branch and not another, or a different number of times, which breaks
    linearity: ``v₁ + v₂`` must be linear in each tracked argument, so a variable used
    once in ``v₁`` must be used once in ``v₂`` as well.
    """
    if not branches:
        return Usage.empty()
    first = branches[0]
    for other in branches[1:]:
        keys = set(first.counts) | set(other.counts)
        for uid in keys:
            if first.counts.get(uid, 0) != other.counts.get(uid, 0):
                nm = names.get(uid, f"?{uid}")
                raise LinearityError(
                    f"linear variable {nm!r} is used a different number of times "
                    f"across the branches of a Sum "
                    f"({first.counts.get(uid, 0)} vs {other.counts.get(uid, 0)}); "
                    f"each additive branch must consume the same linear resources"
                )
    return first


def _usage(node: object, tracked: set[int], names: dict[int, str]) -> Usage:
    """Compute the resource usage of ``node`` w.r.t. the ``tracked`` linear var uids."""
    # A Zero vector consumes nothing — the unit of the resource calculus.
    if isinstance(node, Zero):
        return Usage.empty()

    # A tracked linear var is one consumption; an untracked var/leaf is empty.
    if isinstance(node, Var):
        if node.uid in tracked:
            return Usage.single(node.uid)
        return Usage.empty()

    # A Sum is additive: its summands are alternative branches, each of which must consume
    # the tracked vars identically.
    if isinstance(node, Sum):
        branches = [_usage(v, tracked, names) for v in node.vectors]
        return _add_merge(branches, names)

    # A direct sum ``OPlusV`` is additive too: its slots are the parallel, independent
    # components of ``⊕`` rather than one shared path. So the diagonal ``h ⊕ h`` (as ``d(x⊕x)``
    # builds) consumes ``h`` once, and the injection ``h ⊕ 0`` — the derivative of a slot that
    # does not depend on x — is well formed. A ``Zero`` slot is the additive unit and imposes no
    # constraint; the remaining slots must agree.
    if isinstance(node, OPlusV):
        branches = [_usage(v, tracked, names) for v in node.vectors if not isinstance(v, Zero)]
        return _add_merge(branches, names)

    # A nested LLam binds its own fresh-uid vars, so it cannot shadow an outer linear var; the
    # outer vars it mentions are captured once by the value it denotes. Recurse into the body
    # multiplicatively.
    if isinstance(node, LLam):  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): binder cases else generic multiplicative recursion [approved 2026-07-03]
        return _usage(node.body, tracked, names)

    # Everything else is multiplicative: every child Term contributes to the same path, so
    # their usages add. Embedded Spaces never consume a linear value — a tracked var inside a
    # Space annotation is a dependence-position violation the core's structural guard already
    # rejects — so they are not counted here.
    return _mul_all([_usage(c, tracked, names) for c in _term_children(node)])


def _term_children(node: object) -> list[object]:
    """Collect the child *Terms* of ``node`` (Spaces are not resource-bearing; skipped)."""
    if not is_term(node):
        return []
    out: list[object] = []
    for fname in getattr(node, "__dataclass_fields__", {}):
        val = getattr(node, fname)
        items = val if isinstance(val, tuple) else (val,)
        for c in items:
            if is_term(c) and not is_space(c):
                out.append(c)
    return out


def check_llam_resources(node: LLam) -> None:
    """Enforce the exactly-once discipline for the variables one ``LLam`` binds.

    Each variable bound by ``node`` must be consumed exactly once along every path of the
    body, counted by the multiplicative/additive rules of this module. Raises
    :class:`LinearityError` naming the variable on either under- or over-use.
    """
    tracked = {v.uid for v in node.vars}
    names = {v.uid: v.name for v in node.vars}
    usage = _usage(node.body, tracked, names)
    for v in node.vars:
        n = usage.counts.get(v.uid, 0)
        if n == 0:
            raise LinearityError(
                f"linear variable {v.name!r} is never used; a linear map must consume "
                f"each linear argument exactly once"
            )
        if n > 1:
            raise LinearityError(
                f"linear variable {v.name!r} is used {n} times; a linear map must "
                f"consume each linear argument exactly once"
            )


def check_linearity(term: object) -> None:
    """Enforce the resource discipline on every ``LLam`` anywhere in ``term``.

    Walks the whole term and applies :func:`check_llam_resources` at each linear binder it
    finds, so a violation nested inside an argument is caught as readily as one at the root.
    """
    if not is_term(term):
        return
    if isinstance(term, LLam):
        check_llam_resources(term)
    for c in _term_children(term):
        check_linearity(c)


__all__ = [
    "LinearityError",
    "Usage",
    "check_llam_resources",
    "check_linearity",
]
