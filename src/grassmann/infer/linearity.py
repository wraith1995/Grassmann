"""Resource / usage accounting for the linear binder :class:`~grassmann.core.term.LLam`.

The core (:mod:`grassmann.core.term`) ships a *structural* well-formedness guard for
``LLam``: it walks the body and (a) rejects a linear var appearing in a dependence
position (a ``Lam`` binder-domain annotation, or an explicit space/nat arg of an
``NEval``), and (b) requires each linear var to occur *at least once*. That guard is
coarse: "occurs at least once" is not linear discipline — a var used twice, or used in
one additive branch but not another, slips through.

This module replaces that coarse occurrence check with a real **resource calculus**: it
computes, per linear-bound variable, how many times the variable is *consumed* along
every evaluation path, and requires **exactly one** use of each linear var (the linear
discipline ``⊸``). It keeps the structural dependence-position guard (still enforced by
the core constructor) but makes the *counting* precise and multi-variable-aware.

The two combinators
-------------------
A term tree has two flavors of node for resource purposes:

- **Multiplicative** (``⊗``-like / "parallel"): the node's children all contribute to the
  *same* path, so their usage counts **add**. ``OProd``/``OPlus`` stacks, ``Eval`` (the
  map and every argument), ``Compose``, ``Scale``, ``Kron``, the inverses, etc.
- **Additive** (``&``-like / a choice between branches): the branches are alternative
  paths and must each consume **the same** resources. ``Sum`` is the one additive node:
  ``Σ_σ sgn(σ)·(v_{σ0}⊗…)`` uses every ``vᵢ`` exactly once *in each summand*, so the
  whole sum uses each ``vᵢ`` exactly once. We merge additive branches by requiring their
  usage maps to agree (rather than summing them).

``Zero`` is the unit that **consumes nothing**: ``Usage()`` (empty). This is what makes
the requirement ``x ↦ x ⊕ 0`` typecheck — ``OPlus(x, 0)`` is multiplicative, so its
usage is ``usage(x) + usage(0) = {x:1} + {} = {x:1}``: a single ordinary use of ``x``,
not a misuse or double use.

Nested binders (``LLam``/``Lam``) bind their *own* fresh vars (each a distinct ``uid``), so a
nested binder simply contributes the usage of *its body* w.r.t. the outer linear vars —
the value it denotes captures the outer var, which is one consumption. We therefore treat
binder bodies multiplicatively (a binder has a single body child).
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
    """Raised when a ``LLam`` body violates the linear (exactly-once) discipline."""


@dataclass(frozen=True)
class Usage:
    """A multiset of consumptions of the tracked linear vars: ``uid -> count``."""

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
    """Multiplicative combine: counts add (children share one path)."""
    out = Counter(a.counts)
    out.update(b.counts)
    return Usage(out)


def _mul_all(us: list[Usage]) -> Usage:
    out: Counter[int] = Counter()
    for u in us:
        out.update(u.counts)
    return Usage(out)


def _add_merge(branches: list[Usage], names: dict[int, str]) -> Usage:
    """Additive merge: every branch must consume the tracked vars identically.

    Returns the (shared) usage of a single branch. A disagreement means a linear var is
    used in one summand but not another (or a different number of times) — a linearity
    violation, since ``v₁ + v₂`` must be linear in each tracked argument.
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

    # A Sum is ADDITIVE: its summands are alternative branches that must each consume the tracked
    # vars identically.
    if isinstance(node, Sum):
        branches = [_usage(v, tracked, names) for v in node.vectors]
        return _add_merge(branches, names)

    # A direct sum ``OPlusV`` is ALSO additive — its slots are PARALLEL, independent components of
    # ``⊕``, not a shared multiplicative path.  So ``h ⊕ h`` (the direct-sum diagonal, e.g.
    # ``d(x⊕x)``) consumes ``h`` ONCE, not twice, and ``h ⊕ 0`` (an injection, the derivative of a
    # slot that is x-independent) is fine.  A ``Zero`` slot is the additive unit (no constraint);
    # the non-zero slots must consume identically, so genuine nonlinearity in a slot (``h⊗h``, which
    # is multiplicative inside the slot ⇒ count 2) is still rejected by the outer exactly-once check.
    if isinstance(node, OPlusV):
        branches = [_usage(v, tracked, names) for v in node.vectors if not isinstance(v, Zero)]
        return _add_merge(branches, names)

    # A nested LLam binds its OWN (fresh-uid) vars; the outer linear vars it mentions are
    # captured once by the value it denotes. Recurse into the body multiplicatively.
    if isinstance(node, LLam):  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): binder cases else generic multiplicative recursion [approved 2026-07-03]
        return _usage(node.body, tracked, names)

    # Everything else is MULTIPLICATIVE: every child Term contributes to the same path,
    # so their usages add. (Embedded Spaces never consume a linear value — a tracked var
    # appearing inside a Space annotation is a dependence-position violation already
    # rejected by the core structural guard, so we do not count it here.)
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
    """Enforce the exactly-once linear discipline for a single ``LLam``.

    Each variable bound by ``node`` must be consumed **exactly once** along every path of
    the body. Raises :class:`LinearityError` on under- or over-use.
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
    """Recursively enforce the resource discipline on every ``LLam`` in ``term``.

    This is the inference-layer counterpart of the core's structural guard: it walks the
    whole term and applies :func:`check_llam_resources` at each ``LLam`` it finds.
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
