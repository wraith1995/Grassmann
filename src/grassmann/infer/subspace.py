"""Subspace relation ``⊑`` (`25 §6`) — for project / embed.

```
s ⊑ ⊕(…, s, …)          (a summand)
Ker f ⊑ dom f   Img f ⊑ cod f   CoImg f ⊑ dom f   CoKer f ⊑ cod f
reflexive + transitive
```

Coordinate realization happens at lowering (``CompleteBasis``); here we only decide the
relation structurally (with dim-equality for the reflexive/leaf case).
"""

from __future__ import annotations

from grassmann.core.space import (
    CoImg,
    CoKer,
    DimLit,
    Dual,
    Img,
    Ker,
    OPlus,
    OPlusRep,
    OProd,
    Space,
)


def _norm(s: Space) -> Space:
    """Canonicalize a space for structural comparison: expand static ``⊕ⁿ s`` to ``⊕(s,…,s)``
    (recursively), so ``ℝⁿ`` compares equal whether written as ``OPlusRep`` or ``OPlus``."""
    match s:
        case OPlusRep(count, inner) if isinstance(count, DimLit):
            return OPlus(tuple([_norm(inner)] * count.value))
        case OPlus(spaces):
            return OPlus(tuple(_norm(x) for x in spaces))
        case OProd(spaces):
            return OProd(tuple(_norm(x) for x in spaces))
        case Dual(inner):
            return Dual(_norm(inner))
        case _:
            return s


def _eq(a: Space, b: Space) -> bool:
    # Compare up to the ``⊕ⁿ ≡ ⊕(…)`` canonical form (the ℝⁿ ground can appear either way).
    return a == b or _norm(a) == _norm(b)


def is_subspace(sub: Space, amb: Space) -> bool:
    """Decide ``sub ⊑ amb`` (`25 §6`), reflexive + transitive."""
    if _eq(sub, amb):
        return True

    # FFS ⊑ ambient.
    if isinstance(amb, OPlus):
        # A summand (recursively/transitively) is a subspace.
        for comp in amb.spaces:
            if is_subspace(sub, comp):
                return True

    # FFS of a map sits inside its dom/cod.
    if isinstance(sub, (Ker, CoImg)):
        dom = _ffs_dom(sub)
        if dom is not None and is_subspace_or_eq(dom, amb):
            return True
    if isinstance(sub, (Img, CoKer)):
        cod = _ffs_cod(sub)
        if cod is not None and is_subspace_or_eq(cod, amb):
            return True

    return False


def is_subspace_or_eq(a: Space, b: Space) -> bool:
    return _eq(a, b) or is_subspace(a, b)


def _ffs_dom(node: Ker | CoImg) -> Space | None:
    if node.source is not None:
        return node.source
    from .synth import map_domain_codomain

    try:
        dom, _ = map_domain_codomain(node.f)
        return dom
    except Exception:
        return None


def _ffs_cod(node: Img | CoKer) -> Space | None:
    if node.target is not None:
        return node.target
    from .synth import map_domain_codomain

    try:
        _, cod = map_domain_codomain(node.f)
        return cod
    except Exception:
        return None


__all__ = ["is_subspace", "is_subspace_or_eq"]
