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

from typing import Literal, TypeGuard, assert_never

from grassmann.core.space import (
    App,
    CoImg,
    CoKer,
    DimLit,
    Dual,
    Img,
    Ker,
    OPlus,
    OPlusRep,
    OProd,
    Scalar,
    Space,
    SpaceMeta,
    SVar,
    WithInner,
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
            # Drop the ⊗-unit (Scalar) and collapse a single-factor product to its factor:
            # ``X ⊗ ℝ ≅ X`` and ``OProd((X,)) ≅ X`` — so a k=1 exterior power ``Λ¹(V) = Img(alt)``,
            # whose codomain is the 1-tuple ``V⊗¹ = OProd((V,))``, is recognized as ``⊑ V``.
            ns = [_norm(x) for x in spaces if not isinstance(x, Scalar)]
            if not ns:
                return Scalar()
            return ns[0] if len(ns) == 1 else OProd(tuple(ns))
        case Dual(inner):
            return Dual(_norm(inner))
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): canonicalize specific else passthrough [approved Teo 2026-07-03]
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
    if isinstance(sub, (Img, CoKer)):  # audit: ok EX-ISINSTANCE-LADDER — safe direction: False only blocks project/embed [approved Teo 2026-07-03]
        cod = _ffs_cod(sub)
        if cod is not None and is_subspace_or_eq(cod, amb):
            return True

    return False


def is_subspace_or_eq(a: Space, b: Space) -> bool:
    return _eq(a, b) or is_subspace(a, b)


def _ffs_dom(node: Ker | CoImg) -> Space | None:
    if node.source is not None:
        return node.source
    from .synth import InferError, map_domain_codomain

    try:
        dom, _ = map_domain_codomain(node.f)
        return dom
    except InferError:
        return None


def _ffs_cod(node: Img | CoKer) -> Space | None:
    if node.target is not None:
        return node.target
    from .synth import InferError, map_domain_codomain

    try:
        _, cod = map_domain_codomain(node.f)
        return cod
    except InferError:
        return None


# ---------------------------------------------------------------------------
# Canonical FFS / map-space probes (2026-07-03 consolidation — fem audit #18).
# These replace the per-module hand-rolled copies (pullback._is_ffs/_ffs_superspace,
# diff._subspace_defining_map/_ffs_kind/_map_dom_cod, choose._operator_of,
# embed.ops._split_map_space, _dims' peel, represent's membership gates).
# ---------------------------------------------------------------------------

FfsKind = Literal["img", "ker", "coimg", "coker"]


def is_ffs(s: Space) -> TypeGuard[Img | Ker | CoImg | CoKer]:
    """Whether ``s`` is one of the four fundamental subspaces of a map."""
    return isinstance(s, (Img, Ker, CoImg, CoKer))  # audit: ok EX-ISINSTANCE-LADDER — THE canonical FFS membership predicate (definitional 4-member subset) [approved Teo 2026-07-03]


def ffs_defining_map(s: Space) -> object | None:
    """The defining map ``f`` (a Term) of ``Img(f)/Ker(f)/CoImg(f)/CoKer(f)``, else ``None``."""
    match s:
        case Img(f, _, _) | Ker(f, _, _) | CoImg(f, _, _) | CoKer(f, _, _):
            return f
        case Scalar() | SVar() | Dual() | OPlus() | OProd() | OPlusRep() | WithInner() | App() | SpaceMeta():
            return None
        case _ as unreachable:
            assert_never(unreachable)


def ffs_kind(s: Space) -> FfsKind | None:
    """The FFS kind of ``s``, else ``None``."""
    match s:
        case Img():
            return "img"
        case Ker():
            return "ker"
        case CoImg():
            return "coimg"
        case CoKer():
            return "coker"
        case Scalar() | SVar() | Dual() | OPlus() | OProd() | OPlusRep() | WithInner() | App() | SpaceMeta():
            return None
        case _ as unreachable:
            assert_never(unreachable)


def ffs_superspace(s: Space) -> Space | None:
    """The ambient an FFS embeds into (``Ker``/``CoImg`` ⊑ dom f; ``Img``/``CoKer`` ⊑ cod f),
    from the explicit endpoint or decoded from ``f`` — ``None`` if ``s`` is not an FFS or the
    map space cannot be decoded."""
    match s:
        case Ker() | CoImg():
            return _ffs_dom(s)
        case Img() | CoKer():
            return _ffs_cod(s)
        case Scalar() | SVar() | Dual() | OPlus() | OProd() | OPlusRep() | WithInner() | App() | SpaceMeta():
            return None
        case _ as unreachable:
            assert_never(unreachable)


def split_map_space(s: Space) -> tuple[Space, Space] | None:
    """Strict one-step peel of the linear-map encoding: ``OProd(Dual(dom), cod)`` → ``(dom, cod)``,
    else ``None``. (The general multi-factor decoder is :func:`grassmann.infer.synth._decode_map_space`
    via :func:`map_domain_codomain`; this is the syntactic 2-factor case, no synthesis.)"""
    if isinstance(s, OProd) and len(s.spaces) == 2 and isinstance(s.spaces[0], Dual):  # audit: ok EX-ISINSTANCE-LADDER — canonical syntactic peel: OProd(Dual, ·) or None by contract [approved Teo 2026-07-03]
        return s.spaces[0].space, s.spaces[1]
    return None


__all__ = [
    "is_subspace",
    "is_subspace_or_eq",
    "is_ffs",
    "ffs_defining_map",
    "ffs_kind",
    "ffs_superspace",
    "split_map_space",
    "FfsKind",
]

