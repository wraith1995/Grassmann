"""Capture-avoiding substitution on Spaces (and a Term passthrough), `25 §0`.

``s[α := s']`` replaces a rigid space variable ``SVar(α)`` (or, equivalently, an ``SVar``
named ``α`` standing for a Π-bound space parameter) by a space throughout ``s``. This is
the operation that *closes open spaces on application* (`25 §3` note).

Only the substitutions the inference rules actually need are implemented; nats/ints are
not substituted here (they are concrete per the locked decision).
"""

from __future__ import annotations

from dataclasses import fields, replace

from grassmann.core.space import (
    App,
    CoImg,
    CoKer,
    Dual,
    Img,
    Ker,
    OPlus,
    OPlusRep,
    OProd,
    Space,
    SVar,
    WithInner,
    is_space,
)


def subst_space(s: Space, name: str, repl: Space) -> Space:
    """``s[name := repl]`` over an ``SVar(name)`` rigid space variable.

    Descends into a four-fundamental-subspace defining map ``f`` (a Term) and an ``App``'s
    SpaceFunction / args, so a placeholder threaded *inside* a term — e.g. the tangent ``SVar("T")``
    in ``Λᵏ(T) = Img(alt(T, k))`` (the ``alt`` antisymmetrizer binds vars of space ``T``) — is closed
    too. Safe: the value binders inside ``f`` never shadow a rigid *space* variable.
    """
    match s:
        case SVar():
            return repl if s.name == name else s
        case Dual():
            return Dual(subst_space(s.space, name, repl))
        case WithInner():
            return WithInner(subst_space(s.space, name, repl), s.metric)
        case OPlus():
            return OPlus(tuple(subst_space(c, name, repl) for c in s.spaces))
        case OProd():
            return OProd(tuple(subst_space(c, name, repl) for c in s.spaces))
        case OPlusRep():
            return OPlusRep(s.count, subst_space(s.space, name, repl))
        case Ker() | Img() | CoKer() | CoImg():
            f = _subst_space_in_term(s.f, name, repl)
            src = subst_space(s.source, name, repl) if s.source is not None else None
            tgt = subst_space(s.target, name, repl) if s.target is not None else None
            return type(s)(f, src, tgt)
        case App():
            g = _subst_space_in_term(s.g, name, repl)
            args = tuple(_subst_in_app_arg(a, name, repl) for a in s.args)
            return App(g, s.positions, args)
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): structural recursion; leaves unchanged [approved Teo 2026-07-03]
            # Scalar / Base / SVar(other) / metavars: unchanged.
            return s


def _subst_in_app_arg(a: object, name: str, repl: Space) -> object:
    from grassmann.core.term import is_term

    if is_space(a):
        return subst_space(a, name, repl)
    if is_term(a):
        return _subst_space_in_term(a, name, repl)
    return a  # a Dim — not substituted (nats/ints are concrete)


def _subst_space_in_term(term: object, name: str, repl: Space) -> object:
    """Substitute ``SVar(name) := repl`` through every Space embedded in a Term (the spaces of
    binder vars, asserted spaces, nested FFS maps, …).  Local to :mod:`grassmann.infer` to avoid an
    ``infer → transform`` import cycle; mirrors ``transform.rewrite._subst_space_in_term``."""
    from grassmann.core.term import is_term

    if not is_term(term):
        return term

    def go(node: object) -> object:
        new_fields: dict[str, object] = {}
        changed = False
        for fld in fields(node):
            val = getattr(node, fld.name)
            if isinstance(val, tuple):
                new = tuple(
                    go(c) if is_term(c) else (subst_space(c, name, repl) if is_space(c) else c)
                    for c in val
                )
                if any(a is not b for a, b in zip(new, val, strict=True)):
                    changed = True
                new_fields[fld.name] = new
            elif is_term(val):
                nv = go(val)
                if nv is not val:
                    changed = True
                new_fields[fld.name] = nv
            elif is_space(val):
                nv = subst_space(val, name, repl)
                if nv is not val:
                    changed = True
                new_fields[fld.name] = nv
            else:
                new_fields[fld.name] = val
        return replace(node, **new_fields) if changed else node

    return go(term)


def space_occurs(needle: Space, haystack: Space) -> bool:
    """Does the space ``needle`` occur anywhere in ``haystack`` — descending into FFS defining maps
    and ``App`` like :func:`subst_space`?  The "does a tangent ``Img(J)`` occur in this space?"
    traversal pointwise needs (e.g. to detect whether a value space still mentions the chart)."""
    from grassmann.core.term import is_term

    if not is_space(haystack):
        return False
    if haystack == needle:
        return True

    def in_node(node: object) -> bool:
        for fld in fields(node):
            val = getattr(node, fld.name)
            for c in (val if isinstance(val, tuple) else (val,)):
                if is_space(c) and space_occurs(needle, c):
                    return True
                if is_term(c) and in_node(c):
                    return True
        return False

    from dataclasses import is_dataclass

    return is_dataclass(haystack) and in_node(haystack)


def subst_type_space(t: object, name: str, repl: Space) -> object:
    """``τ[name := repl]`` over a rigid space var ``SVar(name)`` *throughout a Type*.

    Recurses into ``Π`` domains/codomains and ``⟦s⟧`` wrappers so that applying a space
    parameter closes every later occurrence of it in the codomain type (`25 §3` note:
    *substitution closes open spaces on apply*). Capture-avoidance: a nested ``Π`` whose
    own space binder shadows ``name`` stops the substitution under it.
    """
    from grassmann.core.kinds import SPACE, SpaceSort
    from grassmann.core.types import Pi, SpaceAsType, TypeApp

    match t:
        case SpaceAsType():
            return SpaceAsType(subst_space(t.space, name, repl))
        case Pi():
            dom = t.domain
            if isinstance(dom, SpaceSort):
                dom = SpaceSort(subst_space(dom.space, name, repl))  # type: ignore[arg-type]
            # Shadowing: a nested Π binding the same space name hides outer `name`.
            if dom == SPACE and t.binder == name:
                return Pi(t.binder, dom, t.codomain)
            return Pi(t.binder, dom, subst_type_space(t.codomain, name, repl))
        case TypeApp():
            return t
        case _:
            return t


def subst_term(t: object, name: str, repl: object) -> object:
    """Term-level substitution passthrough.

    The §26 examples never substitute a *value* into a synthesized Space (codomains there
    are value-independent), so this is the identity in v1. Kept as an entry point so the
    synth rules read uniformly and future dependent codomains can hook in here.
    """
    return t


__all__ = ["subst_space", "subst_type_space", "subst_term", "space_occurs"]
