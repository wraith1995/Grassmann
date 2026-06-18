"""Capture-avoiding substitution on Spaces (and a Term passthrough), `25 §0`.

``s[α := s']`` replaces a rigid space variable ``SVar(α)`` (or, equivalently, an ``SVar``
named ``α`` standing for a Π-bound space parameter) by a space throughout ``s``. This is
the operation that *closes open spaces on application* (`25 §3` note).

Only the substitutions the inference rules actually need are implemented; nats/ints are
not substituted here (they are concrete per the locked decision).
"""

from __future__ import annotations

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
)


def subst_space(s: Space, name: str, repl: Space) -> Space:
    """``s[name := repl]`` over an ``SVar(name)`` rigid space variable."""
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
            src = subst_space(s.source, name, repl) if s.source is not None else None
            tgt = subst_space(s.target, name, repl) if s.target is not None else None
            return type(s)(s.f, src, tgt)
        case App():
            return s
        case _:
            # Scalar / Base / SVar(other) / metavars: unchanged.
            return s


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


__all__ = ["subst_space", "subst_type_space", "subst_term"]
