"""Substitution of a space for a rigid space variable — what closes an open space.

``s[α := s']`` replaces every occurrence of the rigid space variable
:class:`~grassmann.core.space.SVar` ``SVar(α)`` throughout ``s``. An ``SVar`` is how a
Π-bound space parameter appears inside the spaces under the binder, so applying a space
argument means substituting for it, and that application is what turns an open space into
a closed one.

The traversal reaches everywhere a space can hide, including inside terms: the defining
map of a fundamental subspace, an ``App``'s function and arguments, and the space
annotations on binder variables. That matters because a placeholder often sits inside a
term rather than beside it — the tangent ``SVar("T")`` in ``Λᵏ(T) = Img(alt(T, k))`` lives
on the binder vars of the antisymmetrizer's ``LLam``, and substituting only the outer
space would leave it open. Value binders never shadow a rigid space variable, so the
descent needs no renaming.

Dimensions are not substituted: nats are always concrete, so a nat parameter is supplied
explicitly rather than solved for. :func:`subst_term` is the value-level counterpart and
is the identity, since a synthesized space never depends on a value.
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
    """Give ``s[name := repl]``, replacing the rigid space variable ``SVar(name)`` throughout.

    The recursion is structural over the Space sum and additionally enters the terms a
    space embeds — a fundamental subspace's defining map and an ``App``'s function and
    arguments — so a placeholder threaded inside a term is closed along with the rest.
    Leaves that cannot mention the variable come back unchanged.
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
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): structural recursion; leaves unchanged [approved 2026-07-03]
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
    """Substitute ``SVar(name) := repl`` through every Space a Term embeds.

    That covers binder variables' spaces, asserted spaces, nested fundamental-subspace maps
    and anything else in a Space-typed field. It lives in :mod:`grassmann.infer` to avoid an
    ``infer → transform`` import cycle, and mirrors ``transform.rewrite._subst_space_in_term``.
    """
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
    """Report whether the space ``needle`` occurs anywhere inside ``haystack``.

    The descent matches :func:`subst_space`, reaching into fundamental-subspace defining
    maps and ``App``. It answers questions of the form "does a tangent ``Img(J)`` still
    occur here?" — whether a value's space continues to mention the chart it came from.
    """
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
    """Give ``τ[name := repl]``, replacing the rigid space variable ``SVar(name)`` in a Type.

    The recursion enters ``Π`` domains and codomains and ``⟦s⟧`` wrappers, so applying a
    space parameter closes every later occurrence of it in the codomain. A nested ``Π``
    whose own space binder carries the same name shadows the outer one, and the
    substitution stops under it.
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
    """Substitute a value for a term variable inside a classifier — the identity.

    Synthesized codomains do not depend on the values flowing through them, so there is
    nothing for a value substitution to change. The entry point exists so the synthesis
    rules can call it uniformly wherever the typing judgement calls for it.
    """
    return t


__all__ = ["subst_space", "subst_type_space", "subst_term", "space_occurs"]
