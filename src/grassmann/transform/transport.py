"""Transport along a map — pushforward, the map-driven dual of :mod:`pullback`.

``pullback(v, target)`` (see :mod:`grassmann.transform.pullback`) transports a value
into a *target space* using only **static structural / subspace** relations — it never
applies a map.  Carrying a value or basis **along a chart map** ``f : dom → cod`` is a
*different* operation: it must **apply ``f``** and corestrict to its image ``Img(f)``.
This is the "coordinate convert" ``Rᵗ → Img(J)`` that ``pullback`` must **not** be used
for: ``dom(f)`` and ``Img(f)`` are dim-equal but structurally unrelated, so ``pullback``
would silently fall to a dim-equal ``Assert`` *relabel* (the §45 footgun) instead of
applying ``f``.

This module names that missing primitive — the **restriction of a map to produce values
in its image space** (domain → image), as opposed to the within-ambient inclusion /
restriction that ``embed`` / ``project`` model:

============================  ===========================================================
``pushforward(v, f)``         value  ``v : dom(f) ⤳ Img(f)``  ==  ``Project(Eval(f,·,v), Img(f))``
``pull_along(v, f)``          inverse  ``cod ⤳ dom``  via ``f†``  ==  ``Eval(Pinv(f),·,v)``
``transport_basis(b, f)``     the fixed reference basis ``b`` of ``dom(f)`` carried to
                              ``Img(f)``  ==  ``ImageBasis(f, source=b)``  (raw pushforward)
``transport_cobasis(b, f)``   the ``T^{-⋆}`` dual / functional side  ==
                              ``DualBasis(ImageBasis(f, source=b))``
============================  ===========================================================

Rationality.  Everything here is assembled from ``Eval`` / ``Project`` / ``Img`` /
``ImageBasis`` / ``Pinv`` only — never an orthonormalizing node
(:class:`~grassmann.core.basis.ProjectedBasis` / :class:`~grassmann.core.basis.OrthonormalBasis`)
and never :class:`~grassmann.core.term.Sqrt`.  So for an affine / injective ``f`` the
transport stays a **rational** function of ``f``.  The metric rides free via
``WithInner`` / ``Inner`` over ``Img(f)`` (identity-in-frame) and is never materialized
here.  :func:`assert_rational` is the litmus that distinguishes this (rational) transport
from per-cell L²-orthonormalization (irrational).

This is the grassmann-algebra frame for chartLib's ``sample(..., mode="transport")``: the
transported reference basis sampled there is exactly ``transport_basis`` of the reference
basis along the chart immersion, and chartLib is its numeric / symbolic realization.
"""
from __future__ import annotations

import dataclasses

from grassmann.core.basis import (
    Basis,
    DualBasis,
    ImageBasis,
    OrthonormalBasis,
    ProjectedBasis,
    is_basis,
)
from grassmann.core.space import Img, Space, is_space
from grassmann.core.term import (
    Eval,
    Pinv,
    Project,
    Sqrt,
    Term,
    is_term,
)


class TransportError(Exception):
    """Raised when a transport-along-a-map cannot be built (or would be irrational)."""


def _require_map(f: Term, ctx: str) -> None:
    if is_space(f):
        raise TransportError(
            f"{ctx}: expected a linear MAP term, got a Space {f!r}. "
            f"Transport is map-driven (domain → image); to move a value "
            f"between related spaces by structure use pullback() instead."
        )
    if not is_term(f):
        raise TransportError(f"{ctx}: expected a Term map, got {f!r}")


def pushforward(v: Term, f: Term, *, target: Space | None = None) -> Term:
    """Carry the value ``v : dom(f)`` along the map ``f`` into ``Img(f)``.

    Returns ``Project(Eval(f, (0,), (v,)), Img(f))`` — apply ``f`` then corestrict to its
    image.  ``target`` optionally pins the codomain endpoint on the ``Img`` node.
    """
    _require_map(f, "pushforward")
    if not is_term(v):
        raise TransportError(f"pushforward: expected a Term value, got {v!r}")
    applied = Eval(f, (0,), (v,))
    return Project(applied, Img(f, target=target))


def pull_along(v: Term, f: Term) -> Term:
    """Carry the value ``v : cod(f)`` back along ``f`` to ``dom(f)`` via the pseudo-inverse.

    Returns ``Eval(Pinv(f), (0,), (v,))`` — for an injective ``f`` this is the exact
    left inverse (``f† f = id``), the ``Rᵗ`` side of the coordinate convert.
    """
    _require_map(f, "pull_along")
    if not is_term(v):
        raise TransportError(f"pull_along: expected a Term value, got {v!r}")
    return Eval(Pinv(f), (0,), (v,))


def transport_basis(
    b: Basis | None, f: Term, *, target: Basis | None = None
) -> ImageBasis:
    """The reference basis ``b`` of ``dom(f)`` transported along ``f`` into ``Img(f)``.

    Returns ``ImageBasis(f, source=b, target=target)`` — the **literal pushforward**
    columns ``f(bᵢ)``, *not* orthonormalized, so it stays rational.  ``b=None`` uses the
    standard basis of ``dom(f)``.  This is the grassmann-native object for "the fixed
    reference basis carried into the chart" (the proposal's transported basis).
    """
    _require_map(f, "transport_basis")
    if b is not None and not is_basis(b):
        raise TransportError(f"transport_basis: source must be a Basis, got {b!r}")
    return ImageBasis(f, source=b, target=target)


def transport_cobasis(
    b: Basis | None, f: Term, *, target: Basis | None = None
) -> DualBasis:
    """The ``T^{-⋆}`` dual / functional side of :func:`transport_basis`.

    Returns ``DualBasis(ImageBasis(f, source=b, target=target))`` — the dual of the
    transported reference basis, i.e. the covector / DOF-functional pullback ``f^{-⋆}``.
    """
    return DualBasis(transport_basis(b, f, target=target))


# ----------------------------------------------------------------------
# Rationality guard — the litmus that transport ≠ L²-orthonormalization.
# ----------------------------------------------------------------------

_IRRATIONAL_NODES = (Sqrt, ProjectedBasis, OrthonormalBasis)


def _walk(obj, seen: set) -> object:
    """Yield ``obj`` and every nested Term / Basis / Space dataclass field."""
    if id(obj) in seen:
        return
    if isinstance(obj, (Term, Basis, Space)) or dataclasses.is_dataclass(obj):
        seen.add(id(obj))
        yield obj
        if dataclasses.is_dataclass(obj):
            for fld in dataclasses.fields(obj):
                yield from _walk(getattr(obj, fld.name), seen)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _walk(item, seen)


def assert_rational(obj, *, ctx: str = "transport") -> None:
    """Assert a transported term/basis introduces no irrational atom (the litmus test).

    Scans for an orthonormalizing basis node
    (:class:`~grassmann.core.basis.ProjectedBasis` /
    :class:`~grassmann.core.basis.OrthonormalBasis`) or a
    :class:`~grassmann.core.term.Sqrt` — the only ways a ``√det`` / measure factor can
    enter.  Raises :class:`TransportError` if one is found.  A clean pass means the object
    stays a rational function of the map (the property the symbolic Vandermonde needs).
    """
    for node in _walk(obj, set()):
        if isinstance(node, _IRRATIONAL_NODES):
            raise TransportError(
                f"{ctx}: transported object contains an irrational atom "
                f"{type(node).__name__!r} — transport must stay rational "
                f"(no orthonormalization / √det)."
            )
