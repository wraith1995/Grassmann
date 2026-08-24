"""Carrying a value or a basis along a map, into the map's own image space.

Transport applies a map. Given ``f : dom → cod`` and a value in ``dom(f)``, it pushes the
value through ``f`` and corestricts the result to ``Img(f)``, so the answer lives in the
image as a space in its own right rather than somewhere in the ambient codomain. The four
entry points are that operation and its variants::

    pushforward(v, f)        dom(f) ⤳ Img(f)    Project(Eval(f, ·, v), Img(f))
    pull_along(v, f)         cod(f) ⤳ dom(f)    Eval(Pinv(f), ·, v)
    transport_basis(b, f)    a basis of dom(f) carried to Img(f)
                                                ImageBasis(f, source=b)
    transport_cobasis(b, f)  the dual side, f^{-⋆}
                                                DualBasis(ImageBasis(f, source=b))

This is a different operation from :func:`~grassmann.transform.pullback.pullback`, which
moves a value between spaces that are already related structurally — a summand of a direct
sum, a factor of a tensor product, a subspace of its ambient — and never applies a map.
The distinction matters because it is invisible to dimension: ``dom(f)`` and ``Img(f)``
have the same dimension but no structural relation, so asking ``pullback`` for the
coordinate convert ``Rᵗ → Img(J)`` would get a dimension-equal relabel where applying ``f``
was meant.

Everything here is built from ``Eval``, ``Project``, ``Img``, ``ImageBasis`` and ``Pinv``,
and from nothing that orthonormalizes: no :class:`~grassmann.core.basis.ProjectedBasis` or
:class:`~grassmann.core.basis.OrthonormalBasis`, and no
:class:`~grassmann.core.term.Sqrt`. That is what keeps a transport along an affine or
injective ``f`` a **rational** function of ``f``, since orthonormalization is where a
``√det`` would enter. The metric rides along through ``WithInner``/``Inner`` over ``Img(f)``
and is never materialized. :func:`assert_rational` checks the property directly, which is
what separates this transport from a per-cell L²-orthonormalization.

The transported reference basis is :func:`transport_basis` of the reference basis along the
chart immersion; it is realized numerically only at the lowering boundary.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator

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
    """Raised when a transport cannot be built, or when one would introduce an irrational atom.

    The second case is what :func:`assert_rational` reports: an orthonormalizing basis node
    or a ``Sqrt`` on the transported object means it is no longer a rational function of the
    map.
    """


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
    """Carry the value ``v : dom(f)`` along ``f`` and land it in ``Img(f)``.

    The construction applies ``f`` and then corestricts, giving
    ``Project(Eval(f, (0,), (v,)), Img(f))``, so the result is expressed in the image's own
    coordinates rather than the ambient codomain's. ``target`` pins the codomain endpoint on
    the ``Img`` node when the caller knows it.
    """
    _require_map(f, "pushforward")
    if not is_term(v):
        raise TransportError(f"pushforward: expected a Term value, got {v!r}")
    applied = Eval(f, (0,), (v,))
    return Project(applied, Img(f, target=target))


def pull_along(v: Term, f: Term) -> Term:
    """Carry the value ``v : cod(f)`` back to ``dom(f)`` through the pseudo-inverse of ``f``.

    The construction is ``Eval(Pinv(f), (0,), (v,))``. For an injective ``f`` the
    pseudo-inverse is the exact left inverse, ``f⁺f = id``, so this genuinely undoes
    :func:`pushforward`.
    """
    _require_map(f, "pull_along")
    if not is_term(v):
        raise TransportError(f"pull_along: expected a Term value, got {v!r}")
    return Eval(Pinv(f), (0,), (v,))


def transport_basis(
    b: Basis | None, f: Term, *, target: Basis | None = None
) -> ImageBasis:
    """Carry the basis ``b`` of ``dom(f)`` along ``f`` to coordinatize ``Img(f)``.

    The result is an ``ImageBasis``: the literal pushforward columns ``f(b₁), …, f(bₙ)``,
    left as they are rather than orthonormalized, which is what keeps the transport
    rational. Passing ``b=None`` transports the standard basis of ``dom(f)``. This is the
    fixed reference basis carried into a chart.
    """
    _require_map(f, "transport_basis")
    if b is not None and not is_basis(b):
        raise TransportError(f"transport_basis: source must be a Basis, got {b!r}")
    return ImageBasis(f, source=b, target=target)


def transport_cobasis(
    b: Basis | None, f: Term, *, target: Basis | None = None
) -> DualBasis:
    """Carry the cobasis along ``f`` — the dual side of :func:`transport_basis`.

    The result is the dual of the transported reference basis, which is how a covector or a
    degree-of-freedom functional moves: by ``f^{-⋆}``, contravariantly to the vectors.
    """
    return DualBasis(transport_basis(b, f, target=target))


# ----------------------------------------------------------------------
# The rationality guard — what distinguishes transport from L²-orthonormalization.
# ----------------------------------------------------------------------

_IRRATIONAL_NODES = (Sqrt, ProjectedBasis, OrthonormalBasis)


def _walk(obj: object, seen: set[int]) -> Iterator[object]:
    """Yield ``obj`` and every Term, Basis or Space nested in its dataclass fields.

    Nodes already visited are skipped, so a shared subterm is yielded once.
    """
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


def assert_rational(obj: object, *, ctx: str = "transport") -> None:
    """Check that a transported term or basis stays a rational function of the map.

    The scan looks for the only nodes that can introduce a ``√det`` or measure factor: an
    orthonormalizing basis (:class:`~grassmann.core.basis.ProjectedBasis` or
    :class:`~grassmann.core.basis.OrthonormalBasis`) and
    :class:`~grassmann.core.term.Sqrt`. Finding one raises :class:`TransportError`. A clean
    pass is the property a symbolic Vandermonde construction depends on.
    """
    for node in _walk(obj, set()):
        if isinstance(node, _IRRATIONAL_NODES):
            raise TransportError(
                f"{ctx}: transported object contains an irrational atom "
                f"{type(node).__name__!r} — transport must stay rational "
                f"(no orthonormalization / √det)."
            )
