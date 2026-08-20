"""Asserted map-property propagation and the four fundamental-subspace dimensions.

A map's :class:`~grassmann.core.props.MapProps` (carried on a
:class:`~grassmann.core.term.Var`) are **asserted/trusted** facts: booleans
``inj``/``surj``/``bij`` plus an optional (possibly symbolic) ``rank``/``dim_ker``/
``dim_img``. They (a) set ``rank(f)`` for the fundamental-subspace dimension rules and (b)
license :class:`~grassmann.core.term.Solve` (``bij``) and sharpen
:class:`~grassmann.core.term.Pinv` (``inj``/``surj``).

This module exposes the property-to-FFS-dimension flow: given a map ``f`` it computes the
dimensions of its kernel, image, cokernel, and coimage
(:class:`~grassmann.core.space.Ker` / :class:`~grassmann.core.space.Img` /
:class:`~grassmann.core.space.CoKer` / :class:`~grassmann.core.space.CoImg`) from the asserted rank, or from the
runtime rank atom ``δ_f`` when the rank is unspecified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from grassmann.core.props import MapProps
from grassmann.core.space import CoImg, Dim, Img, Ker, Space
from grassmann.core.space import CoKer as _CoKer
from grassmann.core.term import Dagger, Pinv, Solve, Var

from .dims import dim_diff, dim_of, rank_of


def map_props(f: object) -> MapProps | None:
    """Give a map term's asserted properties, if any.

    Asserted properties (:class:`~grassmann.core.props.MapProps`) live on a
    :class:`~grassmann.core.term.Var`; returns ``None`` for any other term or for a
    ``Var`` carrying no (non-empty) assertion.
    """
    if isinstance(f, Var):
        p = f.props
        if p is not None and not p.is_empty():
            return p
    return None


def is_bijective(f: object) -> bool:
    """Report whether ``f`` is asserted bijective (inverses of bijections count)."""
    p = map_props(f)
    if p is not None and p.bij:
        return True
    # The inverse / metric-adjoint of a bijection is itself a bijection — so a re-inverted
    # bijection (e.g. ``Solve(Solve(X))``, which the derivative rules for a determinant of an
    # inverse build) type-checks without a fresh ``bij`` assertion.
    if isinstance(f, (Solve, Pinv, Dagger)):  # audit: ok EX-ISINSTANCE-LADDER — safe direction: refusal-to-license bijectivity [approved 2026-07-03]
        return is_bijective(f.f)
    return False


def is_injective(f: object) -> bool:
    """Report whether ``f`` is asserted injective."""
    p = map_props(f)
    return p is not None and p.inj


def is_surjective(f: object) -> bool:
    """Report whether ``f`` is asserted surjective."""
    p = map_props(f)
    return p is not None and p.surj


@dataclass(frozen=True)
class FFSDims:
    """The four FFS dimensions of a map ``f``."""

    rank: Dim
    dim_ker: Dim
    dim_img: Dim
    dim_coker: Dim
    dim_coimg: Dim


def ffs_dims(f: object, dom: Space, cod: Space) -> FFSDims:
    """Compute the four FFS dims for a map ``f : dom ⊸ cod``.

    ``rank`` is asserted (static) or the runtime atom ``δ_f``. Then
    ``|Img|=|CoImg|=rank``, ``|Ker|=|dom|−rank``, ``|CoKer|=|cod|−rank``.
    """
    r = rank_of(f)
    return FFSDims(
        rank=r,
        dim_ker=dim_diff(dim_of(dom), r),
        dim_img=r,
        dim_coker=dim_diff(dim_of(cod), r),
        dim_coimg=r,
    )


def ffs_dim(node: Ker | Img | _CoKer | CoImg, dom: Space, cod: Space) -> Dim:
    """Give the dimension of one FFS space node (``Img``/``CoImg``/``Ker``/``CoKer``)."""
    dims = ffs_dims(node.f, dom, cod)
    match node:
        case Img():
            return dims.dim_img
        case CoImg():
            return dims.dim_coimg
        case Ker():
            return dims.dim_ker
        case _CoKer():
            return dims.dim_coker
        case _ as unreachable:
            assert_never(unreachable)


__all__ = [
    "map_props",
    "is_bijective",
    "is_injective",
    "is_surjective",
    "FFSDims",
    "ffs_dims",
    "ffs_dim",
]
