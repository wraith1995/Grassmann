"""Asserted map-property propagation + FFS dimensions (`25 §7`).

Map ``props`` (on a ``Var``) are **asserted/trusted** in v1: booleans
``inj``/``surj``/``bij`` plus an optional (possibly symbolic) ``rank``/``dim_ker``/
``dim_img``. They (a) set ``rank(f)`` for the FFS dimension rules (`25 §1`) and (b)
license ``Solve`` (``bij``) / sharpen ``Pinv`` (``inj``/``surj``).

This module exposes the property-to-FFS-dimension flow: given a map ``f`` it computes the
dimensions of ``Ker f`` / ``Img f`` / ``CoKer f`` / ``CoImg f`` from its asserted rank
(or the runtime atom ``δ_f`` when rank is unspecified — F3).
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
    """The asserted properties of a map term, if any (props live on ``Var``)."""
    if isinstance(f, Var):
        p = f.props
        if p is not None and not p.is_empty():
            return p
    return None


def is_bijective(f: object) -> bool:
    p = map_props(f)
    if p is not None and p.bij:
        return True
    # The inverse / metric-adjoint of a bijection is itself a bijection — so a re-inverted
    # bijection (e.g. ``Solve(Solve(X))``, which the ``d det``/``d logdet`` rules build when
    # differentiating a determinant of an inverse) type-checks without a fresh ``bij`` assertion.
    if isinstance(f, (Solve, Pinv, Dagger)):
        return is_bijective(f.f)
    return False


def is_injective(f: object) -> bool:
    p = map_props(f)
    return p is not None and p.inj


def is_surjective(f: object) -> bool:
    p = map_props(f)
    return p is not None and p.surj


@dataclass(frozen=True)
class FFSDims:
    """The four FFS dimensions of a map ``f`` (`25 §1`/§7)."""

    rank: Dim
    dim_ker: Dim
    dim_img: Dim
    dim_coker: Dim
    dim_coimg: Dim


def ffs_dims(f: object, dom: Space, cod: Space) -> FFSDims:
    """Compute the four FFS dims for a map ``f : dom ⊸ cod`` (`25 §1`).

    ``rank`` is asserted (static) or the runtime atom ``δ_f`` (F3). Then
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
    """Dimension of a specific FFS space node (dispatch helper)."""
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
