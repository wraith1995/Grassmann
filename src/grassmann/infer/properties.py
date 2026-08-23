"""Asserted map properties, and the four fundamental-subspace dimensions they determine.

A map's :class:`~grassmann.core.props.MapProps`, carried on its
:class:`~grassmann.core.term.Var`, are facts the user asserted and inference trusts:
``inj``, ``surj``, ``bij``, and optionally a symbolic ``rank``, ``dim_ker`` or ``dim_img``.
They do three things. They fix ``rank(f)``, which is the one quantity the four fundamental
subspaces' dimensions all depend on; they license :class:`~grassmann.core.term.Solve`,
which is only meaningful for a bijection; and they sharpen
:class:`~grassmann.core.term.Pinv`, whose realization differs for an injective and a
surjective map.

The rank determines the rest by the rank–nullity theorem::

    f : dom ⊸ cod        r = rank(f)   — asserted, else the runtime atom δ_f

        |Img f|   = r            |CoImg f| = r
        |Ker f|   = |dom| − r    |CoKer f| = |cod| − r

With nothing asserted, ``r`` is the runtime atom ``δ_f`` and every one of those dimensions
carries it, so the subspaces size dynamic axes at lowering instead of static ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from grassmann.core.props import MapProps
from grassmann.core.space import CoImg, Dim, Img, Ker, Space
from grassmann.core.space import CoKer as _CoKer
from grassmann.core.term import Assert, Dagger, Pinv, Solve, Var

from .dims import dim_diff, dim_of, rank_of


def map_props(f: object) -> MapProps | None:
    """Give the properties asserted of a map term, or ``None`` when none are.

    Properties live on a :class:`~grassmann.core.term.Var` or on an
    :class:`~grassmann.core.term.Assert`. The assertion node matters because a *derived*
    map can carry properties too: differentiating a determinant asserts its operand ``bij``
    so that the inverse in Jacobi's formula types. An empty property set reads as ``None``.
    """
    if isinstance(f, Var):
        p = f.props
        if p is not None and not p.is_empty():
            return p
    if isinstance(f, Assert):
        p = f.props
        if p is not None and not p.is_empty():
            return p
    return None


def is_bijective(f: object) -> bool:
    """Report whether ``f`` is asserted bijective, following inverses and adjoints.

    The inverse, pseudo-inverse and adjoint of a bijection are bijections, so a re-inverted
    bijection answers ``True`` without a second assertion. That is what lets the derivative
    of a determinant of an inverse, which builds a ``Solve`` of a ``Solve``, type-check.
    """
    p = map_props(f)
    if p is not None and p.bij:
        return True
    # The inverse and the metric adjoint of a bijection are themselves bijections, so a
    # re-inverted bijection — ``Solve(Solve(X))``, which the derivative of a determinant of an
    # inverse builds — needs no fresh ``bij`` assertion.
    if isinstance(f, (Solve, Pinv, Dagger)):  # audit: ok EX-ISINSTANCE-LADDER — safe direction: refusal-to-license bijectivity [approved 2026-07-03]
        return is_bijective(f.f)
    return False


def is_injective(f: object) -> bool:
    """Report whether ``f`` is asserted injective — the condition ``Corestrict`` requires."""
    p = map_props(f)
    return p is not None and p.inj


def is_surjective(f: object) -> bool:
    """Report whether ``f`` is asserted surjective, so its pseudo-inverse is a right inverse."""
    p = map_props(f)
    return p is not None and p.surj


@dataclass(frozen=True)
class FFSDims:
    """The rank of a map together with the dimensions of its four fundamental subspaces.

    Every field is a symbolic ``Dim``, and every one is derived from ``rank``, so all five
    mention the runtime atom ``δ_f`` together when no rank was asserted.
    """

    rank: Dim
    dim_ker: Dim
    dim_img: Dim
    dim_coker: Dim
    dim_coimg: Dim


def ffs_dims(f: object, dom: Space, cod: Space) -> FFSDims:
    """Compute the rank and the four fundamental-subspace dimensions of ``f : dom ⊸ cod``.

    :func:`~grassmann.infer.dims.rank_of` supplies the rank, static or the runtime atom
    ``δ_f``, and rank–nullity supplies the rest: the image and coimage have the rank as
    their dimension, and the kernel and cokernel take what is left of the domain and the
    codomain.
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
    """Give the dimension of a single fundamental-subspace node.

    Computes all four via :func:`ffs_dims` and selects the one the node names.
    """
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
