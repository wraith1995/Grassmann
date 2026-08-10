"""Asserted map properties — replaces ``IsoSpace``.

A linear-map var may carry **user-asserted, trusted** properties:

- the booleans ``inj`` / ``surj`` / ``bij`` (v1), and/or
- a (possibly symbolic) ``rank`` — or directly ``dim_ker`` / ``dim_img``.

These are recorded on ``Var`` (properties live ON VARS ONLY in v1). They (a) set
``rank(f)`` for the FFS dimension rules and (b) license ``Solve`` (``bij``) / sharpen
``Pinv`` (``inj``/``surj``). If a rank is left unspecified it becomes a runtime atom
``δ_f`` at lowering — **no static rank-inference** in v1 (F3). Property *inference*
(``inj∘inj``, ``embed ⇒ inj``, …) is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

from .space import Dim, as_dim


class PropError(Exception):
    """Raised when an asserted property set is inconsistent or malformed."""


@dataclass(frozen=True)
class MapProps:
    """A bundle of asserted properties for a linear-map variable.

    Boolean flags ``inj``/``surj``/``bij`` plus optional symbolic rank/dim fields.
    All fields default to "unasserted". ``bij`` implies ``inj`` and ``surj`` and is
    normalized as such at construction.
    """

    inj: bool = False
    surj: bool = False
    bij: bool = False
    rank: Dim | None = None
    dim_ker: Dim | None = None
    dim_img: Dim | None = None

    def __post_init__(self) -> None:
        # Normalize: bij ⇒ inj ∧ surj.
        inj = self.inj or self.bij
        surj = self.surj or self.bij
        object.__setattr__(self, "inj", inj)
        object.__setattr__(self, "surj", surj)
        # Coerce symbolic-dim fields.
        if self.rank is not None:
            object.__setattr__(self, "rank", as_dim(self.rank))
        if self.dim_ker is not None:
            object.__setattr__(self, "dim_ker", as_dim(self.dim_ker))
        if self.dim_img is not None:
            object.__setattr__(self, "dim_img", as_dim(self.dim_img))
        # dim_img is rank by definition; if both given they must be the *same* node.
        if (
            self.rank is not None
            and self.dim_img is not None
            and self.rank != self.dim_img
        ):
            raise PropError(
                f"rank ({self.rank!r}) and dim_img ({self.dim_img!r}) disagree"
            )

    def is_empty(self) -> bool:
        """Report whether no property at all is asserted."""
        return not (
            self.inj
            or self.surj
            or self.bij
            or self.rank is not None
            or self.dim_ker is not None
            or self.dim_img is not None
        )

    def __repr__(self) -> str:
        parts = []
        if self.bij:
            parts.append("bij=True")
        else:
            if self.inj:
                parts.append("inj=True")
            if self.surj:
                parts.append("surj=True")
        if self.rank is not None:
            parts.append(f"rank={self.rank!r}")
        if self.dim_ker is not None:
            parts.append(f"dim_ker={self.dim_ker!r}")
        if self.dim_img is not None:
            parts.append(f"dim_img={self.dim_img!r}")
        return f"MapProps({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Smart constructors / convenience singletons (mirror the builder `props=` API).
# ---------------------------------------------------------------------------

inj = MapProps(inj=True)
surj = MapProps(surj=True)
bij = MapProps(bij=True)


def rank(r: object) -> MapProps:
    """Assert a (possibly symbolic) rank."""
    return MapProps(rank=as_dim(r))


def dim_ker(k: object) -> MapProps:
    """Assert the kernel dimension."""
    return MapProps(dim_ker=as_dim(k))


def dim_img(r: object) -> MapProps:
    """Assert the image dimension (= rank)."""
    return MapProps(dim_img=as_dim(r))


def props(
    *,
    inj: bool = False,
    surj: bool = False,
    bij: bool = False,
    rank: object | None = None,
    dim_ker: object | None = None,
    dim_img: object | None = None,
) -> MapProps:
    """General constructor for a property bundle."""
    return MapProps(
        inj=inj,
        surj=surj,
        bij=bij,
        rank=None if rank is None else as_dim(rank),
        dim_ker=None if dim_ker is None else as_dim(dim_ker),
        dim_img=None if dim_img is None else as_dim(dim_img),
    )


__all__ = [
    "PropError",
    "MapProps",
    "inj",
    "surj",
    "bij",
    "rank",
    "dim_ker",
    "dim_img",
    "props",
]
