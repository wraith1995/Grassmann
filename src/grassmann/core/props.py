"""Asserted properties of a map variable: injectivity, surjectivity, rank.

A linear-map variable may carry properties the user asserts and the language
trusts: the booleans ``inj``, ``surj``, and ``bij``, and a rank, which may be
symbolic and may be given directly as ``dim_ker`` or ``dim_img``. Properties live
on variables only.

They do two things. They fix ``rank(f)``, which the dimension rules for the four
fundamental subspaces need, and they license or sharpen operations: ``bij``
licenses :class:`~grassmann.core.term.Solve`, and ``inj`` or ``surj`` sharpen
:class:`~grassmann.core.term.Pinv`. A rank left unspecified becomes a runtime
atom resolved at lowering rather than inferred statically; inference of
properties across compositions is not done here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .space import Dim, as_dim


class PropError(Exception):
    """Raised when an asserted property set is inconsistent or malformed."""


@dataclass(frozen=True)
class MapProps:
    """A bundle of the properties asserted of a linear-map variable.

    The booleans ``inj``, ``surj``, and ``bij`` record injectivity, surjectivity, and
    bijectivity; ``rank``, ``dim_ker``, and ``dim_img`` record the dimensions, each a
    possibly symbolic :class:`~grassmann.core.space.Dim`. Every field defaults to the
    unasserted state, so an empty bundle claims nothing. Bijectivity implies the other
    two and is normalized to set them at construction, and ``dim_img`` is the rank by
    definition, so giving both requires them to be the same node.
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
    """Assert the image dimension, which is the rank."""
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
    """Build a property bundle from the assertions given, coercing any int dimension."""
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
