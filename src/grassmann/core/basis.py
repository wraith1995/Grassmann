"""Basis tower (`20 §7`) — first-class, mirrors Space / Type / the Function layer.

```
Basis  b ::=
  -- mirror Space (§5)
  | ScalarBasis | BaseBasis(name, vectors?, space?, backup?)   -- vectors = numeric matrix OR
  |                                  Term frame; space ⇒ self-sizing; backup ⇒ RREF-complete
  | BVar(β)                          basis variable (the basis of an SVar α)
  | DualBasis(b) | PrimalBasis(b)
  | BasisSum(b…) | BasisProd(b…)     -- ⊕ / ⊗  (n-ary)
  | BasisRep(n, b)                   -- ⊕ⁿ
  | ProjectedBasis(f, which, source?, target?)  -- subspace basis (Image/Ker/CoImg/CoKer)
  | ImageBasis(f)                    -- its OWN construct: the literal map-application basis
  | ListBasis(vectors…)              -- explicit list of Term basis vectors
  | WithInnerBasis(b, g)             -- g-orthonormal basis
  | CompleteBasis(subBasis, complement)  -- complete a subspace basis to its ambient
  -- mirror Type (§4) AND the Space/Type-Function layer
  | BLam(b:σ̂. bodyBasis)            value → Frame; basis → basis-function; nat → builder
  | BApp(f, positions, args…)        apply (mirrors App/Eval)
```

``BLam``/``BApp`` mirror ``Lam``/``App`` across the whole tower. A ``BLam`` over a
**value** is the **Frame** (the paper's frame; basis varies with a runtime value) — note
:class:`grassmann.core.term.Frame` is the dedicated Term-level frame node; here ``BLam``
with a ``SpaceSort`` domain is the basis-tower spelling. A ``BLam`` over a **basis** is a
**basis function** (mirror of a Space/Type Function); over a **nat**, a nat-parametric
builder.

No stored ``coeffs`` (that was impl change-of-basis caching; recompute at lowering).
``which`` for :class:`ProjectedBasis` is one of the FFS selectors below.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kinds import Kind, is_kind
from .space import Dim, Space, _tuple_repr, as_dim
from .term import Term, Var


class BasisError(Exception):
    """Raised when a Basis is structurally malformed."""


# Valid `which` selectors for ProjectedBasis (the projection-built FFS).
#
# ``"image"`` is the **Q / orthonormal range** (the immersed side): the ``Q`` of ``QrOp``
# when ``f`` is asserted injective (static rank ``=|dom|``, recovering the old ``IsoSpace``
# path), else ``U[:, :δ_f]`` of ``SvdOp`` in general (`73 Topic 2`, `50 §A1`).  It is the
# **orthonormalized** counterpart of :class:`ImageBasis` (the raw map-application basis);
# the two are related by ``QrOp.R`` / ``QrOp.Q`` (injective) or the SVD factors (general).
PROJECTED_WHICH = ("image", "ker", "coimg", "coker")


@dataclass(frozen=True)
class ScalarBasis:
    """The standard basis of ``Scalar`` (ℝ)."""

    def __repr__(self) -> str:
        return "ScalarBasis()"


@dataclass(frozen=True)
class BaseBasis:
    """A concrete basis of a ``Base`` space — the term-level "term basis" (`20 §7`).

    Fields (all but ``name`` optional, so the historical ``BaseBasis(name)`` /
    ``BaseBasis(name, vectors)`` forms are unchanged):

    * ``vectors`` — the basis vectors.  Either a precomputed numeric matrix (columns = basis
      vectors in ambient coords) OR a tuple of :class:`~grassmann.core.term.Term` vectors (a
      *frame* realized at lowering, like :class:`ListBasis`).  ``None`` ⇒ the standard basis
      (the ambient identity).
    * ``space`` — the :class:`~grassmann.core.space.Space` this basis is OF.  Lets the basis
      recover its dimension/ambient even when ``vectors`` is ``None`` (so a standard basis is
      self-sizing instead of relying on the surrounding space).
    * ``backup`` — a fallback :class:`Basis` used to **complete** ``vectors`` to a full basis
      when the frame is not a maximal independent set (rank-deficient).  At lowering the
      independent columns of ``vectors`` are kept (RREF / pivoted selection) and the missing
      directions are filled from ``backup`` (default: the standard/ambient basis).
    """

    name: str
    vectors: object | None = None
    space: object | None = None
    backup: object | None = None

    def __repr__(self) -> str:
        v = f", vectors={self.vectors!r}" if self.vectors is not None else ""
        s = f", space={self.space!r}" if self.space is not None else ""
        bk = f", backup={self.backup!r}" if self.backup is not None else ""
        return f"BaseBasis({self.name!r}{v}{s}{bk})"


@dataclass(frozen=True)
class BVar:
    """A basis variable ``β`` — the basis of an ``SVar α`` (rides on a ``Lam``)."""

    name: str

    def __repr__(self) -> str:
        return f"BVar({self.name!r})"


@dataclass(frozen=True)
class DualBasis:
    """``DualBasis(b)`` — the dual basis."""

    basis: Basis

    def __post_init__(self) -> None:
        _require_basis(self.basis, "DualBasis")

    def __repr__(self) -> str:
        return f"DualBasis({self.basis!r})"


@dataclass(frozen=True)
class PrimalBasis:
    """``PrimalBasis(b)`` — the primal basis of a dual basis."""

    basis: Basis

    def __post_init__(self) -> None:
        _require_basis(self.basis, "PrimalBasis")

    def __repr__(self) -> str:
        return f"PrimalBasis({self.basis!r})"


@dataclass(frozen=True)
class BasisSum:
    """``BasisSum(b…)`` — n-ary ⊕ basis."""

    bases: tuple[Basis, ...]

    def __post_init__(self) -> None:
        for b in self.bases:
            _require_basis(b, "BasisSum")

    def __repr__(self) -> str:
        return f"BasisSum({_tuple_repr(self.bases)})"


@dataclass(frozen=True)
class BasisProd:
    """``BasisProd(b…)`` — n-ary ⊗ basis."""

    bases: tuple[Basis, ...]

    def __post_init__(self) -> None:
        if not self.bases:
            raise BasisError("BasisProd needs at least one factor")
        for b in self.bases:
            _require_basis(b, "BasisProd")

    def __repr__(self) -> str:
        return f"BasisProd({_tuple_repr(self.bases)})"


@dataclass(frozen=True)
class BasisRep:
    """``BasisRep(n, b)`` — ⊕ⁿ basis (mirrors ``OPlusRep``)."""

    count: Dim
    basis: Basis

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", as_dim(self.count))
        _require_basis(self.basis, "BasisRep")

    def __repr__(self) -> str:
        return f"BasisRep({self.count!r}, {self.basis!r})"


@dataclass(frozen=True)
class ProjectedBasis:
    """``ProjectedBasis(f, which, source?, target?)`` — subspace basis by projection.

    For ``Image`` / ``Ker`` / ``CoImg`` / ``CoKer`` (``which`` ∈ :data:`PROJECTED_WHICH`).
    ``source`` / ``target`` are the bases of ``f``'s domain / codomain (default the
    standard basis, represented as ``None`` here).

    ``which == "image"`` is the **Q / orthonormal range** of ``f`` (the immersed side):
    realized as the ``Q`` factor of ``QrOp`` when ``f`` is injective (static rank), else
    ``U[:, :δ_f]`` of ``SvdOp`` (`73 Topic 2`). It is the orthonormalized counterpart of
    the literal :class:`ImageBasis` (the raw map-application columns); the two are related
    by the ``QrOp.R``/``Q`` (injective) or SVD factors (general) change of basis.
    """

    f: Term  # a linear-map Term
    which: str
    source: Basis | None = None
    target: Basis | None = None

    def __post_init__(self) -> None:
        if self.f is None:
            raise BasisError("ProjectedBasis: linear-map argument `f` is required")
        if self.which not in PROJECTED_WHICH:
            raise BasisError(
                f"ProjectedBasis: which must be one of {PROJECTED_WHICH!r}, got {self.which!r}"
            )
        if self.source is not None:
            _require_basis(self.source, "ProjectedBasis.source")
        if self.target is not None:
            _require_basis(self.target, "ProjectedBasis.target")

    def __repr__(self) -> str:
        parts = [repr(self.f), repr(self.which)]
        if self.source is not None:
            parts.append(f"source={self.source!r}")
        if self.target is not None:
            parts.append(f"target={self.target!r}")
        return f"ProjectedBasis({', '.join(parts)})"


@dataclass(frozen=True)
class ImageBasis:
    """``ImageBasis(f)`` — the **literal map-application** basis of ``f``'s range.

    Its columns are ``f`` applied to the source/immersed basis (the ``f(eᵢ)`` vectors,
    the R/fwd side), **not** orthonormalized.  This is the natural / iso side of the map
    (FEMOracle's source-side ``IsoBasis``, realized by ``R`` for an injective non-square
    map and ``fwd`` when full — `73 Topic 2`).  The *orthonormalized* range basis (the Q
    of QR / first ``δ_f`` columns of the SVD ``U``) is :class:`ProjectedBasis` with
    ``which == "image"``; the two are connected by a change of basis (``QrOp.R``/``Q`` or
    the SVD factors), so a value in one is expressible in the other.
    """

    f: Term

    def __post_init__(self) -> None:
        if self.f is None:
            raise BasisError("ImageBasis: linear-map argument `f` is required")

    def __repr__(self) -> str:
        return f"ImageBasis({self.f!r})"


@dataclass(frozen=True)
class ListBasis:
    """``ListBasis(vectors…)`` — a basis given as an EXPLICIT LIST OF TERM VECTORS.

    The first-class, term-level analogue of :class:`BaseBasis`'s precomputed ``vectors``
    numpy array: each entry is a :class:`~grassmann.core.term.Term` vector (all of the
    same space), and the basis is their span in the listed order.  At lowering each
    vector is represented and the columns are stacked into the basis matrix.
    """

    vectors: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not self.vectors:
            raise BasisError("ListBasis needs at least one vector")
        for v in self.vectors:
            if v is None:
                raise BasisError("ListBasis: a basis vector must be a Term, got None")

    def __repr__(self) -> str:
        return f"ListBasis({_tuple_repr(self.vectors)})"


@dataclass(frozen=True)
class WithInnerBasis:
    """``WithInnerBasis(b, g)`` — a ``g``-orthonormal basis."""

    basis: Basis
    metric: Term

    def __post_init__(self) -> None:
        _require_basis(self.basis, "WithInnerBasis")

    def __repr__(self) -> str:
        return f"WithInnerBasis({self.basis!r}, {self.metric!r})"


@dataclass(frozen=True)
class CompleteBasis:
    """``CompleteBasis(subBasis, complement)`` — complete a subspace basis to its
    ambient. ``complement`` is user-supplied. embed = pad-zeros, project =
    drop-complement (realized FROM this; no Embed/ProjectBasis)."""

    sub_basis: Basis
    complement: object

    def __post_init__(self) -> None:
        _require_basis(self.sub_basis, "CompleteBasis.sub_basis")
        if self.complement is None:
            raise BasisError("CompleteBasis: complement is required (user-supplied)")

    def __repr__(self) -> str:
        return f"CompleteBasis({self.sub_basis!r}, {self.complement!r})"


@dataclass(frozen=True)
class BLam:
    """``BLam(b:σ̂. bodyBasis)`` — basis-level binder; machinery to BUILD a basis.

    Domain ``σ`` (a :class:`~grassmann.core.kinds.Kind`) decides the instance:
    a value (``SpaceSort``) ⇒ a **Frame**; a basis ⇒ a **basis function**; a nat ⇒ a
    nat-parametric builder. ``body`` is the produced :class:`Basis`. ``term_var`` /
    ``basis_var`` hold the bound handle when applicable.
    """

    binders: tuple[tuple[str, Kind], ...]
    body: Basis
    term_var: Var | None = None
    basis_var: BVar | None = None

    def __post_init__(self) -> None:
        if not self.binders:
            raise BasisError("BLam needs at least one binder")
        for name, sort in self.binders:
            if not isinstance(name, str):
                raise BasisError(f"BLam: binder name must be a str, got {name!r}")
            if not is_kind(sort):
                raise BasisError(f"BLam: binder sort must be a sort/kind, got {sort!r}")
        _require_basis(self.body, "BLam.body")

    def __repr__(self) -> str:
        binders = _tuple_repr(self.binders)
        extra = ""
        if self.term_var is not None or self.basis_var is not None:
            extra = f", {self.term_var!r}, {self.basis_var!r}"
        return f"BLam({binders}, {self.body!r}{extra})"


@dataclass(frozen=True)
class BApp:
    """``BApp(f, positions, args…)`` — apply a ``BLam`` (mirrors ``App``/``Eval``)."""

    f: Basis
    positions: tuple[int, ...]
    args: tuple[Term | Space | Dim | Basis, ...]

    def __post_init__(self) -> None:
        _require_basis(self.f, "BApp.f")
        if len(self.positions) != len(self.args):
            raise BasisError(
                f"BApp: positions ({len(self.positions)}) and args "
                f"({len(self.args)}) must have equal length"
            )
        for p in self.positions:
            if not isinstance(p, int) or p < 0:
                raise BasisError(f"BApp: positions must be non-negative ints, got {p!r}")

    def __repr__(self) -> str:
        return f"BApp({self.f!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.args)})"


Basis = ScalarBasis | BaseBasis | BVar | DualBasis | PrimalBasis | BasisSum | BasisProd | BasisRep | ProjectedBasis | ImageBasis | ListBasis | WithInnerBasis | CompleteBasis | BLam | BApp

_BASIS_TYPES = (
    ScalarBasis,
    BaseBasis,
    BVar,
    DualBasis,
    PrimalBasis,
    BasisSum,
    BasisProd,
    BasisRep,
    ProjectedBasis,
    ImageBasis,
    ListBasis,
    WithInnerBasis,
    CompleteBasis,
    BLam,
    BApp,
)


def is_basis(b: object) -> bool:
    """Return True iff ``b`` is a well-formed Basis node."""
    return isinstance(b, _BASIS_TYPES)


def _require_basis(b: object, ctx: str) -> None:
    if not is_basis(b):
        raise BasisError(f"{ctx}: expected a Basis, got {b!r}")


# ---------------------------------------------------------------------------
# Smart constructors.
# ---------------------------------------------------------------------------


def scalar_basis() -> ScalarBasis:
    return ScalarBasis()


def base_basis(
    name: str,
    vectors: object | None = None,
    space: object | None = None,
    backup: object | None = None,
) -> BaseBasis:
    return BaseBasis(name, vectors, space, backup)


def bvar(name: str) -> BVar:
    return BVar(name)


def dual_basis(b: Basis) -> DualBasis:
    return DualBasis(b)


def primal_basis(b: Basis) -> PrimalBasis:
    return PrimalBasis(b)


def basis_sum(*bases: Basis) -> BasisSum:
    return BasisSum(tuple(bases))


def basis_prod(*bases: Basis) -> BasisProd:
    return BasisProd(tuple(bases))


def basis_rep(count: object, b: Basis) -> BasisRep:
    return BasisRep(as_dim(count), b)


def projected_basis(
    f: Term, which: str, source: Basis | None = None, target: Basis | None = None
) -> ProjectedBasis:
    return ProjectedBasis(f, which, source, target)


def image_basis(f: Term) -> ImageBasis:
    return ImageBasis(f)


def list_basis(*vectors: Term) -> ListBasis:
    return ListBasis(tuple(vectors))


def with_inner_basis(b: Basis, metric: Term) -> WithInnerBasis:
    return WithInnerBasis(b, metric)


def complete_basis(sub_basis: Basis, complement: object) -> CompleteBasis:
    return CompleteBasis(sub_basis, complement)


def blam(
    binders: tuple[tuple[str, Kind], ...],
    body: Basis,
    term_var: Var | None = None,
    basis_var: BVar | None = None,
) -> BLam:
    return BLam(tuple(binders), body, term_var, basis_var)


def bapp(f: Basis, positions: tuple[int, ...], *args: Term | Space | Dim | Basis) -> BApp:
    return BApp(f, tuple(positions), tuple(args))


__all__ = [
    "BasisError",
    "PROJECTED_WHICH",
    "Basis",
    "ScalarBasis",
    "BaseBasis",
    "BVar",
    "DualBasis",
    "PrimalBasis",
    "BasisSum",
    "BasisProd",
    "BasisRep",
    "ProjectedBasis",
    "ImageBasis",
    "ListBasis",
    "WithInnerBasis",
    "CompleteBasis",
    "BLam",
    "BApp",
    "is_basis",
    "scalar_basis",
    "base_basis",
    "bvar",
    "dual_basis",
    "primal_basis",
    "basis_sum",
    "basis_prod",
    "basis_rep",
    "projected_basis",
    "image_basis",
    "list_basis",
    "with_inner_basis",
    "complete_basis",
    "blam",
    "bapp",
]
