"""Basis tower (`20 §7`) — first-class, mirrors Space / Type / the Function layer.

```
Basis  b ::=
  -- mirror Space (§5)
  | ScalarBasis | BaseBasis(name, vectors?)
  | BVar(β)                          basis variable (the basis of an SVar α)
  | DualBasis(b) | PrimalBasis(b)
  | BasisSum(b…) | BasisProd(b…)     -- ⊕ / ⊗  (n-ary)
  | BasisRep(n, b)                   -- ⊕ⁿ
  | ProjectedBasis(f, which, source?, target?)  -- subspace basis (Ker/CoImg/CoKer)
  | ImageBasis(f)                    -- its OWN construct: the range of f
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
from .space import Dim, _tuple_repr, as_dim
from .term import Var


class BasisError(Exception):
    """Raised when a Basis is structurally malformed."""


# Valid `which` selectors for ProjectedBasis (the projection-built FFS).
PROJECTED_WHICH = ("ker", "coimg", "coker")


@dataclass(frozen=True)
class ScalarBasis:
    """The standard basis of ``Scalar`` (ℝ)."""

    def __repr__(self) -> str:
        return "ScalarBasis()"


@dataclass(frozen=True)
class BaseBasis:
    """A concrete basis of a ``Base`` space; optional explicit ``vectors``."""

    name: str
    vectors: object | None = None

    def __repr__(self) -> str:
        v = f", vectors={self.vectors!r}" if self.vectors is not None else ""
        return f"BaseBasis({self.name!r}{v})"


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

    For ``Ker`` / ``CoImg`` / ``CoKer`` (``which`` ∈ :data:`PROJECTED_WHICH`).
    ``source`` / ``target`` are the bases of ``f``'s domain / codomain (default the
    standard basis, represented as ``None`` here). ``ImageBasis`` is separate.
    """

    f: object  # a linear-map Term
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
    """``ImageBasis(f)`` — the range basis of ``f`` (Q of QR at lowering)."""

    f: object

    def __post_init__(self) -> None:
        if self.f is None:
            raise BasisError("ImageBasis: linear-map argument `f` is required")

    def __repr__(self) -> str:
        return f"ImageBasis({self.f!r})"


@dataclass(frozen=True)
class WithInnerBasis:
    """``WithInnerBasis(b, g)`` — a ``g``-orthonormal basis."""

    basis: Basis
    metric: object

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
    args: tuple[object, ...]

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


Basis = ScalarBasis | BaseBasis | BVar | DualBasis | PrimalBasis | BasisSum | BasisProd | BasisRep | ProjectedBasis | ImageBasis | WithInnerBasis | CompleteBasis | BLam | BApp

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


def base_basis(name: str, vectors: object | None = None) -> BaseBasis:
    return BaseBasis(name, vectors)


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
    f: object, which: str, source: Basis | None = None, target: Basis | None = None
) -> ProjectedBasis:
    return ProjectedBasis(f, which, source, target)


def image_basis(f: object) -> ImageBasis:
    return ImageBasis(f)


def with_inner_basis(b: Basis, metric: object) -> WithInnerBasis:
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


def bapp(f: Basis, positions: tuple[int, ...], *args: object) -> BApp:
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
    "with_inner_basis",
    "complete_basis",
    "blam",
    "bapp",
]
