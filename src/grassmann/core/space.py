"""Space ADT + symbolic dimensions (`20 §5`).

```
Dim   := symbolic Nat expr   (literal | n | dim(s) | n+m | n·m | …)
Space :=
  | Scalar                         -- ℝ (real only);  ⊥ = OPlus()  (nullary ⊕)
  -- (no Base atom: ℝⁿ = ⊕ⁿ Scalar; Scalar is the only ground)
  | SVar(α)                        -- space variable, bound by a Lam over `space`
  | Dual(Space)                    -- s*
  | OPlus(Space, …)                -- n-ary ⊕
  | OProd(Space, …)                -- n-ary ⊗
  | OPlusRep(Dim n, Space s)       -- ⊕ⁿ s  (symbolic-count direct sum); NO OProdRep
  | Ker/Img/CoKer/CoImg(f, source?, target?)   -- four explicit FFS, NO co-flag
  | WithInner(Space s, g)          -- s re-metricized; g : s*⊗s*
  | App(g, positions, args…)       -- apply a SpaceFunction → a Space
```

Deltas from FEMOracle ``base/algebra.py``: add ``SVar``/``OPlusRep``/``WithInner``/
``App``, symbolic ``Dim``, four explicit FFS (no ``co`` flag); drop ``IsoSpace``;
real-only (no ``Field``); no ``OProdRep``.

Well-formedness here is structural only (every child must itself be a ``Space``,
arities respected). Full *type* checking (dims/properties matching) is IW3.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # `term` imports `space` at runtime, so importing it here would be a true
    # cycle; under TYPE_CHECKING it is only used for forward annotations.
    from .term import Term


class SpaceError(Exception):
    """Raised when a Space (or Dim) is structurally malformed."""


def _tuple_repr(items) -> str:
    """Repr a tuple field so it round-trips through the dataclass constructor.

    Produces ``()`` for empty, ``(x,)`` for one element, ``(x, y, ...)`` otherwise.
    """
    inner = ", ".join(repr(i) for i in items)
    if len(items) == 1:
        return f"({inner},)"
    return f"({inner})"


# ===========================================================================
# Symbolic dimensions (Dim) — a small Nat-expression algebra (`25 §1`).
# ===========================================================================


@dataclass(frozen=True)
class DimLit:
    """A literal non-negative dimension."""

    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or self.value < 0:
            raise SpaceError(f"DimLit must be a non-negative int, got {self.value!r}")

    def __repr__(self) -> str:
        return f"DimLit({self.value})"


@dataclass(frozen=True)
class DimVar:
    """A symbolic dimension parameter (e.g. ``n``, ``m``)."""

    name: str

    def __repr__(self) -> str:
        return f"DimVar({self.name!r})"


@dataclass(frozen=True)
class DimOf:
    """``dim(s)`` — the dimension of a space (used for SVar / opaque spaces)."""

    space: Space

    def __repr__(self) -> str:
        return f"DimOf({self.space!r})"


@dataclass(frozen=True)
class DimAtomRef:
    """A runtime rank/dimension atom ``δ_f`` keyed by a term ``f`` (`25 §1`, F3).

    Opaque Nat the SVD deduces at lowering; ``f`` is the keying term (a forward
    ref to ``Term`` to avoid a runtime term.py import cycle).
    """

    key: Term

    def __repr__(self) -> str:
        return f"DimAtomRef({self.key!r})"


@dataclass(frozen=True)
class DimSum:
    """``n + m + …`` (n-ary sum of dims)."""

    terms: tuple[Dim, ...]

    def __repr__(self) -> str:
        return f"DimSum({_tuple_repr(self.terms)})"


@dataclass(frozen=True)
class DimProd:
    """``n · m · …`` (n-ary product of dims)."""

    terms: tuple[Dim, ...]

    def __repr__(self) -> str:
        return f"DimProd({_tuple_repr(self.terms)})"


@dataclass(frozen=True)
class DimNeg:
    """``-d`` (negated dim) — appears only inside a ``DimSum`` to express subtraction,
    e.g. ``|Ker f| = |dom| + (-rank)``. Keeps the ``Dim`` algebra closed under the
    differences the FFS dimension laws require (`20 §5`)."""

    term: Dim

    def __repr__(self) -> str:
        return f"DimNeg({self.term!r})"


@dataclass(frozen=True)
class DimMod:
    """``a mod b`` (modular reduction of a dim) — closes the ``Dim`` algebra under
    the cyclic/periodic dimensions that e.g. circulant or torus constructions need
    (`20 §5`). Convention: Python-style non-negative remainder, so ``0 ≤ a mod b < b``
    and ``a mod 0`` is undefined (rejected at construction). Mirrors :class:`DimNeg`'s
    style: a thin frozen node; constant-folding happens in :mod:`grassmann.infer.dims`,
    kept symbolic otherwise."""

    term: Dim
    modulus: Dim

    def __repr__(self) -> str:
        return f"DimMod({self.term!r}, {self.modulus!r})"


Dim = DimLit | DimVar | DimOf | DimAtomRef | DimSum | DimProd | DimNeg | DimMod


def is_dim(d: object) -> bool:
    return isinstance(
        d, (DimLit, DimVar, DimOf, DimAtomRef, DimSum, DimProd, DimNeg, DimMod)
    )


def as_dim(d: object) -> Dim:
    """Coerce ints to :class:`DimLit`; pass through existing Dims."""
    if isinstance(d, int):
        return DimLit(d)
    if is_dim(d):
        return d  # type: ignore[return-value]
    raise SpaceError(f"expected a Dim (or int), got {d!r}")


def dim_sum(*ds: object) -> Dim:
    parts = tuple(as_dim(d) for d in ds)
    if not parts:
        return DimLit(0)
    if len(parts) == 1:
        return parts[0]
    return DimSum(parts)


def dim_prod(*ds: object) -> Dim:
    parts = tuple(as_dim(d) for d in ds)
    if not parts:
        return DimLit(1)
    if len(parts) == 1:
        return parts[0]
    return DimProd(parts)


def dim_mod(term: object, modulus: object) -> Dim:
    """``term mod modulus`` (Python-style non-negative remainder).

    Coerces ints to :class:`DimLit`. The modulus must be non-zero; a literal ``0``
    modulus is rejected here, and a symbolic-but-statically-zero modulus is caught by
    the constant-folder in :mod:`grassmann.infer.dims`. No folding is done here (kept
    structural, mirroring :func:`dim_sum`/:func:`dim_prod`)."""
    m = as_dim(modulus)
    if isinstance(m, DimLit) and m.value == 0:
        raise SpaceError("DimMod: modulus must be non-zero")
    return DimMod(as_dim(term), m)


# ===========================================================================
# Space ADT.
# ===========================================================================


@dataclass(frozen=True)
class Scalar:
    """ℝ — the scalar space (dim 1, real only)."""

    def __repr__(self) -> str:
        return "Scalar()"


@dataclass(frozen=True)
class SVar:
    """A space variable, bound by a ``Lam`` over the sort ``space`` (`20 §5`)."""

    name: str

    def __repr__(self) -> str:
        return f"SVar({self.name!r})"


@dataclass(frozen=True)
class Dual:
    """``s*`` — the dual of a space."""

    space: Space

    def __post_init__(self) -> None:
        _require_space(self.space, "Dual")

    def __repr__(self) -> str:
        return f"Dual({self.space!r})"


@dataclass(frozen=True)
class OPlus:
    """``⊕`` — n-ary direct sum. Nullary ``OPlus()`` is the bottom space ⊥."""

    spaces: tuple[Space, ...]

    def __post_init__(self) -> None:
        for s in self.spaces:
            _require_space(s, "OPlus")

    def __repr__(self) -> str:
        return f"OPlus({_tuple_repr(self.spaces)})"


@dataclass(frozen=True)
class OProd:
    """``⊗`` — n-ary tensor product."""

    spaces: tuple[Space, ...]

    def __post_init__(self) -> None:
        for s in self.spaces:
            _require_space(s, "OProd")

    def __repr__(self) -> str:
        return f"OProd({_tuple_repr(self.spaces)})"


@dataclass(frozen=True)
class OPlusRep:
    """``⊕ⁿ s`` — symbolic-count direct sum (`20 §5`). No ``OProdRep``."""

    count: Dim
    space: Space

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", as_dim(self.count))
        _require_space(self.space, "OPlusRep")

    def __repr__(self) -> str:
        return f"OPlusRep({self.count!r}, {self.space!r})"


# --- The four explicit FFS (no `co` flag); source/target optional. ----------


@dataclass(frozen=True)
class Ker:
    """``Ker(f)`` — kernel of a linear map. source = dom(f), target = cod(f)."""

    f: Term  # a linear map; forward ref to avoid a runtime term.py import cycle
    source: Space | None = None
    target: Space | None = None

    def __post_init__(self) -> None:
        _check_ffs_endpoints(self)

    def __repr__(self) -> str:
        return _ffs_repr("Ker", self)


@dataclass(frozen=True)
class Img:
    """``Img(f)`` — image (range) of a linear map."""

    f: Term
    source: Space | None = None
    target: Space | None = None

    def __post_init__(self) -> None:
        _check_ffs_endpoints(self)

    def __repr__(self) -> str:
        return _ffs_repr("Img", self)


@dataclass(frozen=True)
class CoKer:
    """``CoKer(f)`` — cokernel of a linear map."""

    f: Term
    source: Space | None = None
    target: Space | None = None

    def __post_init__(self) -> None:
        _check_ffs_endpoints(self)

    def __repr__(self) -> str:
        return _ffs_repr("CoKer", self)


@dataclass(frozen=True)
class CoImg:
    """``CoImg(f)`` — coimage of a linear map."""

    f: Term
    source: Space | None = None
    target: Space | None = None

    def __post_init__(self) -> None:
        _check_ffs_endpoints(self)

    def __repr__(self) -> str:
        return _ffs_repr("CoImg", self)


@dataclass(frozen=True)
class WithInner:
    """``WithInner(s, g)`` — ``s`` re-metricized by ``g : s*⊗s*`` (`20` point 9).

    ``g`` is a metric Term (a forward ref to ``Term`` to avoid a runtime cycle);
    the structural check is only that ``s`` is a space.
    """

    space: Space
    metric: Term

    def __post_init__(self) -> None:
        _require_space(self.space, "WithInner")

    def __repr__(self) -> str:
        return f"WithInner({self.space!r}, {self.metric!r})"


@dataclass(frozen=True)
class App:
    """``App(g, positions, args…)`` — apply a SpaceFunction → a Space (`20 §5`).

    ``g`` is a SpaceFunction (a ``Lam`` whose body is a Space); ``positions`` selects
    the slots filled; ``args`` are the supplied arguments (spaces, nats, or values).
    Forward refs (``Term``) avoid a runtime cycle — kind-level checking of the
    application is IW3.
    """

    g: Term
    positions: tuple[int, ...]
    args: tuple[Space | Dim | Term, ...]

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.args):
            raise SpaceError(
                f"App: positions ({len(self.positions)}) and args "
                f"({len(self.args)}) must have equal length"
            )
        for p in self.positions:
            if not isinstance(p, int) or p < 0:
                raise SpaceError(f"App: positions must be non-negative ints, got {p!r}")

    def __repr__(self) -> str:
        return f"App({self.g!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.args)})"


_meta_counter = itertools.count()


@dataclass(frozen=True)
class SpaceMeta:
    """A space metavariable ``?α`` — an unsolved space hole used during elaboration
    (`25 §0`). A first-class Space node so it can sit inside ``Dual``/``OProd``/… while
    inference resolves it; the resolution store ``Θ`` lives in
    :mod:`grassmann.infer.unify`. ``uid`` is identity; ``hint`` a name for messages."""

    uid: int = field(default_factory=lambda: next(_meta_counter))
    hint: str = "α"

    def __repr__(self) -> str:
        return f"?{self.hint}{self.uid}"


# A Space is any of the node types above.
Space = (
    Scalar | SVar | Dual | OPlus | OProd | OPlusRep
    | Ker | Img | CoKer | CoImg | WithInner | App | SpaceMeta
)

_SPACE_TYPES = (
    Scalar,
    SVar,
    Dual,
    OPlus,
    OProd,
    OPlusRep,
    Ker,
    Img,
    CoKer,
    CoImg,
    WithInner,
    App,
    SpaceMeta,
)


def is_space(s: object) -> bool:
    """Return True iff ``s`` is a well-formed Space node."""
    return isinstance(s, _SPACE_TYPES)


def _require_space(s: object, ctx: str) -> None:
    if not is_space(s):
        raise SpaceError(f"{ctx}: expected a Space, got {s!r}")


def _check_ffs_endpoints(node: object) -> None:
    src = getattr(node, "source")
    tgt = getattr(node, "target")
    cls = type(node).__name__
    if src is not None:
        _require_space(src, f"{cls}.source")
    if tgt is not None:
        _require_space(tgt, f"{cls}.target")
    # f must be present; full "f is a linear map" check is IW3.
    if getattr(node, "f") is None:
        raise SpaceError(f"{cls}: linear-map argument `f` is required")


def _ffs_repr(name: str, node: object) -> str:
    parts = [repr(getattr(node, "f"))]
    src = getattr(node, "source")
    tgt = getattr(node, "target")
    if src is not None:
        parts.append(f"source={src!r}")
    if tgt is not None:
        parts.append(f"target={tgt!r}")
    return f"{name}({', '.join(parts)})"


# ===========================================================================
# Smart constructors (enforce arity / structural kind well-formedness).
# ===========================================================================


def scalar() -> Scalar:
    return Scalar()


# ⊥ = OPlus()  (nullary ⊕)
def bot() -> OPlus:
    return OPlus(())


def base(name: str, dim: object) -> OPlusRep:
    """A primitive ``ℝⁿ`` — grounded on ``Scalar`` as ``⊕ⁿ ℝ`` (`Base` is gone; `Scalar` is
    the only atom).  ``name`` is accepted for call-site compatibility but carries no identity
    (two equal-dimension primitives are the SAME space)."""
    return OPlusRep(as_dim(dim), Scalar())


def svar(name: str) -> SVar:
    return SVar(name)


def dual(s: Space) -> Dual:
    return Dual(s)


def oplus(*spaces: Space) -> OPlus:
    """n-ary ⊕ (variadic, may be nullary = ⊥)."""
    return OPlus(tuple(spaces))


def oprod(*spaces: Space) -> OProd:
    """n-ary ⊗ (variadic). Requires at least one factor."""
    if not spaces:
        raise SpaceError("OProd needs at least one factor")
    return OProd(tuple(spaces))


def oplus_rep(count: object, space: Space) -> OPlusRep:
    return OPlusRep(as_dim(count), space)


def ker(f: Term, source: Space | None = None, target: Space | None = None) -> Ker:
    return Ker(f, source, target)


def img(f: Term, source: Space | None = None, target: Space | None = None) -> Img:
    return Img(f, source, target)


def coker(f: Term, source: Space | None = None, target: Space | None = None) -> CoKer:
    return CoKer(f, source, target)


def coimg(f: Term, source: Space | None = None, target: Space | None = None) -> CoImg:
    return CoImg(f, source, target)


def with_inner(s: Space, metric: Term) -> WithInner:
    return WithInner(s, metric)


def space_app(g: Term, positions: tuple[int, ...], *args: Space | Dim | Term) -> App:
    return App(g, tuple(positions), tuple(args))


# Sugar: linear-map space  s0 ⊸ s1  ≡  ~s0 ⊗ s1  (`25 §F4`).
def linear_map_space(s0: Space, s1: Space) -> OProd:
    """``s0 ⊸ s1 = ~s0 * s1`` — the space of linear maps ``s0 → s1``."""
    return oprod(dual(s0), s1)


__all__ = [
    "SpaceError",
    # Dim
    "Dim",
    "DimLit",
    "DimVar",
    "DimOf",
    "DimAtomRef",
    "DimSum",
    "DimProd",
    "DimNeg",
    "DimMod",
    "is_dim",
    "as_dim",
    "dim_sum",
    "dim_prod",
    "dim_mod",
    # Space nodes
    "Space",
    "SpaceMeta",
    "Scalar",
    "SVar",
    "Dual",
    "OPlus",
    "OProd",
    "OPlusRep",
    "Ker",
    "Img",
    "CoKer",
    "CoImg",
    "WithInner",
    "App",
    "is_space",
    # smart constructors
    "scalar",
    "bot",
    "base",
    "svar",
    "dual",
    "oplus",
    "oprod",
    "oplus_rep",
    "ker",
    "img",
    "coker",
    "coimg",
    "with_inner",
    "space_app",
    "linear_map_space",
]
