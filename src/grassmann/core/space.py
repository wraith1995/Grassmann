"""The Space sum type and its symbolic dimensions.

A :class:`Space` is an abstract real vector space. It is one of a closed set of
node kinds, and every space in the language is built from them::

    Scalar                     ℝ, the only ground space (ℝⁿ is ⊕ⁿ Scalar; the
                               empty ⊕ is the zero space)
    SVar(α)                    a space variable, bound by a lambda over spaces
    Dual(s)                    the dual space s*
    OPlus(s, …)                the n-ary direct sum  s₁ ⊕ … ⊕ sₙ
    OProd(s, …)                the n-ary tensor product  s₁ ⊗ … ⊗ sₙ
    OPlusRep(n, s)             ⊕ⁿ s, a direct sum whose count n is symbolic
    Ker/Img/CoKer/CoImg(f)     the four fundamental subspaces of a map f
    WithInner(s, g)            s re-metricized by an inner product g : s* ⊗ s*
    App(g, positions, args…)   the space produced by applying a space function g

The four fundamental subspaces are first-class spaces, not a flag on a map, so
one builds further maps against ``img(f)`` or ``ker(f)`` directly. A metricized
space ``WithInner(s, g)`` is a genuinely different space from ``s``; crossing
between them is the explicit :class:`~grassmann.core.term.Remetric`, never a
silent identification. A dimension :class:`Dim` is a symbolic natural-number
expression — a literal, a variable, the dimension of a space, or a sum or product
of these — so a space can have a size that is known only symbolically.

Well-formedness enforced here is structural only: every child is itself a
:class:`Space`, and arities are respected. Matching dimensions and map
properties is type checking, done in :mod:`grassmann.infer`.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    # `term` imports `space` at runtime, so importing it here would be a true
    # cycle; under TYPE_CHECKING it is only used for forward annotations.
    from .term import Term


class SpaceError(Exception):
    """Raised when a Space (or Dim) is structurally malformed."""


def _tuple_repr(items: Sequence[object]) -> str:
    """Repr a tuple field so it round-trips through the dataclass constructor.

    Produces ``()`` for empty, ``(x,)`` for one element, ``(x, y, ...)`` otherwise.
    """
    inner = ", ".join(repr(i) for i in items)
    if len(items) == 1:
        return f"({inner},)"
    return f"({inner})"


# ===========================================================================
# Symbolic dimensions (Dim) — a small Nat-expression algebra.
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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise SpaceError(f"DimVar.name: expected a str, got {self.name!r}")

    def __repr__(self) -> str:
        return f"DimVar({self.name!r})"


@dataclass(frozen=True)
class DimOf:
    """The dimension of a space, used for a space variable or an opaque space."""

    space: Space

    def __post_init__(self) -> None:
        _require_space(self.space, "DimOf")

    def __repr__(self) -> str:
        return f"DimOf({self.space!r})"


@dataclass(frozen=True)
class DimAtomRef:
    """A runtime rank/dimension atom ``δ_f`` keyed by a term ``f``.

    Opaque Nat the SVD deduces at lowering; ``f`` is the keying term (a forward
    ref to ``Term`` to avoid a runtime term.py import cycle).
    """

    key: Term

    def __post_init__(self) -> None:
        _require_term(self.key, "DimAtomRef.key")

    def __repr__(self) -> str:
        return f"DimAtomRef({self.key!r})"


@dataclass(frozen=True)
class DimSum:
    """The n-ary sum of dimensions ``n + m + …``."""

    terms: tuple[Dim, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(as_dim(t) for t in self.terms))

    def __repr__(self) -> str:
        return f"DimSum({_tuple_repr(self.terms)})"


@dataclass(frozen=True)
class DimProd:
    """The n-ary product of dimensions ``n · m · …``."""

    terms: tuple[Dim, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(as_dim(t) for t in self.terms))

    def __repr__(self) -> str:
        return f"DimProd({_tuple_repr(self.terms)})"


@dataclass(frozen=True)
class DimNeg:
    """A negated dimension.

    Appears only inside a ``DimSum``, where it expresses subtraction — for example
    ``|Ker f| = |dom| + (-rank)``. Keeps the ``Dim`` algebra closed under the differences the FFS
    dimension laws require.
    """

    term: Dim

    def __post_init__(self) -> None:
        object.__setattr__(self, "term", as_dim(self.term))

    def __repr__(self) -> str:
        return f"DimNeg({self.term!r})"


@dataclass(frozen=True)
class DimMod:
    """A dimension reduced modulo another.

    Closes the ``Dim`` algebra under the cyclic/periodic dimensions that e.g. circulant or torus constructions
    need. Convention: Python-style non-negative remainder, so ``0 ≤ a mod b < b`` and ``a mod 0`` is
    undefined (rejected at construction). Mirrors :class:`DimNeg`'s style: a thin frozen node;
    constant-folding happens in :mod:`grassmann.infer.dims`, kept symbolic otherwise.
    """

    term: Dim
    modulus: Dim

    def __post_init__(self) -> None:
        object.__setattr__(self, "term", as_dim(self.term))
        modulus = as_dim(self.modulus)
        if isinstance(modulus, DimLit) and modulus.value == 0:
            raise SpaceError("DimMod: modulus must be non-zero")
        object.__setattr__(self, "modulus", modulus)

    def __repr__(self) -> str:
        return f"DimMod({self.term!r}, {self.modulus!r})"


Dim = DimLit | DimVar | DimOf | DimAtomRef | DimSum | DimProd | DimNeg | DimMod


def is_dim(d: object) -> bool:
    """Report whether ``d`` is a Dim node."""
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
    """Build the symbolic sum of ``ds`` (empty ⇒ ``0``, single ⇒ itself)."""
    parts = tuple(as_dim(d) for d in ds)
    if not parts:
        return DimLit(0)
    if len(parts) == 1:
        return parts[0]
    return DimSum(parts)


def dim_prod(*ds: object) -> Dim:
    """Build the symbolic product of ``ds`` (empty ⇒ ``1``, single ⇒ itself)."""
    parts = tuple(as_dim(d) for d in ds)
    if not parts:
        return DimLit(1)
    if len(parts) == 1:
        return parts[0]
    return DimProd(parts)


def dim_mod(term: object, modulus: object) -> Dim:
    """Build the reduction ``term mod modulus`` with the Python-style non-negative remainder.

    Coerces ints to :class:`DimLit`. The modulus must be non-zero; a literal ``0``
    modulus is rejected here, and a symbolic-but-statically-zero modulus is caught by
    the constant-folder in :mod:`grassmann.infer.dims`. No folding is done here (kept
    structural, mirroring :func:`dim_sum`/:func:`dim_prod`).
    """
    m = as_dim(modulus)
    if isinstance(m, DimLit) and m.value == 0:
        raise SpaceError("DimMod: modulus must be non-zero")
    return DimMod(as_dim(term), m)


# ===========================================================================
# Space ADT.
# ===========================================================================


@dataclass(frozen=True)
class Scalar:
    """The scalar space ``ℝ``, the one-dimensional ground space (real only)."""

    def __repr__(self) -> str:
        return "Scalar()"


@dataclass(frozen=True)
class SVar:
    """A space variable, bound by a ``Lam`` over the sort ``space``."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise SpaceError(f"SVar.name: expected a str, got {self.name!r}")

    def __repr__(self) -> str:
        return f"SVar({self.name!r})"


@dataclass(frozen=True)
class Dual:
    """The dual space ``s*``: the space of linear functionals on ``s``."""

    space: Space

    def __post_init__(self) -> None:
        _require_space(self.space, "Dual")

    def __repr__(self) -> str:
        return f"Dual({self.space!r})"


@dataclass(frozen=True)
class OPlus:
    """The n-ary direct sum of spaces. The nullary ``OPlus()`` is the bottom space ⊥."""

    spaces: tuple[Space, ...]

    def __post_init__(self) -> None:
        for s in self.spaces:
            _require_space(s, "OPlus")

    def __repr__(self) -> str:
        return f"OPlus({_tuple_repr(self.spaces)})"


@dataclass(frozen=True)
class OProd:
    """The n-ary tensor product ``s₁ ⊗ … ⊗ sₙ`` of spaces."""

    spaces: tuple[Space, ...]

    def __post_init__(self) -> None:
        for s in self.spaces:
            _require_space(s, "OProd")

    def __repr__(self) -> str:
        return f"OProd({_tuple_repr(self.spaces)})"


@dataclass(frozen=True)
class OPlusRep:
    """The replicated direct sum ``⊕ⁿ s``, whose count is symbolic (there is no ``OProdRep``)."""

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
    """The kernel of a map: the subspace of its domain that ``f`` sends to zero, a
    first-class space. ``source`` and ``target`` record the bases of ``f``'s domain
    and codomain."""

    f: Term  # a linear map; forward ref to avoid a runtime term.py import cycle
    source: Space | None = None
    target: Space | None = None

    def __post_init__(self) -> None:
        _check_ffs_endpoints(self)

    def __repr__(self) -> str:
        return _ffs_repr("Ker", self)


@dataclass(frozen=True)
class Img:
    """The image of a map: the subspace of its codomain that ``f`` reaches, a
    first-class space."""

    f: Term
    source: Space | None = None
    target: Space | None = None

    def __post_init__(self) -> None:
        _check_ffs_endpoints(self)

    def __repr__(self) -> str:
        return _ffs_repr("Img", self)


@dataclass(frozen=True)
class CoKer:
    """The cokernel of a map: its codomain modulo the image, equivalently the kernel of
    the adjoint. A first-class space."""

    f: Term
    source: Space | None = None
    target: Space | None = None

    def __post_init__(self) -> None:
        _check_ffs_endpoints(self)

    def __repr__(self) -> str:
        return _ffs_repr("CoKer", self)


@dataclass(frozen=True)
class CoImg:
    """The coimage of a map: its domain modulo the kernel, equivalently the orthogonal
    complement of the kernel. A first-class space."""

    f: Term
    source: Space | None = None
    target: Space | None = None

    def __post_init__(self) -> None:
        _check_ffs_endpoints(self)

    def __repr__(self) -> str:
        return _ffs_repr("CoImg", self)


@dataclass(frozen=True)
class WithInner:
    """The space ``s`` re-metricized by an inner product ``g`` — a genuinely different space from ``s``.

    Written ``WithInner(s, g)``, with ``g`` a metric of type ``s*⊗s*``. ``g`` is a metric **Term** (a forward ref to ``Term`` to avoid a runtime cycle): the metric
    is part of the language, NOT a raw numeric array.  A concrete numeric metric is supplied as a
    ``Var`` fed at run time (or any symbolic term), never embedded as an array — that keeps the
    space hashable and the lowering uniform (a single ``_rep(metric)`` path).
    """

    space: Space
    metric: Term

    def __post_init__(self) -> None:
        _require_space(self.space, "WithInner")
        _require_term(self.metric, "WithInner.metric (the metric must be a Term, not an array)")

    def __repr__(self) -> str:
        return f"WithInner({self.space!r}, {self.metric!r})"


@dataclass(frozen=True)
class App:
    """The space produced by applying a space function to its arguments.

    Written ``App(g, positions, args…)``: ``g`` is a SpaceFunction (a ``Lam`` whose
    body is a Space); ``positions`` selects the slots filled; ``args`` are the supplied
    arguments (spaces, nats, or values). Forward refs (``Term``) avoid a runtime cycle —
    kind-level checking of the application is done during type inference.
    """

    g: Term
    positions: tuple[int, ...]
    args: tuple[Space | Dim | Term, ...]

    def __post_init__(self) -> None:
        from .term import is_term
        _require_term(self.g, "App.g")
        if len(self.positions) != len(self.args):
            raise SpaceError(
                f"App: positions ({len(self.positions)}) and args "
                f"({len(self.args)}) must have equal length"
            )
        for p in self.positions:
            if not isinstance(p, int) or p < 0:
                raise SpaceError(f"App: positions must be non-negative ints, got {p!r}")
        for a in self.args:
            if not (is_space(a) or is_dim(a) or is_term(a)):
                raise SpaceError(f"App.arg: expected a Space/Dim/Term, got {a!r}")

    def __repr__(self) -> str:
        return f"App({self.g!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.args)})"


_meta_counter = itertools.count()


@dataclass(frozen=True)
class SpaceMeta:
    """A space metavariable ``?α`` — an unsolved space hole used during elaboration.

    A first-class Space node so it can sit inside ``Dual``/``OProd``/… while inference resolves it; the
    resolution store ``Θ`` lives in :mod:`grassmann.infer.unify`. ``uid`` is identity; ``hint`` a name for
    messages.
    """

    uid: int = field(default_factory=lambda: next(_meta_counter))
    hint: str = "α"

    def __post_init__(self) -> None:
        if not isinstance(self.uid, int):
            raise SpaceError(f"SpaceMeta.uid: expected an int, got {self.uid!r}")
        if not isinstance(self.hint, str):
            raise SpaceError(f"SpaceMeta.hint: expected a str, got {self.hint!r}")

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


def _require_term(t: object, ctx: str) -> None:
    # Lazy import: ``term`` imports ``space`` at runtime, so the dependency only flows one way.
    from .term import is_term
    if not is_term(t):
        raise SpaceError(f"{ctx}: expected a Term, got {t!r}")


def _check_ffs_endpoints(node: object) -> None:
    src = getattr(node, "source")
    tgt = getattr(node, "target")
    cls = type(node).__name__
    if src is not None:
        _require_space(src, f"{cls}.source")
    if tgt is not None:
        _require_space(tgt, f"{cls}.target")
    # f must be present and a Term; full "f is a linear map" check is done during type inference.
    if getattr(node, "f") is None:
        raise SpaceError(f"{cls}: linear-map argument `f` is required")
    _require_term(getattr(node, "f"), f"{cls}.f")


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
    """Build the scalar space ``ℝ`` — the only atom."""
    return Scalar()


# ⊥ = OPlus()  (nullary ⊕)
def bot() -> OPlus:
    """Build the zero space ``⊥`` (the nullary ⊕)."""
    return OPlus(())


def base(name: str, dim: object) -> OPlusRep:
    """Build a primitive ``ℝⁿ``, grounded on ``Scalar`` as ``⊕ⁿ ℝ``.

    ``name`` is accepted for call-site compatibility but carries no identity: two
    equal-dimension primitives are the SAME space.
    """
    return OPlusRep(as_dim(dim), Scalar())


def svar(name: str) -> SVar:
    """Build an abstract base space named ``name``."""
    return SVar(name)


def dual(s: Space) -> Dual:
    """Build the dual space ``~s``."""
    return Dual(s)


def oplus(*spaces: Space) -> OPlus:
    """Build the n-ary direct sum (variadic; may be nullary, giving ⊥)."""
    return OPlus(tuple(spaces))


def oprod(*spaces: Space) -> OProd:
    """Build the n-ary tensor product (variadic); requires at least one factor."""
    if not spaces:
        raise SpaceError("OProd needs at least one factor")
    return OProd(tuple(spaces))


def spliced_oprod(*parts: Space) -> OProd:
    """Build the flat tensor product of ``parts``, splicing in any operand that is itself a product — ``X ⊗ (Y ⊗ Z) ↦ X ⊗ Y ⊗ Z``.

    ⊗ is associative, so a nested and a flat product denote ONE space; but they are different
    NODES, each top-level factor is a separate ARRAY AXIS (``lower.represent._shape_of_space``),
    and :func:`grassmann.infer.unify.unify` compares ``OProd`` arities.  The two spellings are
    therefore two LAYOUTS that do not meet: from three leaves up they raise
    ``AssociationMismatch``.  This builder is the FLAT spelling, and the ⊗-towers that want one
    axis per leaf are built with it (``stdlib.exterior.tensor_power``,
    ``infer.synth._spliced_oprod`` for ``Kron``'s endpoints, ``transform.diff``'s derivative
    tower).

    Parameters
    ----------
    *parts
        The factors.  A part that is itself a multi-factor ``OProd`` is spliced in; anything else
        contributes one factor.

    Returns
    -------
    OProd
        The flat product.  Leaf ORDER is preserved exactly —
        ``tensor_leaves(spliced_oprod(*parts))`` is the concatenation of the parts' leaves — and
        only one nesting level is removed per operand, which is all any producer here builds.

    Notes
    -----
    ⚠ **Not a canonicalizer, and NOT what** :func:`oprod` **should do.**  A map space
    ``linear_map_space(dom, cod) = Dual(dom) ⊗ cod`` is deliberately TWO factors — an
    ``(|dom|, |cod|)`` MATRIX — however compound ``cod`` is; splicing it hands the FFS/SVD seam a
    rank-``2k`` array where it expects the ``(nᵏ, nᵏ)`` matrix (``Alt₃ : V⊗³ ⊸ V⊗³`` would stop
    being ``(27, 27)``).  Use this ONLY where the flat, one-axis-per-leaf layout is the intended
    one; use :func:`linear_map_space` where a matrix is.
    """
    flat: list[Space] = []
    for p in parts:
        if isinstance(p, OProd) and len(p.spaces) > 1:
            flat.extend(p.spaces)
        else:
            flat.append(p)
    return oprod(*flat)


def drop_unit_factors(s: Space) -> Space:
    """Apply the tensor-unit law ``𝟙 ⊗ λ ≅ λ`` — an explicit, opt-in canonicalizer.

    Drops the ⊗-unit ``Scalar`` factors of an ``OProd`` (``𝟙`` is the monoidal unit of ⊗,
    ``|𝟙| = 1``), returning the canonical space they multiply to:

    - ``OProd((𝟙, λ)) → λ``          — a single non-unit survivor is UNWRAPPED (``𝟙⊗λ ≅ λ``);
    - ``OProd((𝟙, λ, μ)) → OProd((λ, μ))``   — units dropped, ≥2 survivors stay a product;
    - ``OProd((𝟙, 𝟙)) → 𝟙``          — an all-unit product ``≅`` the unit itself;
    - a unit-free ``OProd`` is returned UNCHANGED (identity), **including** the deliberate
      1-ary ``OProd((V,))`` that ``tensor_power(V, 1)`` keeps for ``Λ¹`` codomain agreement —
      it has no unit factor, so nothing is dropped;
    - any non-``OProd`` space is returned unchanged.

    Only the literal ⊗-unit ``Scalar()`` is eliminated: a *coincidental* dimension-1 factor
    (a rank-1 ``Img``, an ``SVar`` of dim 1, …) is NOT the unit object, so ``V ⊗ W`` with
    ``|W| = 1`` is left intact (there is no canonical ``V ⊗ W ≅ V`` unless ``W = 𝟙``).

    This is a PURE space operator, applied by the consumer that wants the reduced form (grassmann
    does not fold it silently anywhere): e.g. a transported top-form whose codomain is built as
    ``OProd((𝟙, Img(alt∘J)))`` applies it to expose a plain ``Λⁿ = Img`` codomain.
    """
    match s:
        case OProd(spaces=spaces):
            survivors = tuple(f for f in spaces if f != Scalar())
            if not survivors:
                return Scalar()
            if len(survivors) == len(spaces):
                return s                 # unit-free (incl. OProd((V,))) — identity, no churn
            if len(survivors) == 1:
                return survivors[0]      # 𝟙 ⊗ λ ≅ λ — unwrap the lone non-unit factor
            return OProd(survivors)
        case _:
            return s


def tensor_leaves(s: Space) -> list[Space]:
    """Give the ordered ⊗-leaves of a (possibly nested) ``OProd``.

    The SPACE-level twin of ``lower.represent._basis_prod_leaves``.  ⊗ is associative on the
    underlying space, so ``a⊗(b⊗c)``, ``(a⊗b)⊗c`` and ``a⊗b⊗c`` all have THIS leaf sequence; what
    differs between them is only the NESTING — how the leaves are grouped into array axes, which
    is what ``lower.represent._shape_of_space`` reads.  This is therefore the reader that says
    "same leaves?" while leaving "same layout?" to the caller; the two questions are genuinely
    different, and a caller that conflates them lets mismatched layouts meet silently.

    Parameters
    ----------
    s
        Any space.  Only ``OProd`` is descended through.

    Returns
    -------
    list[Space]
        The leaves in ⊗ order.  A non-product space is its own single leaf, so
        ``tensor_leaves(V) == [V]``.

    Notes
    -----
    ⚠ **``Dual`` is a LEAF, not a distribution point.**  ``Dual(A⊗B)`` is one axis of size
    ``|A|·|B|`` (``_shape_of_space``'s single-axis default) while ``Dual(A)⊗Dual(B)`` is two axes;
    they are different layouts and this reader must not identify them.  ``_basis_prod_leaves``
    makes the same call — ``default_basis(Dual(A⊗B))`` is a ``DualBasis``, not a ``BasisProd``, so
    it stops there too.

    ⚠ The ⊗-unit ``Scalar()`` is KEPT as a leaf: it occupies no array axis but it does occupy a
    factor position, and dropping it here would put this reader out of step with
    ``_basis_prod_leaves`` (whose ``BasisProd((…, ScalarBasis()))`` tails are leaves).  A caller
    that wants the unit-free sequence filters it itself — ``drop_unit_factors`` is the
    canonicalizer and it, too, is opt-in.

    Pure and total: no error paths, no normalization, no interning.
    """
    match s:
        case OProd(spaces=spaces):
            out: list[Space] = []
            for f in spaces:
                out.extend(tensor_leaves(f))
            return out
        case (Scalar() | SVar() | Dual() | OPlus() | OPlusRep()
              | Ker() | Img() | CoKer() | CoImg() | WithInner() | App() | SpaceMeta()):
            return [s]
        case _ as unreachable:
            assert_never(unreachable)


def oplus_rep(count: object, space: Space) -> OPlusRep:
    """Build the ``count``-fold replicated sum ``⊕^count space``."""
    return OPlusRep(as_dim(count), space)


def ker(f: Term, source: Space | None = None, target: Space | None = None) -> Ker:
    """Build the kernel space ``Ker(f)``."""
    return Ker(f, source, target)


def img(f: Term, source: Space | None = None, target: Space | None = None) -> Img:
    """Build the image space ``Img(f)``."""
    return Img(f, source, target)


def coker(f: Term, source: Space | None = None, target: Space | None = None) -> CoKer:
    """Build the cokernel space ``CoKer(f)``."""
    return CoKer(f, source, target)


def coimg(f: Term, source: Space | None = None, target: Space | None = None) -> CoImg:
    """Build the coimage space ``CoImg(f)``."""
    return CoImg(f, source, target)


def with_inner(s: Space, metric: Term) -> WithInner:
    """Build ``WithInner(s, metric)`` — ``s`` re-metricized, a DIFFERENT space."""
    return WithInner(s, metric)


def space_app(g: Term, positions: tuple[int, ...], *args: Space | Dim | Term) -> App:
    """Apply the space-level function ``g`` to ``args`` at ``positions``."""
    return App(g, tuple(positions), tuple(args))


# Sugar: linear-map space  s0 ⊸ s1  ≡  ~s0 ⊗ s1.
def linear_map_space(s0: Space, s1: Space) -> OProd:
    """Build the space of linear maps ``s0 → s1``, namely ``~s0 ⊗ s1``."""
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
    "spliced_oprod",
    "drop_unit_factors",
    "tensor_leaves",
    "oplus_rep",
    "ker",
    "img",
    "coker",
    "coimg",
    "with_inner",
    "space_app",
    "linear_map_space",
]
