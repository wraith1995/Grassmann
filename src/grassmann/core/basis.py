"""The Basis sum type: how a space is coordinatized.

A :class:`Basis` records the coordinates chosen for a space. The family mirrors
:class:`~grassmann.core.space.Space` node for node, so every space has a basis
built the same way the space is::

    ScalarBasis                the basis of ℝ, the only ground
    BVar(β)                    a basis variable (the basis of a space variable)
    DualBasis(b) PrimalBasis(b)   the dual and primal of a basis
    BasisSum(b…) BasisProd(b…)    the ⊕ and ⊗ of bases (n-ary)
    BasisRep(n, b)             the basis of ⊕ⁿ s
    ProjectedBasis(f, which)   an orthonormal basis of one of f's four subspaces
    ImageBasis(f)              the literal pushforward columns f(sᵢ) of Img(f)
    ListBasis(vectors…)        an explicit list of vectors, optionally reduced
    WithInnerBasis(b, g)       b carrying a metric g, kept raw
    OrthonormalBasis(b)        the explicit orthonormalization of b
    CompleteBasis(sub, …)      a subspace basis completed to its ambient space
    BLam(b:σ̂. body)            a basis abstracted over a value, basis, or nat
    BApp(f, positions, args…)  the basis produced by applying a basis function

Two bases coordinatize an image differently and deliberately.
:class:`ImageBasis` takes the literal pushforward columns ``f(s₁), …, f(sₙ)`` —
the frame chartlib transports along ``f`` — and does not orthonormalize them.
:class:`ProjectedBasis` gives the orthonormalized basis of a chosen subspace
instead; orthonormalization is always explicit here, never implicit.

``BLam`` and ``BApp`` mirror the term-level ``Lam`` and ``App`` across the whole
tower. A ``BLam`` over a value is a frame — a basis that varies with a runtime
value; the dedicated term node is :class:`~grassmann.core.term.Frame`, and this
is its basis-tower spelling. A ``BLam`` over a basis is a basis function, and
over a nat a nat-parametric builder. A basis stores no coefficients; a change of
basis is recomputed at lowering rather than cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .kinds import Kind, is_kind
from .space import Dim, Space, _tuple_repr, as_dim
from .term import Term, Var, is_term


class Reduce(Enum):
    """How a list/completed basis combines its vectors into a basis.

    * ``NONE`` — use the vectors as given (assumed linearly independent and complete);
    * ``RREF`` — greedy RREF pivot: keep the linearly-independent columns (and complete from a
      ``backup`` basis when rank-deficient);
    * ``ORTHONORMALIZE`` — orthonormalize the columns (QR / Gram–Schmidt).
    """

    NONE = "none"
    RREF = "rref"
    ORTHONORMALIZE = "orthonormalize"


class TargetBasis(Enum):
    """A *policy* for the basis ``represent``/``_rep`` lower a node into (not a basis node).

    Used as the ``target`` of :func:`grassmann.lower.represent` (and ``_rep``) alongside a
    concrete :class:`Basis`:

    * ``AUTO`` — free choice by the underlying system: each space keeps its natural basis
      (a ``basis_env`` pin, else the min-symbolic ``choose``).  The default.
    * ``STD`` — the standard orthonormal basis of each space (``default_basis(space)``),
      propagated through the whole recursion so every node is expressed in it.

    A concrete :class:`Basis` target (not this enum) names the *result*'s basis directly.
    """

    AUTO = "auto"
    STD = "std"


#: Free choice by the system (natural / min-symbolic basis) — the ``represent`` default.
AUTO = TargetBasis.AUTO
#: Standard orthonormal basis of each space, propagated through the recursion.
STD = TargetBasis.STD


class BasisError(Exception):
    """Raised when a Basis is structurally malformed."""


# Valid `which` selectors for ProjectedBasis (the projection-built FFS).
#
# ``"image"`` is the **Q / orthonormal range** (the immersed side): the ``Q`` of ``QrOp``
# when ``f`` is asserted injective (static rank ``=|dom|``), else ``U[:, :δ_f]`` of
# ``SvdOp`` in general.  It is the
# **orthonormalized** counterpart of :class:`ImageBasis` (the raw map-application basis);
# the two are related by ``QrOp.R`` / ``QrOp.Q`` (injective) or the SVD factors (general).
PROJECTED_WHICH = ("image", "ker", "coimg", "coker")


@dataclass(frozen=True)
class ScalarBasis:
    """The standard basis of ``Scalar`` (ℝ)."""

    def __repr__(self) -> str:
        return "ScalarBasis()"


@dataclass(frozen=True)
class BVar:
    """A basis variable — the basis of a space variable ``SVar α``, riding on a ``Lam``."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise BasisError(f"BVar.name: expected a str, got {self.name!r}")

    def __repr__(self) -> str:
        return f"BVar({self.name!r})"


@dataclass(frozen=True)
class DualBasis:
    """The dual basis of a basis."""

    basis: Basis

    def __post_init__(self) -> None:
        _require_basis(self.basis, "DualBasis")

    def __repr__(self) -> str:
        return f"DualBasis({self.basis!r})"


@dataclass(frozen=True)
class PrimalBasis:
    """The primal basis underlying a dual basis."""

    basis: Basis

    def __post_init__(self) -> None:
        _require_basis(self.basis, "PrimalBasis")

    def __repr__(self) -> str:
        return f"PrimalBasis({self.basis!r})"


@dataclass(frozen=True)
class BasisSum:
    """The direct-sum basis of finitely many bases (the basis of an ``n``-ary ``⊕``)."""

    bases: tuple[Basis, ...]

    def __post_init__(self) -> None:
        for b in self.bases:
            _require_basis(b, "BasisSum")

    def __repr__(self) -> str:
        return f"BasisSum({_tuple_repr(self.bases)})"


@dataclass(frozen=True)
class BasisProd:
    """The tensor-product basis of finitely many bases (the basis of an ``n``-ary ``⊗``)."""

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
    """The basis of an ``n``-fold direct sum ``⊕ⁿ s``.

    Mirrors the ``OPlusRep`` space at the basis level.
    """

    count: Dim
    basis: Basis

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", as_dim(self.count))
        _require_basis(self.basis, "BasisRep")

    def __repr__(self) -> str:
        return f"BasisRep({self.count!r}, {self.basis!r})"


@dataclass(frozen=True)
class ProjectedBasis:
    """The orthonormalized basis of one of ``f``'s four fundamental subspaces, built by projection.

    The subspace is chosen by ``which`` — ``Image`` / ``Ker`` / ``CoImg`` / ``CoKer``
    (``which`` ∈ :data:`PROJECTED_WHICH`).
    ``source`` / ``target`` are the bases of ``f``'s domain / codomain (default the
    standard basis, represented as ``None`` here).

    ``which == "image"`` is the **Q / orthonormal range** of ``f`` (the immersed side):
    realized as the ``Q`` factor of ``QrOp`` when ``f`` is injective (static rank), else
    ``U[:, :δ_f]`` of ``SvdOp``. It is the orthonormalized counterpart of
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
        _require_term(self.f, "ProjectedBasis.f")
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
    """The literal map-application basis of ``f``'s range — the raw, non-orthonormalized pushforward frame.

    Its columns are ``f`` applied to the **source** basis of ``f``'s domain — ``f(sᵢ)``,
    the R/fwd side — **not** orthonormalized, read in the codomain ``target`` basis.  ``source``
    (a basis of ``dom f``) defaults to the standard basis (``None``), giving the literal columns
    ``f(eᵢ)``; ``target`` (a basis of ``cod f``) likewise defaults to standard.  The
    *orthonormalized* range basis (the ``Q`` of QR / first ``δ_f`` columns of the SVD ``U``) is
    :class:`ProjectedBasis` with ``which == "image"``; the two are connected by a change of basis
    (``QrOp.R``/``Q`` or the SVD factors).
    """

    f: Term
    source: Basis | None = None
    target: Basis | None = None

    def __post_init__(self) -> None:
        if self.f is None:
            raise BasisError("ImageBasis: linear-map argument `f` is required")
        _require_term(self.f, "ImageBasis.f")
        if self.source is not None:
            _require_basis(self.source, "ImageBasis.source")
        if self.target is not None:
            _require_basis(self.target, "ImageBasis.target")

    def __repr__(self) -> str:
        parts = [repr(self.f)]
        if self.source is not None:
            parts.append(f"source={self.source!r}")
        if self.target is not None:
            parts.append(f"target={self.target!r}")
        return f"ImageBasis({', '.join(parts)})"


@dataclass(frozen=True)
class ListBasis:
    """A basis given as an explicit list of term vectors, always a full basis of its ambient.

    A ``ListBasis`` is **ALWAYS a full basis of its ambient** (the vectors' space).  Each entry
    is a :class:`~grassmann.core.term.Term` vector (all of the same space); the listed vectors
    are stacked into the basis matrix at lowering, and a **short** frame (fewer vectors than the
    ambient dim) is completed to the full ambient from the ``backup``.  The *term-level*
    successor of the old ``BaseBasis`` frame:

    * ``reduce`` (:class:`Reduce`) — how to combine the vectors: ``NONE`` (assume independent),
      ``RREF`` (keep the linearly-independent columns then complete), ``ORTHONORMALIZE`` (QR,
      then complete from the orthonormalized backup);
    * ``backup`` — the :class:`Basis` that **completes** a short/rank-deficient frame to the full
      ambient basis (default, when ``None``: the ambient's *standard* basis).  So a short ℝⁿ
      ``ListBasis`` realizes to a full ``n×n`` ambient basis (the explicit vectors plus a
      standard completion), and a full-rank ``n``-vector frame is used unchanged.

    To build a basis of a genuine *subspace* (not its ambient), use a subspace basis directly
    (e.g. a :class:`ProjectedBasis`) and :class:`CompleteBasis` to lift it to the ambient.
    """

    vectors: tuple[Term, ...]
    backup: Basis | None = None
    reduce: Reduce = Reduce.NONE

    def __post_init__(self) -> None:
        if not self.vectors:
            raise BasisError("ListBasis needs at least one vector")
        for v in self.vectors:
            _require_term(v, "ListBasis.vector")
        if self.backup is not None:
            _require_basis(self.backup, "ListBasis.backup")
        if not isinstance(self.reduce, Reduce):
            raise BasisError(f"ListBasis.reduce must be a Reduce, got {self.reduce!r}")

    def __repr__(self) -> str:
        extra = ""
        if self.backup is not None:
            extra += f", backup={self.backup!r}"
        if self.reduce is not Reduce.NONE:
            extra += f", reduce={self.reduce}"
        return f"ListBasis({_tuple_repr(self.vectors)}{extra})"


@dataclass(frozen=True)
class WithInnerBasis:
    """The basis ``b`` living in the metricized space ``WithInner(V, g)``, carrying the metric raw.

    Carries the metric ``g`` as metadata; it is **not** assumed ``g``-orthonormal — it is just
    ``b`` living in the metricized space.  To get the ``g``-orthonormal basis, wrap it in
    :class:`OrthonormalBasis` (the explicit dual of ``remetric`` — orthonormalization is never
    implicit in a change of basis).
    """

    basis: Basis
    metric: Term

    def __post_init__(self) -> None:
        _require_basis(self.basis, "WithInnerBasis")
        _require_term(self.metric, "WithInnerBasis.metric (must be a Term, not an array)")

    def __repr__(self) -> str:
        return f"WithInnerBasis({self.basis!r}, {self.metric!r})"


@dataclass(frozen=True)
class OrthonormalBasis:
    """The explicit orthonormalization of a basis.

    An **explicit** basis node (the dual of ``remetric``): orthonormalizes ``b`` w.r.t. its
    metric when ``b`` is a :class:`WithInnerBasis` (g-orthonormal), else w.r.t. the standard
    inner product.  Makes orthonormalization a written step, never an implicit cob.
    """

    basis: Basis

    def __post_init__(self) -> None:
        _require_basis(self.basis, "OrthonormalBasis.basis")

    def __repr__(self) -> str:
        return f"OrthonormalBasis({self.basis!r})"


@dataclass(frozen=True)
class CompleteBasis:
    """The completion of a subspace basis to a full basis of its ambient space.

    Completes a basis of an EXPLICIT subspace to its ambient, by filling the missing directions from
    ``complement``.

    ``complement`` is a :class:`Basis` (e.g. the ambient standard basis) — the source of the
    completing directions.  ``reduce`` (:class:`Reduce`) selects how ``[sub | complement]`` is
    combined: ``RREF`` keeps the sub columns then greedily adds independent complement columns;
    ``ORTHONORMALIZE`` orthonormalizes the stack; ``NONE`` stacks as-is.  embed = pad-zeros,
    project = drop-complement (realized FROM this; no Embed/ProjectBasis).

    ``sub_basis`` is intended to be a basis of a **genuine subspace** of the ambient (e.g. a
    :class:`ProjectedBasis` of ``Img(f)``), realized via its AMBIENT immersion when completed.
    To make a full ambient basis from an explicit short term frame, prefer a
    :class:`ListBasis`-with-backup (``ListBasis(*vecs, backup=…, reduce=RREF)``) — a
    ``ListBasis`` is always a full basis of its ambient — rather than wrapping the frame here.
    """

    sub_basis: Basis
    complement: Basis
    reduce: Reduce = Reduce.NONE

    def __post_init__(self) -> None:
        # NB: ``sub_basis`` should be a genuine-subspace basis (not a bare short ``ListBasis``,
        # which is always full-ambient); not hard-rejected to keep the node flexible.
        _require_basis(self.sub_basis, "CompleteBasis.sub_basis")
        _require_basis(self.complement, "CompleteBasis.complement")
        if not isinstance(self.reduce, Reduce):
            raise BasisError(f"CompleteBasis.reduce must be a Reduce, got {self.reduce!r}")

    def __repr__(self) -> str:
        extra = f", reduce={self.reduce}" if self.reduce is not Reduce.NONE else ""
        return f"CompleteBasis({self.sub_basis!r}, {self.complement!r}{extra})"


@dataclass(frozen=True)
class BLam:
    """The basis-level binder: the machinery that builds a basis, abstracted over a value, basis, or nat.

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
        if self.term_var is not None and not isinstance(self.term_var, Var):
            raise BasisError(f"BLam.term_var: expected a Var or None, got {self.term_var!r}")
        if self.basis_var is not None and not isinstance(self.basis_var, BVar):
            raise BasisError(f"BLam.basis_var: expected a BVar or None, got {self.basis_var!r}")

    def __repr__(self) -> str:
        binders = _tuple_repr(self.binders)
        extra = ""
        if self.term_var is not None or self.basis_var is not None:
            extra = f", {self.term_var!r}, {self.basis_var!r}"
        return f"BLam({binders}, {self.body!r}{extra})"


@dataclass(frozen=True)
class BApp:
    """The basis produced by applying a basis function (a ``BLam``) to arguments.

    Mirrors ``App``/``Eval`` at the basis level.
    """

    f: Basis
    positions: tuple[int, ...]
    args: tuple[Term | Space | Dim | Basis, ...]

    def __post_init__(self) -> None:
        from .space import is_dim, is_space
        _require_basis(self.f, "BApp.f")
        if len(self.positions) != len(self.args):
            raise BasisError(
                f"BApp: positions ({len(self.positions)}) and args "
                f"({len(self.args)}) must have equal length"
            )
        for p in self.positions:
            if not isinstance(p, int) or p < 0:
                raise BasisError(f"BApp: positions must be non-negative ints, got {p!r}")
        for a in self.args:
            if not (is_term(a) or is_basis(a) or is_dim(a) or is_space(a)):
                raise BasisError(
                    f"BApp.arg: expected a Term/Basis/Dim/Space, got {a!r}"
                )

    def __repr__(self) -> str:
        return f"BApp({self.f!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.args)})"


Basis = ScalarBasis | BVar | DualBasis | PrimalBasis | BasisSum | BasisProd | BasisRep | ProjectedBasis | ImageBasis | ListBasis | WithInnerBasis | OrthonormalBasis | CompleteBasis | BLam | BApp

_BASIS_TYPES = (
    ScalarBasis,
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
    OrthonormalBasis,
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


def _require_term(t: object, ctx: str) -> None:
    if not is_term(t):
        raise BasisError(f"{ctx}: expected a Term, got {t!r}")


# ---------------------------------------------------------------------------
# Smart constructors.
# ---------------------------------------------------------------------------


def scalar_basis() -> ScalarBasis:
    """Give the standard basis of ``Scalar`` (ℝ)."""
    return ScalarBasis()


def bvar(name: str) -> BVar:
    """Build a basis variable named ``name``."""
    return BVar(name)


def dual_basis(b: Basis) -> DualBasis:
    """Build the dual (cobasis) of ``b``."""
    return DualBasis(b)


def primal_basis(b: Basis) -> PrimalBasis:
    """Build the primal basis underlying the cobasis ``b``."""
    return PrimalBasis(b)


def basis_sum(*bases: Basis) -> BasisSum:
    """Build the direct-sum basis of ``bases`` (the basis of ``⊕``)."""
    return BasisSum(tuple(bases))


def basis_prod(*bases: Basis) -> BasisProd:
    """Build the tensor-product basis of ``bases`` (the basis of ``⊗``)."""
    return BasisProd(tuple(bases))


def basis_rep(count: object, b: Basis) -> BasisRep:
    """Build the ``count``-fold replicated basis ``⊕^count b``."""
    return BasisRep(as_dim(count), b)


def projected_basis(
    f: Term, which: str, source: Basis | None = None, target: Basis | None = None
) -> ProjectedBasis:
    """Build the orthonormalized subspace basis of ``f`` selected by ``which``."""
    return ProjectedBasis(f, which, source, target)


def image_basis(
    f: Term, source: Basis | None = None, target: Basis | None = None
) -> ImageBasis:
    """Build the literal pushforward basis ``f(sᵢ)`` of ``Img(f)`` (not orthonormalized)."""
    return ImageBasis(f, source, target)


def list_basis(*vectors: Term, backup: Basis | None = None,
               reduce: Reduce = Reduce.NONE) -> ListBasis:
    """Build a basis from explicit ``vectors``."""
    return ListBasis(tuple(vectors), backup, reduce)


def with_inner_basis(b: Basis, metric: Term) -> WithInnerBasis:
    """Attach ``metric`` to ``b`` without orthonormalizing it."""
    return WithInnerBasis(b, metric)


def orthonormal_basis(b: Basis) -> OrthonormalBasis:
    """Orthonormalize ``b`` explicitly."""
    return OrthonormalBasis(b)


def complete_basis(sub_basis: Basis, complement: Basis,
                   reduce: Reduce = Reduce.NONE) -> CompleteBasis:
    """Complete ``sub_basis`` to a full basis with ``complement``."""
    return CompleteBasis(sub_basis, complement, reduce)


def blam(
    binders: tuple[tuple[str, Kind], ...],
    body: Basis,
    term_var: Var | None = None,
    basis_var: BVar | None = None,
) -> BLam:
    """Build a basis-level lambda binding ``binders`` over ``body``."""
    return BLam(tuple(binders), body, term_var, basis_var)


def bapp(f: Basis, positions: tuple[int, ...], *args: Term | Space | Dim | Basis) -> BApp:
    """Apply the basis function ``f`` to ``args`` at ``positions``."""
    return BApp(f, tuple(positions), tuple(args))


__all__ = [
    "BasisError",
    "PROJECTED_WHICH",
    "Reduce",
    "Basis",
    "ScalarBasis",
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
    "OrthonormalBasis",
    "CompleteBasis",
    "BLam",
    "BApp",
    "is_basis",
    "scalar_basis",
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
    "orthonormal_basis",
    "complete_basis",
    "blam",
    "bapp",
]
