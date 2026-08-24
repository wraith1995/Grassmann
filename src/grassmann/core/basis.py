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
the transported (pushforward) frame along ``f`` — and does not orthonormalize them.
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
    """How a listed or completed basis combines its columns into a basis.

    ``NONE`` takes the columns as given, which assumes they are linearly independent and
    already span. ``RREF`` pivots greedily, keeping the linearly independent columns and
    completing from a backup basis when they fall short. ``ORTHONORMALIZE`` runs a QR or
    Gram–Schmidt over them, which is the only way orthonormalization ever happens: it is
    never implicit in a change of basis.
    """

    NONE = "none"
    RREF = "rref"
    ORTHONORMALIZE = "orthonormalize"


class TargetBasis(Enum):
    """A policy naming the basis :func:`grassmann.lower.represent` should lower a node into.

    It is the ``target`` argument of ``represent`` and of the ``_rep`` arms beneath it.
    ``AUTO``, the default, leaves the choice to the system: each space keeps its natural
    basis, meaning a pin from the ``basis_env`` where there is one and otherwise the
    least symbolic candidate that ``choose`` scores highest. ``STD`` asks for the
    standard orthonormal basis of each space, ``default_basis(space)``, and propagates
    that request through the whole recursion so every node is expressed in it.

    The ``target`` also accepts a concrete :class:`Basis`, which is a different thing
    from this policy: it names the basis of the result directly.
    """

    AUTO = "auto"
    STD = "std"


#: Free choice by the system (natural / min-symbolic basis) — the ``represent`` default.
AUTO = TargetBasis.AUTO
#: Standard orthonormal basis of each space, propagated through the recursion.
STD = TargetBasis.STD


class BasisError(Exception):
    """Raised when a Basis is structurally malformed."""


# The `which` selectors a ProjectedBasis accepts, one per fundamental subspace.
#
# "image" is the orthonormal range of f: the Q factor of QrOp when f is asserted
# injective, so the rank is static and equals |dom|, and U[:, :δ_f] of SvdOp otherwise.
# It is the orthonormalized counterpart of ImageBasis, whose columns are the raw map
# applications; a change of basis relates the two, built from QrOp's R and Q in the
# injective case and from the SVD factors in general.
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
    """The orthonormalized basis of one of ``f``'s four fundamental subspaces.

    ``which`` selects the subspace, one of the strings in :data:`PROJECTED_WHICH`, and
    ``source`` and ``target`` are the bases of ``f``'s domain and codomain, each
    defaulting to the standard basis when left as ``None``.

    For ``which == "image"`` the columns are the orthonormal range of ``f``: the ``Q``
    factor of ``QrOp`` when ``f`` is injective and the rank is therefore static, and the
    leading ``δ_f`` columns of the SVD's ``U`` in general. That is the orthonormalized
    counterpart of :class:`ImageBasis`, which keeps the raw pushforward columns instead;
    a change of basis built from ``QrOp``'s ``R`` and ``Q``, or from the SVD factors,
    carries one to the other.
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
    """The basis of literal pushforward columns ``f(sᵢ)`` of a map's image.

    Given a map ``f`` and a basis ``s`` of its domain, this coordinatizes ``Img(f)`` by
    the vectors ``f(s₁), …, f(sₙ)`` themselves, read in the codomain basis ``target`` —
    the transported (pushforward) frame along ``f``. ``source`` defaults to the standard basis
    of ``dom f`` when left as ``None``, giving the literal columns ``f(eᵢ)``, and
    ``target`` likewise defaults to the standard basis of ``cod f``.

    It is not orthonormalized. When an orthonormal basis for the image is wanted,
    :class:`ProjectedBasis` with ``which == "image"`` supplies it instead — the ``Q`` of
    the QR factorization, or the leading ``δ_f`` columns of the SVD's ``U`` — and a
    change of basis built from those same factors carries one basis to the other.
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
    """A basis given as an explicit list of term vectors, always full for its ambient space.

    Each entry is a :class:`~grassmann.core.term.Term` vector, and all of them share one
    space, the ambient. At lowering the vectors are stacked into the basis matrix, and a
    short basis — fewer vectors than the ambient dimension — is completed from
    ``backup``, so the node always realizes to a full basis of the ambient rather than to
    a basis of the span of its vectors.

    ``reduce`` says how the columns are combined: ``NONE`` takes them as independent,
    ``RREF`` keeps the independent ones and then completes, and ``ORTHONORMALIZE`` runs a
    QR and completes from the orthonormalized backup. ``backup`` is the basis supplying
    the completing directions, the ambient's standard basis when left as ``None``. A
    short ℝⁿ basis therefore realizes as an ``n × n`` ambient basis, the explicit vectors
    followed by a standard completion, while a full-rank basis of ``n`` vectors is used
    unchanged.

    A basis of a genuine subspace is a different thing: build it directly, for instance
    as a :class:`ProjectedBasis`, and lift it to the ambient with :class:`CompleteBasis`.
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
    """The basis ``b`` read in the metricized space ``WithInner(V, g)``, carrying ``g`` raw.

    The metric rides along as metadata and nothing is done to the columns: they are
    exactly ``b``'s, now living in the metricized space, and they are not assumed
    ``g``-orthonormal. Wrapping the result in :class:`OrthonormalBasis` is what produces
    the ``g``-orthonormal basis, and that step is always written out, never folded into a
    change of basis.
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
    """The orthonormalization of a basis, written as a node of its own.

    The inner product it orthonormalizes against is the one ``b`` carries: the metric
    ``g`` when ``b`` is a :class:`WithInnerBasis`, giving a ``g``-orthonormal basis, and
    the standard inner product otherwise. Having the node is what keeps
    orthonormalization a step in the term rather than something a change of basis does on
    its own, mirroring the way :class:`~grassmann.core.term.Remetric` spells out a change
    of metric.
    """

    basis: Basis

    def __post_init__(self) -> None:
        _require_basis(self.basis, "OrthonormalBasis.basis")

    def __repr__(self) -> str:
        return f"OrthonormalBasis({self.basis!r})"


@dataclass(frozen=True)
class CompleteBasis:
    """The completion of a subspace basis to a full basis of its ambient space.

    The directions ``sub_basis`` does not cover are filled in from ``complement``, which
    is any basis of the ambient and is usually its standard one. ``reduce`` says how the
    stack ``[ sub | complement ]`` is combined: ``RREF`` keeps the sub columns and then
    greedily adds the independent complement columns, ``ORTHONORMALIZE`` orthonormalizes
    the whole stack, and ``NONE`` stacks them as they are.

    The subspace moves are realized from this node, so there is no separate embedding or
    projecting basis: an :class:`~grassmann.core.term.Embed` pads the missing coordinates
    with zeros and a :class:`~grassmann.core.term.Project` drops the complement block.

    ``sub_basis`` is meant to be a basis of a genuine subspace of the ambient, such as a
    :class:`ProjectedBasis` of ``Img(f)``, and completion realizes it through its ambient
    immersion. An explicit short basis of term vectors is a different case: a
    :class:`ListBasis` is already full for its ambient, so give it a backup —
    ``ListBasis(*vecs, backup=…, reduce=RREF)`` — rather than wrapping it here.
    """

    sub_basis: Basis
    complement: Basis
    reduce: Reduce = Reduce.NONE

    def __post_init__(self) -> None:
        # `sub_basis` is meant to be a genuine-subspace basis; a bare short `ListBasis`
        # is already full for its ambient, so completing one is a mistake. It is not
        # hard-rejected here, to keep the node usable for the cases that do make sense.
        _require_basis(self.sub_basis, "CompleteBasis.sub_basis")
        _require_basis(self.complement, "CompleteBasis.complement")
        if not isinstance(self.reduce, Reduce):
            raise BasisError(f"CompleteBasis.reduce must be a Reduce, got {self.reduce!r}")

    def __repr__(self) -> str:
        extra = f", reduce={self.reduce}" if self.reduce is not Reduce.NONE else ""
        return f"CompleteBasis({self.sub_basis!r}, {self.complement!r}{extra})"


@dataclass(frozen=True)
class BLam:
    """The basis-level binder, abstracting the construction of a basis over an argument.

    The binder's domain ``σ``, a :class:`~grassmann.core.kinds.Kind`, decides what kind
    of thing this is: a value domain, spelled as a ``SpaceSort``, makes it a frame — a
    basis that varies with a runtime value; a basis domain makes it a basis function; and
    a nat domain makes it a builder parametric in a dimension. ``body`` is the
    :class:`Basis` produced, and ``term_var`` and ``basis_var`` hold the bound handle the
    body refers to, whichever of the two applies.
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
    """The basis produced by applying a basis function to its arguments.

    ``f`` is a :class:`BLam`, ``positions`` names the binder slots being filled, and
    ``args`` supplies them — a term, a space, a nat, or another basis, matching whatever
    the binder's domain is. It is the basis-level counterpart of
    :class:`~grassmann.core.space.App` and :class:`~grassmann.core.term.Eval`.
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
    """Build a basis from the explicit ``vectors``, completed from ``backup`` if short."""
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
