"""The Term sum type: maps between spaces, and the two lambdas.

A :class:`Term` is a linear or nonlinear map. The family is closed and n-ary,
over real spaces, and it carries the operations one expects of a coordinate-free
algebra: application and composition, the adjoint :class:`Dagger`,
:class:`Solve` and :class:`Pinv`, the musical isomorphisms :class:`Flat` and
:class:`Sharp`, projections and embeddings between a space and its subspaces, the
:class:`Trace`, and foreign values. Two binders sit above them:

``LLam`` is the linear lambda, the map arrow ``⊸``. Its result space is
independent of the linear arguments, so it is itself representable as a value in
``⊗(sᵢ*) ⊗ s_e``. Linearity forbids dependence: a linear argument may not appear
in a binder's domain annotation nor as an explicit space or nat argument, since
either would make the result depend on it. That independence is the structural
rule enforced here.

``Lam`` is the one general nonlinear binder ``Lam(b:σ. body)``. Its body may be
a term, a space, a type, or a nat, and its domain ``b:σ`` is a value (an element
of a space ``s``), a space, or a nat. A ``Lam`` whose body is a space or a type
is a space function or a type function.

One index convention runs through the family: an integer index list selects the
parameter slots to feed (``Eval``, ``NEval``, ``Compose``), the tensor indices to
fuse (``Lift``, ``InvLift``), or the indices to contract (``Trace``).

Well-formedness enforced here is structural only — arity, child kinds, and the
linearity guard on ``LLam``. Checking that contracted slots are dual and that
summands share a space is type checking, done in :mod:`grassmann.infer`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .kinds import Kind, SpaceSort, is_kind
from .props import MapProps
from .space import Dim, Space, _tuple_repr, is_space
from .types import Type, is_type

if TYPE_CHECKING:
    # `basis` imports `term` at runtime, so importing it here would be a true
    # cycle; under TYPE_CHECKING it is only used for forward annotations.
    from .basis import Basis


class TermError(Exception):
    """Raised when a Term is structurally malformed."""


_counter = itertools.count()


# ===========================================================================
# Values & algebra.
# ===========================================================================


@dataclass(frozen=True)
class Var:
    """A term variable of a space, optionally carrying asserted map properties.

    ``space`` is the space the variable inhabits, and ``props`` the properties a caller
    asserts of it when it is a map — injectivity, surjectivity, or a rank. A variable is
    the only node that carries them, so a derived map is asserted with
    :class:`Assert` instead. ``uid`` is a fresh identity taking part in equality, so two
    binders that bind the same name in the same space do not capture each other's
    occurrences.
    """

    name: str
    space: Space
    props: MapProps | None = None
    uid: int = field(default_factory=lambda: next(_counter), compare=True)

    def __post_init__(self) -> None:
        if not is_space(self.space):
            raise TermError(f"Var {self.name!r}: space must be a Space, got {self.space!r}")
        if self.props is not None and not isinstance(self.props, MapProps):
            raise TermError(f"Var {self.name!r}: props must be a MapProps, got {self.props!r}")

    def __repr__(self) -> str:
        p = f", props={self.props!r}" if self.props is not None else ""
        # uid is part of identity (binder vars), so include it for repr round-trip.
        return f"Var({self.name!r}, {self.space!r}{p}, uid={self.uid})"


@dataclass(frozen=True)
class External:
    """A foreign function or value — opaque ``data`` together with the argument terms it depends on."""

    space: Space
    data: object
    args: tuple[Term, ...] = ()

    def __post_init__(self) -> None:
        if not is_space(self.space):
            raise TermError(f"External: space must be a Space, got {self.space!r}")
        for a in self.args:
            _require_term(a, "External.args")

    def __repr__(self) -> str:
        return f"External({self.space!r}, {self.data!r}, {_tuple_repr(self.args)})"


@dataclass(frozen=True)
class Zero:
    """The zero vector of a space.

    ``space`` may be given, or left as ``None`` for inference to recover: the node then
    synthesizes a fresh ``SpaceMeta`` that first-order unification solves from the
    surrounding context, such as a sibling summand of a ``Sum`` or a check against the
    space an enclosing :class:`Assert` expects.
    """

    space: Space | None = None

    def __post_init__(self) -> None:
        _require_space_or_none(self.space, "Zero")

    def __repr__(self) -> str:
        return f"Zero({self.space!r})"


@dataclass(frozen=True)
class Identity:
    """The identity map on a space ``s`` (a value in ``~s ⊗ s``)."""

    space: Space

    def __post_init__(self) -> None:
        _require_space_field(self.space, "Identity")

    def __repr__(self) -> str:
        return f"Identity({self.space!r})"


@dataclass(frozen=True)
class Lit:
    """A real scalar literal at the term level."""

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)):
            raise TermError(f"Lit: value must be a real number, got {self.value!r}")

    def __repr__(self) -> str:
        return f"Lit({self.value!r})"


@dataclass(frozen=True)
class Scale:
    """The scalar multiple of a vector by a real number or scalar-valued term.

    ``scalar`` is a real or a scalar-valued ``Term``; ``vector`` is the term it scales.
    """

    scalar: float | Term
    vector: Term

    def __post_init__(self) -> None:
        if not isinstance(self.scalar, (int, float)) and not is_term(self.scalar):
            raise TermError(f"Scale: scalar must be a number or Term, got {self.scalar!r}")
        _require_term(self.vector, "Scale.vector")

    def __repr__(self) -> str:
        return f"Scale({self.scalar!r}, {self.vector!r})"


@dataclass(frozen=True)
class Sum:
    """The n-ary vector addition of terms that share a space."""

    vectors: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not self.vectors:
            raise TermError("Sum needs at least one summand")
        for v in self.vectors:
            _require_term(v, "Sum")

    def __repr__(self) -> str:
        return f"Sum({_tuple_repr(self.vectors)})"


@dataclass(frozen=True)
class OPlusV:
    """The n-ary direct-sum stack of vectors (a value in ``⊕(sᵢ)``)."""

    vectors: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not self.vectors:
            raise TermError("OPlus needs at least one summand")
        for v in self.vectors:
            _require_term(v, "OPlus")

    def __repr__(self) -> str:
        return f"OPlusV({_tuple_repr(self.vectors)})"


@dataclass(frozen=True)
class OProdV:
    """The n-ary tensor product of vectors (a value in ``⊗(sᵢ)``)."""

    vectors: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not self.vectors:
            raise TermError("OProd needs at least one factor")
        for v in self.vectors:
            _require_term(v, "OProd")

    def __repr__(self) -> str:
        return f"OProdV({_tuple_repr(self.vectors)})"


# ===========================================================================
# Application / composition / trace.
# ===========================================================================


def _check_positions_args(ctx: str, positions: tuple[int, ...], args: tuple[Term, ...]) -> None:
    if len(positions) != len(args):
        raise TermError(
            f"{ctx}: positions ({len(positions)}) and args ({len(args)}) "
            f"must have equal length"
        )
    for p in positions:
        if not isinstance(p, int) or p < 0:
            raise TermError(f"{ctx}: positions must be non-negative ints, got {p!r}")
    if len(set(positions)) != len(positions):
        raise TermError(f"{ctx}: positions must be distinct, got {list(positions)!r}")
    for a in args:
        _require_term(a, f"{ctx}.args")


@dataclass(frozen=True)
class Eval:
    """The linear application of a map to arguments, read as a contraction.

    ``f`` is contracted with each of ``args`` at the dual slot named by the matching
    entry of ``positions``, so ``Eval(f, (0,), (x,))`` is the ordinary ``f(x)``:
    the leading dual slot of ``f`` pairs with ``x``. Feeding several slots at once is a
    single node rather than a chain of applications.
    """

    f: Term
    positions: tuple[int, ...]
    args: tuple[Term, ...]

    def __post_init__(self) -> None:
        _require_term(self.f, "Eval.f")
        _check_positions_args("Eval", self.positions, self.args)

    def __repr__(self) -> str:
        return f"Eval({self.f!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.args)})"


@dataclass(frozen=True)
class NEval:
    """The non-linear application of a :class:`Lam` to term, space, or nat arguments.

    ``positions`` names the binder slots being filled and ``args`` supplies them. An
    argument may be a term, and it may equally be a space or a nat, which is how a space
    function or a nat-indexed family is instantiated.
    """

    f: Term
    positions: tuple[int, ...]
    args: tuple[Term | Space | Dim, ...]

    def __post_init__(self) -> None:
        _require_term(self.f, "NEval.f")
        if len(self.positions) != len(self.args):
            raise TermError(
                f"NEval: positions ({len(self.positions)}) and args "
                f"({len(self.args)}) must have equal length"
            )
        for p in self.positions:
            if not isinstance(p, int) or p < 0:
                raise TermError(f"NEval: positions must be non-negative ints, got {p!r}")
        # args may be Terms, Spaces or Nats (explicit space/nat application).
        from .space import is_dim
        for a in self.args:
            if not (is_term(a) or is_space(a) or is_dim(a)):
                raise TermError(f"NEval.arg: expected a Term/Space/Dim, got {a!r}")

    def __repr__(self) -> str:
        return f"NEval({self.f!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.args)})"


@dataclass(frozen=True)
class Compose:
    """Composition of maps: contract each ``gₖ``'s codomain into ``f``'s selected slot."""

    f: Term
    positions: tuple[int, ...]
    gs: tuple[Term, ...]

    def __post_init__(self) -> None:
        _require_term(self.f, "Compose.f")
        _check_positions_args("Compose", self.positions, self.gs)

    def __repr__(self) -> str:
        return f"Compose({self.f!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.gs)})"


@dataclass(frozen=True)
class Trace:
    """The contraction of a map's primal index ``i`` with its dual index ``j``."""

    f: Term
    i: int
    j: int

    def __post_init__(self) -> None:
        _require_term(self.f, "Trace.f")
        if not isinstance(self.i, int) or not isinstance(self.j, int) or self.i < 0 or self.j < 0:
            raise TermError(f"Trace: i, j must be non-negative ints, got {self.i!r}, {self.j!r}")
        if self.i == self.j:
            raise TermError(f"Trace: cannot contract an index with itself ({self.i})")

    def __repr__(self) -> str:
        return f"Trace({self.f!r}, {self.i}, {self.j})"


@dataclass(frozen=True)
class Det:
    """The determinant of a square endomorphism ``f : V ⊸ V`` (a scalar)."""

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Det.f")

    def __repr__(self) -> str:
        return f"Det({self.f!r})"


@dataclass(frozen=True)
class LogDet:
    """The log-determinant ``log det f`` of a square endomorphism ``f : V ⊸ V``, a scalar.

    It is a node of its own rather than a ``Log`` around a :class:`Det` because that is
    what makes its derivative clean: :mod:`grassmann.transform.diff` differentiates it to
    ``tr(f⁻¹·df)``, with no determinant factor left to carry around.
    """

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "LogDet.f")

    def __repr__(self) -> str:
        return f"LogDet({self.f!r})"


@dataclass(frozen=True)
class OpaqueOp:
    """An opaque map ``f : dom → cod``, declared by its endpoints and given no body.

    ``fn`` optionally supplies the numpy kernel that runs it, a callable from a ``dom``
    array to a ``cod`` array. The kernel is excluded from equality and hashing, so two
    ops are equal exactly when their ``name``, ``dom``, and ``cod`` agree. The op is the
    function symbol itself and not a term; :class:`OpaqueApply` is the term that applies
    it.
    """

    name: str
    dom: Space
    cod: Space
    fn: object = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not is_space(self.dom):
            raise TermError(f"OpaqueOp.dom: expected a Space, got {self.dom!r}")
        if not is_space(self.cod):
            raise TermError(f"OpaqueOp.cod: expected a Space, got {self.cod!r}")

    def __repr__(self) -> str:
        return f"OpaqueOp({self.name!r})"


@dataclass(frozen=True)
class OpaqueApply:
    """The application of an opaque map ``op`` to ``arg``, a term in ``op.dom``.

    The result is a term in ``op.cod``. Since the op has no body to differentiate,
    :mod:`grassmann.transform.diff` differentiates the application against a Jacobian
    tracer input it allocates for the purpose.
    """

    op: OpaqueOp
    arg: Term

    def __post_init__(self) -> None:
        if not isinstance(self.op, OpaqueOp):
            raise TermError(f"OpaqueApply.op: expected an OpaqueOp, got {self.op!r}")
        _require_term(self.arg, "OpaqueApply.arg")

    def __repr__(self) -> str:
        return f"OpaqueApply({self.op.name!r}, {self.arg!r})"


# ===========================================================================
# Inverses & adjoint.
# ===========================================================================


@dataclass(frozen=True)
class Solve:
    """The two-sided inverse ``f⁻¹`` of a map asserted bijective.

    Bijectivity is a precondition, not a discovery: ``f`` must carry ``props=bij``, which
    type inference checks and refuses otherwise. For a map that is not invertible the
    least-squares inverse :class:`Pinv` is the node to reach for.
    """

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Solve.f")

    def __repr__(self) -> str:
        return f"Solve({self.f!r})"


@dataclass(frozen=True)
class Pinv:
    """The Moore–Penrose pseudo-inverse ``f⁺``, the least-squares minimum-norm inverse.

    It is defined for every map. What it computes sharpens with the properties asserted
    of ``f``: an exact left inverse when ``f`` is injective, an exact right inverse when
    it is surjective, and the true inverse ``f⁻¹`` when it is bijective.
    """

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Pinv.f")

    def __repr__(self) -> str:
        return f"Pinv({self.f!r})"


@dataclass(frozen=True)
class Corestrict:
    """The corestriction ``f~ : dom(f) ⊸ Img(f)`` of an injective map onto its own image.

    It is the isomorphism the first isomorphism theorem induces, in the ``ker f = 0``
    form, and it makes the factorization of ``f`` through its image explicit::

        dom(f) ───── f~ ─────▶ Img(f) ───── ι ─────▶ cod(f)
          │          ≅                  inclusion      ▲
          └──────────────────── f ─────────────────────┘

    Because ``f~`` is an isomorphism, :class:`Pinv` of it is exact: ``Pinv(f~)`` is the
    two-sided inverse ``Img(f) ⊸ dom(f)``, and the two compose to the identity on either
    side. That is the written form of the identification ``dom(f) ≅ Img(f)``, which the
    calculus has no other way to say, since there are no implicit conversions between
    spaces.

    Injectivity is required: ``f`` must be asserted ``inj``, which
    :mod:`grassmann.infer.synth` checks and refuses loudly otherwise. For a general map
    the domain-side isomorphism of the theorem runs out of ``CoImg(f)`` rather than
    ``dom(f)``, which is a different map with a different domain; this node does not
    cover it.

    It has to be a node rather than a derived form because :class:`Project` and
    :class:`Embed` act on a value, so nothing else in the calculus produces a map whose
    endpoint is ``Img(f)``. The composite ``Pinv(f_B) ∘ f_A`` is not a substitute: it
    synthesizes as ``dom(f_A) ⊸ dom(f_B)``, dropping the ``Img`` identification along
    with the ``R``-factor scaling and the orientation sign carried by the image frames,
    and every consumer that checks a map endpoint strictly — ``Compose``, and through it
    the contravariant ``Dual`` arm of
    :func:`grassmann.transform.pullback.pullback` — rejects it.
    """

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Corestrict.f")

    def __repr__(self) -> str:
        return f"Corestrict({self.f!r})"


@dataclass(frozen=True)
class SolveApply:
    """The solution ``x`` of the system ``f·x = v`` — the applied form of ``Solve``."""

    f: Term
    v: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "SolveApply.f")
        _require_term(self.v, "SolveApply.v")

    def __repr__(self) -> str:
        return f"SolveApply({self.f!r}, {self.v!r})"


@dataclass(frozen=True)
class PinvApply:
    """The pseudo-inverse of a map applied to a vector, ``f†·v``."""

    f: Term
    v: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "PinvApply.f")
        _require_term(self.v, "PinvApply.v")

    def __repr__(self) -> str:
        return f"PinvApply({self.f!r}, {self.v!r})"


@dataclass(frozen=True)
class Dagger:
    """The adjoint ``f†`` of a map, the map satisfying ``⟨f x, y⟩ = ⟨x, f† y⟩``.

    The identity is read against the inner products of the domain and codomain, so the
    adjoint needs both of them and is not the bare index swap. Over real spaces it is
    the metric transpose, with no conjugation, and which transpose that comes out as is
    resolved during lowering from the two metrics: with standard metrics on both sides
    it is the plain transpose, and otherwise it picks up their Gram factors.
    """

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Dagger.f")

    def __repr__(self) -> str:
        return f"Dagger({self.f!r})"


@dataclass(frozen=True)
class Sqrt:
    """The operator square root of an endomorphism ``f : V ⊸ V``: the unique SPD ``S`` with ``S·S = f``.

    It lowers through an eigendecomposition, and it fails loudly rather than silently
    when ``f`` is not symmetric positive definite. It is the orthonormalization factor
    the metric itself supplies: for an immersed metric ``G = JᵀJ``, ``Sqrt(G)`` is the
    stretch and its inverse the whitening, so ``J·Sqrt(G)⁻¹`` is the polar orthonormal
    frame.
    """

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Sqrt.f")

    def __repr__(self) -> str:
        return f"Sqrt({self.f!r})"


@dataclass(frozen=True)
class Inner:
    """The inner-product map of a space, the bilinear form ``s ⊗ s ⊸ Scalar``.

    It is read from the metric that ``s`` itself carries, which is the standard one
    unless ``s`` is a :class:`~grassmann.core.space.WithInner` naming another.
    """

    space: Space

    def __post_init__(self) -> None:
        _require_space_field(self.space, "Inner")

    def __repr__(self) -> str:
        return f"Inner({self.space!r})"


@dataclass(frozen=True)
class Flat:
    """The musical isomorphism ♭, lowering an index: the metric map ``s ⊸ s*``.

    It sends a vector ``v`` to the covector ``⟨v, ·⟩`` that pairs with it, so it reads
    the inner product of ``s``. Its inverse is :class:`Sharp`.
    """

    space: Space

    def __post_init__(self) -> None:
        _require_space_field(self.space, "Flat")

    def __repr__(self) -> str:
        return f"Flat({self.space!r})"


@dataclass(frozen=True)
class Sharp:
    """The musical isomorphism ♯, raising an index: the inverse metric map ``s* ⊸ s``.

    It sends a covector back to the vector representing it, inverting :class:`Flat`, so
    it too reads the inner product of ``s``.
    """

    space: Space

    def __post_init__(self) -> None:
        _require_space_field(self.space, "Sharp")

    def __repr__(self) -> str:
        return f"Sharp({self.space!r})"


# ===========================================================================
# Movement, coercion, reshape.
# ===========================================================================


@dataclass(frozen=True)
class Project:
    """The projection of ``v`` onto a subspace ``s`` (``s ⊑ space(v)``).

    ``space`` names the target subspace, and may be left as ``None`` for inference to
    recover. A space has many subspaces, so the omitted target is recoverable only when
    an expected space flows in from the surrounding context along the check direction;
    that unifies the fresh ``SpaceMeta``, and the deferred obligation ``s ⊑ space(v)`` is
    validated afterwards.
    """

    vector: Term
    space: Space | None = None

    def __post_init__(self) -> None:
        _require_term(self.vector, "Project.vector")
        _require_space_or_none(self.space, "Project")

    def __repr__(self) -> str:
        return f"Project({self.vector!r}, {self.space!r})"


@dataclass(frozen=True)
class Embed:
    """The embedding of ``v`` into a larger space ``s`` (``space(v) ⊑ s``)."""

    vector: Term
    space: Space

    def __post_init__(self) -> None:
        _require_term(self.vector, "Embed.vector")
        _require_space_field(self.space, "Embed")

    def __repr__(self) -> str:
        return f"Embed({self.vector!r}, {self.space!r})"


@dataclass(frozen=True)
class Assert:
    """The claim that ``v`` has space ``s``, checked as ``space(v) ~ s`` during inference.

    It may also carry map ``props`` — ``inj``, ``surj``, ``bij``, or a rank — which ride
    with the value the way a :class:`Var`'s properties do. That is how a derived map,
    which is not a variable and so cannot carry properties itself, gets asserted where a
    rule knows the property holds: differentiating a ``Det`` or a ``LogDet`` asserts its
    operand invertible so that the ``Solve`` in ``tr(f⁻¹·df)`` is licensed.
    """

    vector: Term
    space: Space
    props: MapProps | None = None

    def __post_init__(self) -> None:
        _require_term(self.vector, "Assert.vector")
        _require_space_field(self.space, "Assert")
        if self.props is not None and not isinstance(self.props, MapProps):
            raise TermError(f"Assert: props must be a MapProps, got {self.props!r}")

    def __repr__(self) -> str:
        p = f", props={self.props!r}" if self.props is not None else ""
        return f"Assert({self.vector!r}, {self.space!r}{p})"


@dataclass(frozen=True)
class TupleEmbed:
    """The injection of a value into the ``i``-th ⊕ component of ``s = ⊕(sₖ)`` (zero-padding the rest)."""

    space: Space
    i: int

    def __post_init__(self) -> None:
        _require_space_field(self.space, "TupleEmbed")
        if not isinstance(self.i, int) or self.i < 0:
            raise TermError(f"TupleEmbed: i must be a non-negative int, got {self.i!r}")

    def __repr__(self) -> str:
        return f"TupleEmbed({self.space!r}, {self.i})"


@dataclass(frozen=True)
class TupleProject:
    """The extraction of the ``i``-th ⊕ component of a direct-sum value (a slice)."""

    space: Space
    i: int

    def __post_init__(self) -> None:
        _require_space_field(self.space, "TupleProject")
        if not isinstance(self.i, int) or self.i < 0:
            raise TermError(f"TupleProject: i must be a non-negative int, got {self.i!r}")

    def __repr__(self) -> str:
        return f"TupleProject({self.space!r}, {self.i})"


@dataclass(frozen=True)
class Remetric:
    """The value ``vector`` moved into the metricized space ``WithInner(s, metric)``.

    A metricized space is a different space from the one underneath it, so this crossing
    is always written out; nothing attaches or drops a metric implicitly.
    """

    vector: Term
    metric: Term

    def __post_init__(self) -> None:
        _require_term(self.vector, "Remetric.vector")
        _require_term(self.metric, "Remetric.metric")

    def __repr__(self) -> str:
        return f"Remetric({self.vector!r}, {self.metric!r})"


@dataclass(frozen=True)
class Lift:
    """A reshape fusing the listed tensor indices of a value into a single index.

    The factors named by ``fused`` are curried together, so ``(A ⊗ B) ⊗ C`` read with
    ``fused = (0, 1)`` becomes one index of size ``|A|·|B|`` beside ``C``. It changes the
    layout, never the values, and :class:`InvLift` splits the index back apart.
    """

    vector: Term
    fused: tuple[int, ...]
    k: int | None = None

    def __post_init__(self) -> None:
        _require_term(self.vector, "Lift.vector")
        _check_index_list("Lift", self.fused)
        if self.k is not None and (not isinstance(self.k, int) or self.k < 0):
            raise TermError(f"Lift: k must be a non-negative int or None, got {self.k!r}")

    def __repr__(self) -> str:
        return f"Lift({self.vector!r}, {_tuple_repr(self.fused)}, {self.k!r})"


@dataclass(frozen=True)
class InvLift:
    """The inverse index reshape of :class:`Lift`, splitting a fused index back into its factors (uncurry)."""

    vector: Term
    fused: tuple[int, ...]
    k: int | None = None

    def __post_init__(self) -> None:
        _require_term(self.vector, "InvLift.vector")
        _check_index_list("InvLift", self.fused)
        if self.k is not None and (not isinstance(self.k, int) or self.k < 0):
            raise TermError(f"InvLift: k must be a non-negative int or None, got {self.k!r}")

    def __repr__(self) -> str:
        return f"InvLift({self.vector!r}, {_tuple_repr(self.fused)}, {self.k!r})"


@dataclass(frozen=True)
class Permute:
    """Reorder the tensor factors of ``v``'s space.

    ``perm`` is a permutation: the result's factor ``i`` is ``v``'s factor ``perm[i]``
    (so ``Permute(v, (1, 0))`` transposes a 2-factor tensor / form). The arity must match
    ``v``'s factor count (checked at synthesis). Lowers to an axis transpose — the one
    tensor reordering the reshape-only ``Lift``/``InvLift`` cannot express (e.g. the
    transpose of a ``(0,2)`` form, which ``Dagger`` only provides for proper maps).
    """

    vector: Term
    perm: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_term(self.vector, "Permute.vector")
        _check_index_list("Permute", self.perm)

    def __repr__(self) -> str:
        return f"Permute({self.vector!r}, {_tuple_repr(self.perm)})"


@dataclass(frozen=True)
class TuplePermute:
    """Reorder the ``⊕`` components of ``v`` — the direct-sum analogue of :class:`Permute`, which reorders ``⊗`` factors.

    ``perm`` is a permutation: the result's component ``i`` is ``v``'s component
    ``perm[i]``, so ``v : ⊕(sₖ)`` maps to ``⊕(s_{perm[i]})``. It lowers to a permutation
    of the stack slices, and it is the node that corrects an output into canonical ``⊕``
    order — a reordering :class:`TupleEmbed` and :class:`TupleProject` can express only
    as a heavy sum of injection-after-projection maps.
    """

    vector: Term
    perm: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_term(self.vector, "TuplePermute.vector")
        _check_index_list("TuplePermute", self.perm)

    def __repr__(self) -> str:
        return f"TuplePermute({self.vector!r}, {_tuple_repr(self.perm)})"


@dataclass(frozen=True)
class Kron:
    """The Kronecker product ``f ⊗ g`` of two maps.

    It acts factor-wise on a tensor product, sending ``x ⊗ y`` to ``f(x) ⊗ g(y)``, so its
    domain and codomain are the products of the two maps' domains and codomains.
    """

    f: Term
    g: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Kron.f")
        _require_term(self.g, "Kron.g")

    def __repr__(self) -> str:
        return f"Kron({self.f!r}, {self.g!r})"


# ===========================================================================
# Binders.
# ===========================================================================


@dataclass(frozen=True)
class LLam:
    """The linear lambda ``⊸``, binding one or more linear variables in a body.

    The result space is independent of the linear variables, which is what makes the
    lambda itself a value in ``⊗(sᵢ*) ⊗ s_e``. The guard enforced here is the structural
    half of that rule: a bound variable may not occur inside a :class:`Lam` binder's
    domain annotation, nor as an explicit space or nat argument of an ``NEval`` or an
    ``App``, since either position would let the result space depend on it. Each bound
    variable must also occur somewhere in the body. The full linearity check, which
    counts uses rather than occurrences, runs during type inference.
    """

    vars: tuple[Var, ...]
    body: Term

    def __post_init__(self) -> None:
        if not self.vars:
            raise TermError("LLam needs at least one linear variable")
        for v in self.vars:
            if not isinstance(v, Var):
                raise TermError(f"LLam: bound vars must be Var, got {v!r}")
        _require_term(self.body, "LLam.body")
        _check_llam_linearity(self.vars, self.body)

    def __repr__(self) -> str:
        return f"LLam({_tuple_repr(self.vars)}, {self.body!r})"


@dataclass(frozen=True)
class Lam:
    """The general non-linear binder, abstracting a body over a value, space, or nat.

    Each entry of ``binders`` is a ``(name, sort)`` pair whose sort is a
    :class:`~grassmann.core.kinds.Sort` — ``space``, ``nat``, or ``*`` — or a
    :class:`~grassmann.core.kinds.SpaceSort` for a value domain. The ``body`` is
    polymorphic in the same way: it may be a term, a space, a type, or a nat, and what it
    is decides what the lambda is. A space body makes a space function and a type body a
    type function, each classified by an arrow kind rather than by a Π-type.

    A binder over a value domain introduces a term variable, and the corresponding
    :class:`Var` objects are kept in ``term_vars`` so the body can refer to them.
    """

    binders: tuple[tuple[str, Kind], ...]
    body: Term | Space | Type | Dim
    term_vars: tuple[Var, ...] = ()

    def __post_init__(self) -> None:
        if not self.binders:
            raise TermError("Lam needs at least one binder")
        for name, sort in self.binders:
            if not isinstance(name, str):
                raise TermError(f"Lam: binder name must be a str, got {name!r}")
            if not is_kind(sort):
                raise TermError(f"Lam: binder sort must be a sort/kind, got {sort!r}")
        if not _is_lam_body(self.body):
            raise TermError(
                f"Lam body must be a Term, Space, Type, or Dim; got {self.body!r}"
            )
        for v in self.term_vars:
            if not isinstance(v, Var):
                raise TermError(f"Lam: term_vars must be Var, got {v!r}")

    def __repr__(self) -> str:
        binders = _tuple_repr(self.binders)
        if self.term_vars:
            return f"Lam({binders}, {self.body!r}, {_tuple_repr(self.term_vars)})"
        return f"Lam({binders}, {self.body!r})"


@dataclass(frozen=True)
class Frame:
    """A basis that varies with a runtime value: the binding ``x : s ↦ basis of τ(x)``.

    ``var`` is the bound value variable and ``basis_body`` the basis built from it,
    annotated as a forward reference to ``Basis`` because ``basis`` imports ``term``.
    This is the term-level spelling of a frame; the general basis-level binder and its
    application, ``BLam`` and ``BApp``, live in :mod:`grassmann.core.basis`.
    """

    var: Var
    basis_body: Basis

    def __post_init__(self) -> None:
        if not isinstance(self.var, Var):
            raise TermError(f"Frame: var must be a Var, got {self.var!r}")
        from .basis import is_basis
        if not is_basis(self.basis_body):
            raise TermError(
                f"Frame.basis_body: expected a Basis, got {self.basis_body!r}"
            )

    def __repr__(self) -> str:
        return f"Frame({self.var!r}, {self.basis_body!r})"


# A Term is any of the node types above.
Term = Var | External | Zero | Identity | Lit | Scale | Sum | OPlusV | OProdV | Eval | NEval | Compose | Trace | Det | LogDet | OpaqueApply | Solve | Pinv | Corestrict | Sqrt | SolveApply | PinvApply | Dagger | Inner | Flat | Sharp | Project | Embed | Assert | TupleEmbed | TupleProject | Remetric | Lift | InvLift | Permute | TuplePermute | Kron | LLam | Lam | Frame

_TERM_TYPES = (
    Var,
    External,
    Zero,
    Identity,
    Lit,
    Scale,
    Sum,
    OPlusV,
    OProdV,
    Eval,
    NEval,
    Compose,
    Trace,
    Det,
    LogDet,
    OpaqueApply,
    Solve,
    Pinv,
    Corestrict,
    SolveApply,
    PinvApply,
    Sqrt,
    Dagger,
    Inner,
    Flat,
    Sharp,
    Project,
    Embed,
    Assert,
    TupleEmbed,
    TupleProject,
    Remetric,
    Lift,
    InvLift,
    Permute,
    TuplePermute,
    Kron,
    LLam,
    Lam,
    Frame,
)


def is_term(t: object) -> bool:
    """Return True iff ``t`` is a well-formed Term node."""
    return isinstance(t, _TERM_TYPES)


def _require_term(t: object, ctx: str) -> None:
    if not is_term(t):
        raise TermError(f"{ctx}: expected a Term, got {t!r}")


def _require_space_field(s: object, ctx: str) -> None:
    if not is_space(s):
        raise TermError(f"{ctx}: expected a Space, got {s!r}")


def _require_space_or_none(s: object, ctx: str) -> None:
    """Require ``s`` to be a Space, or ``None`` where the space is left to inference."""
    if s is not None and not is_space(s):
        raise TermError(f"{ctx}: expected a Space or None, got {s!r}")


def _check_index_list(ctx: str, idx: tuple[int, ...]) -> None:
    for p in idx:
        if not isinstance(p, int) or p < 0:
            raise TermError(f"{ctx}: indices must be non-negative ints, got {p!r}")
    if len(set(idx)) != len(idx):
        raise TermError(f"{ctx}: indices must be distinct, got {list(idx)!r}")


def _is_lam_body(body: object) -> bool:
    # Term | Space | Type | Dim. Imported lazily to avoid cycles for Dim.
    from .space import is_dim

    return is_term(body) or is_space(body) or is_type(body) or is_dim(body)


# ---------------------------------------------------------------------------
# LLam linearity guard (structural).
# ---------------------------------------------------------------------------


def _check_llam_linearity(vars: tuple[Var, ...], body: Term) -> None:
    """Reject the structurally detectable ways a linear var makes the result space depend on it.

    Those are: occurring in a ``Lam`` binder domain annotation, or as an explicit space/nat
    argument of an ``NEval``/``App``.

    Also requires each linear var to occur at least once in the body (a vacuous linear
    binder is almost always a mistake; full per-variable linearity is checked during inference).
    """
    var_ids = {v.uid for v in vars}
    seen: set[int] = set()

    def walk(node: object, in_disallowed: bool) -> None:
        # Dispatch over the structurally-significant Term nodes; every other node
        # (the rest of the Term union, plus embedded Spaces) falls through to the
        # generic structural recursion below — this is a PARTIAL dispatch, so the
        # default ``case _`` reproduces that fall-through (no ``assert_never``).
        match node:
            case Var():
                if node.uid in var_ids:
                    seen.add(node.uid)
                    if in_disallowed:
                        raise TermError(
                            f"LLam: linear variable {node.name!r} appears in a "
                            f"dependence position (space/nat arg or binder domain); "
                            f"the result space would depend on it, violating linearity"
                        )
            case Lam():
                # Binder-domain annotations (SpaceSort spaces) are dependence positions.
                for _name, sort in node.binders:
                    if isinstance(sort, SpaceSort):
                        walk(sort.space, True)
                walk(node.body, in_disallowed)
            case NEval():
                walk(node.f, in_disallowed)
                # NEval args may be explicit Space/Nat applications ⇒ dependence positions.
                for a in node.args:
                    if is_space(a) or _is_nat(a):
                        walk(a, True)
                    else:
                        walk(a, in_disallowed)
            case _:  # audit: ok EX-WILDCARD — bail-default (§0): documented generic child recursion [approved 2026-07-03]
                # Generic structural recursion over child Terms/Spaces.
                for child in _children(node):
                    walk(child, in_disallowed)

    walk(body, False)
    missing = var_ids - seen
    if missing:
        names = [v.name for v in vars if v.uid in missing]
        raise TermError(
            f"LLam: linear variable(s) {names!r} do not occur in the body; a linear "
            f"map must use each linear argument"
        )


def _is_nat(x: object) -> bool:
    from .space import is_dim

    return is_dim(x)


def _children(node: object) -> tuple[object, ...]:
    """Yield the child Terms (and any embedded Spaces, walked for var occurrences)."""
    if not is_term(node):
        # Spaces may embed Terms (Ker/Img/.../App.f); walk those for var occurrences.
        if is_space(node):
            return _space_children(node)
        return ()
    out: list[object] = []
    for fname in getattr(node, "__dataclass_fields__", {}):
        val = getattr(node, fname)
        if isinstance(val, tuple):
            out.extend(val)
        else:
            out.append(val)
    # Drop non-walkable scalars (ints, strs, MapProps, None).
    return tuple(c for c in out if is_term(c) or is_space(c))


def _space_children(s: object) -> tuple[object, ...]:
    """Yield the Terms and Spaces a Space node holds, for the occurrence walk."""
    out: list[object] = []
    for fname in getattr(s, "__dataclass_fields__", {}):
        val = getattr(s, fname)
        if isinstance(val, tuple):
            out.extend(val)
        else:
            out.append(val)
    return tuple(c for c in out if is_term(c) or is_space(c))


# ===========================================================================
# Smart constructors (module-level).
# ===========================================================================


def var(name: str, space: Space, props: MapProps | None = None) -> Var:
    """Build a term variable of ``space``, with optional asserted map ``props``."""
    return Var(name, space, props)


def external(space: Space, data: object, *args: Term) -> External:
    """Build a foreign value in ``space`` from opaque ``data`` and its argument dependencies."""
    return External(space, data, tuple(args))


def zero(space: Space | None = None) -> Zero:
    """Build the zero vector of ``space`` (``None`` leaves the space to inference)."""
    return Zero(space)


def identity(space: Space) -> Identity:
    """Build the identity map on ``space``."""
    return Identity(space)


def lit(value: float) -> Lit:
    """Build a real scalar literal."""
    return Lit(value)


def scale(scalar: float | Term, vector: Term) -> Scale:
    """Build the scalar multiple ``scalar · vector``."""
    return Scale(scalar, vector)


def sum_(*vectors: Term) -> Sum:
    """Build the n-ary vector addition of ``vectors``."""
    return Sum(tuple(vectors))


def oplus_v(*vectors: Term) -> OPlusV:
    """Build the direct-sum stack of ``vectors``."""
    return OPlusV(tuple(vectors))


def oprod_v(*vectors: Term) -> OProdV:
    """Build the tensor product of ``vectors``."""
    return OProdV(tuple(vectors))


def eval_(f: Term, positions: tuple[int, ...], *args: Term) -> Eval:
    """Build the linear application of ``f`` to ``args`` at ``positions``."""
    return Eval(f, tuple(positions), tuple(args))


def neval(f: Term, positions: tuple[int, ...], *args: Term | Space | Dim) -> NEval:
    """Build the non-linear application of ``f`` to ``args`` at ``positions``."""
    return NEval(f, tuple(positions), tuple(args))


def compose(f: Term, positions: tuple[int, ...], *gs: Term) -> Compose:
    """Build the composition contracting each ``gₖ``'s codomain into ``f``'s slot ``positions[k]``."""
    return Compose(f, tuple(positions), tuple(gs))


def chain(*fs: Term, at: int = 0) -> Term:
    """Compose ``fs`` in series, ``fs[0] ∘ fs[1] ∘ … ∘ fs[-1]``, right-applied at dual slot ``at``.

    It builds no node of its own, expanding instead into nested one-argument
    :class:`Compose` nodes: ``chain(proj, conv, embed)`` is ``Compose(proj, (0,),
    (Compose(conv, (0,), (embed,)),))``. Each junction feeds one map's codomain into the
    dual slot ``at`` of the map to its left, so ``chain(f, g) == Compose(f, (at,), (g,))``
    and longer chains nest right to left.

    A single map is returned unchanged, and a chain of none is an error.
    """
    if not fs:
        raise TermError("chain needs at least one map")
    if not isinstance(at, int) or at < 0:
        raise TermError(f"chain: at must be a non-negative int, got {at!r}")
    out: Term = fs[-1]
    for f in reversed(fs[:-1]):
        out = Compose(f, (at,), (out,))
    return out


def trace(f: Term, i: int, j: int) -> Trace:
    """Build the contraction of ``f``'s primal index ``i`` with its dual index ``j``."""
    return Trace(f, i, j)


def det(f: Term) -> Det:
    """Build the determinant of the endomorphism ``f``."""
    return Det(f)


def logdet(f: Term) -> LogDet:
    """Build the log-determinant of the endomorphism ``f``."""
    return LogDet(f)


def opaque_apply(op: OpaqueOp, arg: Term) -> OpaqueApply:
    """Build the application of the opaque map ``op`` to ``arg``."""
    return OpaqueApply(op, arg)


def solve(f: Term) -> Solve:
    """Build the inverse map ``f⁻¹`` of a bijective ``f``."""
    return Solve(f)


def pinv(f: Term) -> Pinv:
    """Build the Moore–Penrose pseudo-inverse map ``f†``."""
    return Pinv(f)


def corestrict(f: Term) -> Corestrict:
    """Build the corestriction ``f~ : dom(f) ⊸ Img(f)`` of an injective ``f`` onto its image."""
    return Corestrict(f)


def sqrt(f: Term) -> Sqrt:
    """Build the SPD operator square root of the endomorphism ``f``."""
    return Sqrt(f)


def solve_apply(f: Term, v: Term) -> SolveApply:
    """Build the solution of the system ``f·x = v``."""
    return SolveApply(f, v)


def pinv_apply(f: Term, v: Term) -> PinvApply:
    """Build the applied pseudo-inverse ``f†·v``."""
    return PinvApply(f, v)


def dagger(f: Term) -> Dagger:
    """Build the metric-transpose adjoint ``f†`` of ``f``."""
    return Dagger(f)


def inner(space: Space) -> Inner:
    """Build the inner-product map of ``space``."""
    return Inner(space)


def flat(space: Space) -> Flat:
    """Build the musical isomorphism ♭ ``space ⊸ space*``."""
    return Flat(space)


def sharp(space: Space) -> Sharp:
    """Build the musical isomorphism ♯ ``space* ⊸ space``."""
    return Sharp(space)


def project(vector: Term, space: Space | None = None) -> Project:
    """Build the projection of ``vector`` onto the subspace ``space``.

    ``space=None`` leaves the target subspace to inference.
    """
    return Project(vector, space)


def embed(vector: Term, space: Space) -> Embed:
    """Build the embedding of ``vector`` into the larger space ``space``."""
    return Embed(vector, space)


def assert_(vector: Term, space: Space, props: MapProps | None = None) -> Assert:
    """Build the assertion that ``vector`` has space ``space``, with optional map ``props``."""
    return Assert(vector, space, props)


def tuple_embed(space: Space, i: int) -> TupleEmbed:
    """Build the injection into the ``i``-th ⊕ component of ``space``."""
    return TupleEmbed(space, i)


def tuple_project(space: Space, i: int) -> TupleProject:
    """Build the extraction of the ``i``-th ⊕ component of ``space``."""
    return TupleProject(space, i)


def remetric(vector: Term, metric: Term) -> Remetric:
    """Build the re-metricization moving ``vector`` into ``WithInner(space(vector), metric)``."""
    return Remetric(vector, metric)


def lift(vector: Term, fused: tuple[int, ...], k: int | None = None) -> Lift:
    """Build the reshape fusing the tensor indices ``fused`` of ``vector``."""
    return Lift(vector, tuple(fused), k)


def inv_lift(vector: Term, fused: tuple[int, ...], k: int | None = None) -> InvLift:
    """Build the inverse reshape splitting the indices ``fused`` apart."""
    return InvLift(vector, tuple(fused), k)


def permute(vector: Term, perm: tuple[int, ...]) -> Permute:
    """Build the reordering of the ⊗ factors of ``vector``'s space by ``perm``."""
    return Permute(vector, tuple(perm))


def tuple_permute(vector: Term, perm: tuple[int, ...]) -> TuplePermute:
    """Build the reordering of the ⊕ components of ``vector`` by ``perm``."""
    return TuplePermute(vector, tuple(perm))


def kron(f: Term, g: Term) -> Kron:
    """Build the Kronecker product ``f ⊗ g`` of the maps ``f`` and ``g``."""
    return Kron(f, g)


def llam(*vars: Var, body: Term) -> LLam:
    """Build the linear lambda ``⊸`` over the bound variables ``vars`` and the keyword ``body``."""
    return LLam(tuple(vars), body)


def lam(
    binders: tuple[tuple[str, Kind], ...],
    body: Term | Space | Type | Dim,
    term_vars: tuple[Var, ...] = (),
) -> Lam:
    """Build the general non-linear binder over ``binders``."""
    return Lam(tuple(binders), body, tuple(term_vars))


def frame(var: Var, basis_body: Basis) -> Frame:
    """Build the value-binding lambda that binds ``var`` to the basis ``basis_body``."""
    return Frame(var, basis_body)


__all__ = [
    "TermError",
    "Term",
    # nodes
    "Var",
    "External",
    "Zero",
    "Identity",
    "Lit",
    "Scale",
    "Sum",
    "OPlusV",
    "OProdV",
    "Eval",
    "NEval",
    "Compose",
    "Trace",
    "Det",
    "LogDet",
    "OpaqueOp",
    "OpaqueApply",
    "Solve",
    "Pinv",
    "Corestrict",
    "sqrt",
    "Sqrt",
    "SolveApply",
    "PinvApply",
    "Dagger",
    "Inner",
    "Flat",
    "Sharp",
    "Project",
    "Embed",
    "Assert",
    "TupleEmbed",
    "TupleProject",
    "Remetric",
    "Lift",
    "InvLift",
    "Permute",
    "TuplePermute",
    "Kron",
    "LLam",
    "Lam",
    "Frame",
    "is_term",
    # smart constructors
    "var",
    "external",
    "zero",
    "identity",
    "lit",
    "scale",
    "sum_",
    "oplus_v",
    "oprod_v",
    "eval_",
    "neval",
    "compose",
    "chain",
    "trace",
    "det",
    "logdet",
    "opaque_apply",
    "solve",
    "pinv",
    "corestrict",
    "solve_apply",
    "pinv_apply",
    "dagger",
    "inner",
    "flat",
    "sharp",
    "project",
    "embed",
    "assert_",
    "tuple_embed",
    "tuple_project",
    "remetric",
    "lift",
    "inv_lift",
    "permute",
    "tuple_permute",
    "kron",
    "llam",
    "lam",
    "frame",
]
