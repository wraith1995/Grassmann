"""Term (vector) ADT (`20 §6`), with the two binders (`20 §1`/`§2`).

Favors the FEMOracle impl, n-ary, real-only. The two binders replace the impl's
``Abstract`` + ``nonLinearVars``:

- **``LLam``** — the **linear** lambda ``⊸`` (multi-var). Result space is **independent**
  of the linear args (representable as a value in ``⊗(Dual sᵢ) ⊗ s_e``). Linearity ⊥
  dependence: a linear arg may not appear in a binder domain annotation nor as an
  explicit space/nat argument (those would make the result depend on it) — that is the
  structural well-formedness rule enforced here.
- **``Lam``** — the one general, body-polymorphic **non-linear** binder ``Lam(b:σ. body)``
  whose body may be a Term, Space, Type, or Nat, and whose domain ``b:σ`` is a value
  (a space ``s``), a space, or a nat. A ``Lam`` whose body is a Space / Type is a
  **SpaceFunction / TypeFunction** (`20 §4`).

Position/index convention (`20 §6`): one mechanism — an integer index list selects
parameter slots (``Eval``/``NEval``/``Compose``), tensor indices to fuse
(``Lift``/``InvLift`` + optional ``k``), or indices to contract (``Trace``).

Well-formedness here is **structural only** (arity, child kinds, the LLam linearity
guard). Full *type* checking — that the contracted slots are dual, that summands share a
space, etc. — is IW3.
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
    """``Var(x, props?)`` — a (term) variable, optionally carrying asserted map
    properties (`20 §6`: props ON VARS ONLY in v1).

    ``space`` is the variable's space; ``uid`` is a fresh identity so that two distinct
    binders' variables of the same name/space do not collide (used by binders).
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
    """``External(s, data, args…)`` — foreign fn/value: opaque data + arg deps."""

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
    """``Zero(s?)`` — the zero vector of a space.

    ``space`` is **optional** (B1): when ``None`` the space is left to inference, which
    recovers it via first-order unification (a fresh ``SpaceMeta`` solved by the
    surrounding context — a ``Sum`` sibling, a ``check``/``Assert`` against an expected
    space, etc.). Supplying ``space`` explicitly works exactly as before.
    """

    space: Space | None = None

    def __post_init__(self) -> None:
        _require_space_or_none(self.space, "Zero")

    def __repr__(self) -> str:
        return f"Zero({self.space!r})"


@dataclass(frozen=True)
class Identity:
    """``Identity(s)`` — the identity map on ``s`` (space ``~s ⊗ s``)."""

    space: Space

    def __post_init__(self) -> None:
        _require_space_field(self.space, "Identity")

    def __repr__(self) -> str:
        return f"Identity({self.space!r})"


@dataclass(frozen=True)
class Lit:
    """``Scalar(r)`` — a real scalar literal (term-level)."""

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)):
            raise TermError(f"Lit: value must be a real number, got {self.value!r}")

    def __repr__(self) -> str:
        return f"Lit({self.value!r})"


@dataclass(frozen=True)
class Scale:
    """``Scale(r, v)`` — scalar multiple. ``r`` is a real or a scalar-valued Term."""

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
    """``Sum(v…)`` — n-ary vector addition (all summands share a space; checked IW3)."""

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
    """``OPlus(v…)`` — n-ary direct-sum *stack* of vectors (space ``⊕(sᵢ)``)."""

    vectors: tuple[Term, ...]

    def __post_init__(self) -> None:
        for v in self.vectors:
            _require_term(v, "OPlus")

    def __repr__(self) -> str:
        return f"OPlusV({_tuple_repr(self.vectors)})"


@dataclass(frozen=True)
class OProdV:
    """``OProd(v…)`` — n-ary tensor product of vectors (space ``⊗(sᵢ)``)."""

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
    """``Eval(f, positions, args…)`` — **linear** application (contraction view).

    Apply a linear map ``f`` at the selected dual slots ``positions`` with ``args``.
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
    """``NEval(f, positions, args…)`` — **non-linear** application (apply a ``Lam``)."""

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

    def __repr__(self) -> str:
        return f"NEval({self.f!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.args)})"


@dataclass(frozen=True)
class Compose:
    """``Compose(f, positions, gs…)`` — contract each ``gₖ``'s codomain into ``f``'s
    selected slot."""

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
    """``Trace(f, i, j)`` — contract (primal) index ``i`` with dual index ``j``."""

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


# ===========================================================================
# Inverses & adjoint.
# ===========================================================================


@dataclass(frozen=True)
class Solve:
    """``Solve(f)`` — the inverse map ``f⁻¹`` (requires ``f`` asserted ``bij``; IW3)."""

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Solve.f")

    def __repr__(self) -> str:
        return f"Solve({self.f!r})"


@dataclass(frozen=True)
class Pinv:
    """``Pinv(f)`` — the Moore–Penrose pseudo-inverse map ``f†``."""

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Pinv.f")

    def __repr__(self) -> str:
        return f"Pinv({self.f!r})"


@dataclass(frozen=True)
class SolveApply:
    """``SolveApply(f, v)`` — solve ``f·x = v`` (the applied form of ``Solve``)."""

    f: Term
    v: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "SolveApply.f")
        _require_term(self.v, "SolveApply.v")

    def __repr__(self) -> str:
        return f"SolveApply({self.f!r}, {self.v!r})"


@dataclass(frozen=True)
class PinvApply:
    """``PinvApply(f, v)`` — apply the pseudo-inverse: ``f†·v``."""

    f: Term
    v: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "PinvApply.f")
        _require_term(self.v, "PinvApply.v")

    def __repr__(self) -> str:
        return f"PinvApply({self.f!r}, {self.v!r})"


@dataclass(frozen=True)
class Dagger:
    """``Dagger(f)`` — metric-transpose adjoint (real ⇒ no conjugate)."""

    f: Term

    def __post_init__(self) -> None:
        _require_term(self.f, "Dagger.f")

    def __repr__(self) -> str:
        return f"Dagger({self.f!r})"


@dataclass(frozen=True)
class Inner:
    """``Inner(s)`` — the inner-product map of ``s`` (reads ``s``'s metric)."""

    space: Space

    def __post_init__(self) -> None:
        _require_space_field(self.space, "Inner")

    def __repr__(self) -> str:
        return f"Inner({self.space!r})"


@dataclass(frozen=True)
class Flat:
    """``Flat(s)`` — musical ♭, primal→dual ``s ⊸ s*`` (reads metric)."""

    space: Space

    def __post_init__(self) -> None:
        _require_space_field(self.space, "Flat")

    def __repr__(self) -> str:
        return f"Flat({self.space!r})"


@dataclass(frozen=True)
class Sharp:
    """``Sharp(s)`` — musical ♯, dual→primal ``s* ⊸ s`` (reads metric)."""

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
    """``Project(v, s?)`` — project ``v`` onto a subspace ``s`` (``s ⊑ space(v)``; IW3).

    ``space`` (the *target subspace*) is **optional** (B1): when ``None`` it is left to
    inference. Because a subspace is not unique, the omitted target is recoverable only
    when an expected space flows in from the surrounding context (the ⇐ check direction),
    which unifies the fresh ``SpaceMeta`` and then the deferred ``s ⊑ space(v)`` obligation
    is validated. Supplying ``space`` explicitly works exactly as before.
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
    """``Embed(v, s)`` — embed ``v`` into a larger space ``s`` (``space(v) ⊑ s``)."""

    vector: Term
    space: Space

    def __post_init__(self) -> None:
        _require_term(self.vector, "Embed.vector")
        _require_space_field(self.space, "Embed")

    def __repr__(self) -> str:
        return f"Embed({self.vector!r}, {self.space!r})"


@dataclass(frozen=True)
class Assert:
    """``Assert(v, s)`` — assert ``v`` has space ``s`` (``space(v) ~ s``; IW3)."""

    vector: Term
    space: Space

    def __post_init__(self) -> None:
        _require_term(self.vector, "Assert.vector")
        _require_space_field(self.space, "Assert")

    def __repr__(self) -> str:
        return f"Assert({self.vector!r}, {self.space!r})"


@dataclass(frozen=True)
class TupleEmbed:
    """``TupleEmbed(s, i)`` — inject the ``i``-th ⊕ component (pad). ``s = ⊕(sₖ)``."""

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
    """``TupleProject(s, i)`` — extract the ``i``-th ⊕ component (slice)."""

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
    """``Remetric(v, g)`` — enter ``WithInner(s, g)`` (`20` point 9)."""

    vector: Term
    metric: Term

    def __post_init__(self) -> None:
        _require_term(self.vector, "Remetric.vector")

    def __repr__(self) -> str:
        return f"Remetric({self.vector!r}, {self.metric!r})"


@dataclass(frozen=True)
class Lift:
    """``Lift(v, fused, k?)`` — index reshape (≡ curry); fuse the listed tensor indices."""

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
    """``InvLift(v, fused, k?)`` — inverse index reshape (≡ uncurry)."""

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
class Kron:
    """``Kron(f, g)`` — Kronecker product of maps (first-class)."""

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
    """``LLam(xs:[x:s]. e)`` — the linear lambda ``⊸`` (multi-var).

    Result space must be **independent** of the linear vars (`20 §1`/`§2`). The
    structural well-formedness guard enforced here: each bound var must NOT appear in a
    position that makes the result space depend on it — i.e. it may not occur inside a
    ``Lam`` binder *domain* annotation, nor as an explicit space/nat argument of an
    ``NEval``/``App``. (Full linearity checking is IW3; this catches the structurally
    detectable violations the spec calls out.)
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
    """``Lam(bs:[b:σ]. body)`` — the general non-linear binder (`20 §4`).

    Each bound entry is a ``(name, sort)`` pair where ``sort`` is a :class:`Sort`
    (``space``/``nat``/``*``) or a :class:`SpaceSort` (a value domain). The ``body`` may
    be a Term, Space, Type, or Nat (``Dim``) — body-polymorphic. A Space body makes this
    a SpaceFunction; a Type body a TypeFunction.

    Bound names that are value-domain (a ``SpaceSort``) introduce a term ``Var``; the
    corresponding ``Var`` objects are stored in ``term_vars`` for reference by the body.
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

    def __repr__(self) -> str:
        binders = _tuple_repr(self.binders)
        if self.term_vars:
            return f"Lam({binders}, {self.body!r}, {_tuple_repr(self.term_vars)})"
        return f"Lam({binders}, {self.body!r})"


@dataclass(frozen=True)
class Frame:
    """``Frame`` over a value — the value-binding ``BLam`` (`20 §7`).

    Listed in the IW1 Term scope (`60 §1`). A ``Frame`` binds a runtime value ``x:s`` to
    a basis (``x ↦ basis of τ(x)``). The basis-producing body lives in :mod:`basis`; we
    store the bound term ``Var`` and the basis body (a ``Basis`` forward ref, to avoid a
    runtime basis.py import cycle).
    The richer basis tower (``BLam``/``BApp``) is in :mod:`grassmann.core.basis`.
    """

    var: Var
    basis_body: Basis

    def __post_init__(self) -> None:
        if not isinstance(self.var, Var):
            raise TermError(f"Frame: var must be a Var, got {self.var!r}")

    def __repr__(self) -> str:
        return f"Frame({self.var!r}, {self.basis_body!r})"


# A Term is any of the node types above.
Term = Var | External | Zero | Identity | Lit | Scale | Sum | OPlusV | OProdV | Eval | NEval | Compose | Trace | Solve | Pinv | SolveApply | PinvApply | Dagger | Inner | Flat | Sharp | Project | Embed | Assert | TupleEmbed | TupleProject | Remetric | Lift | InvLift | Kron | LLam | Lam | Frame

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
    Solve,
    Pinv,
    SolveApply,
    PinvApply,
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
    """Like :func:`_require_space_field` but permits ``None`` (an OPTIONAL space left to
    inference, B1)."""
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
    """Reject the structurally detectable ways a linear var would make the result
    space *depend* on it (`20 §1`/`§2`): occurring in a ``Lam`` binder domain
    annotation, or as an explicit space/nat argument of an ``NEval``/``App``.

    Also requires each linear var to occur at least once in the body (a vacuous linear
    binder is almost always a mistake; full per-variable linearity is IW3).
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
                            f"the result space would depend on it, violating linearity "
                            f"(20 §1/§2)"
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
            case _:
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
    return Var(name, space, props)


def external(space: Space, data: object, *args: Term) -> External:
    return External(space, data, tuple(args))


def zero(space: Space | None = None) -> Zero:
    return Zero(space)


def identity(space: Space) -> Identity:
    return Identity(space)


def lit(value: float) -> Lit:
    return Lit(value)


def scale(scalar: float | Term, vector: Term) -> Scale:
    return Scale(scalar, vector)


def sum_(*vectors: Term) -> Sum:
    return Sum(tuple(vectors))


def oplus_v(*vectors: Term) -> OPlusV:
    return OPlusV(tuple(vectors))


def oprod_v(*vectors: Term) -> OProdV:
    return OProdV(tuple(vectors))


def eval_(f: Term, positions: tuple[int, ...], *args: Term) -> Eval:
    return Eval(f, tuple(positions), tuple(args))


def neval(f: Term, positions: tuple[int, ...], *args: Term | Space | Dim) -> NEval:
    return NEval(f, tuple(positions), tuple(args))


def compose(f: Term, positions: tuple[int, ...], *gs: Term) -> Compose:
    return Compose(f, tuple(positions), tuple(gs))


def chain(*fs: Term, at: int = 0) -> Term:
    """Series composition ``fs[0] ∘ fs[1] ∘ … ∘ fs[-1]`` (right-applied at dual slot
    ``at``).

    A thin smart constructor — **no new core node**. It mimics FEMOracle's nested
    single-arg ``Compose`` (``pullback.py:778`` builds ``Compose(proj, (0,),
    [Compose(conv, (0,), [embed])])`` for ``proj ∘ conv ∘ embed``). Each junction feeds
    one map's codomain into the dual slot ``at`` of the map to its left, so
    ``chain(f, g) == Compose(f, (at,), (g,))`` and longer chains nest right-to-left.

    A single map is returned unchanged; ``chain()`` with no args is an error.
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
    return Trace(f, i, j)


def solve(f: Term) -> Solve:
    return Solve(f)


def pinv(f: Term) -> Pinv:
    return Pinv(f)


def solve_apply(f: Term, v: Term) -> SolveApply:
    return SolveApply(f, v)


def pinv_apply(f: Term, v: Term) -> PinvApply:
    return PinvApply(f, v)


def dagger(f: Term) -> Dagger:
    return Dagger(f)


def inner(space: Space) -> Inner:
    return Inner(space)


def flat(space: Space) -> Flat:
    return Flat(space)


def sharp(space: Space) -> Sharp:
    return Sharp(space)


def project(vector: Term, space: Space | None = None) -> Project:
    return Project(vector, space)


def embed(vector: Term, space: Space) -> Embed:
    return Embed(vector, space)


def assert_(vector: Term, space: Space) -> Assert:
    return Assert(vector, space)


def tuple_embed(space: Space, i: int) -> TupleEmbed:
    return TupleEmbed(space, i)


def tuple_project(space: Space, i: int) -> TupleProject:
    return TupleProject(space, i)


def remetric(vector: Term, metric: Term) -> Remetric:
    return Remetric(vector, metric)


def lift(vector: Term, fused: tuple[int, ...], k: int | None = None) -> Lift:
    return Lift(vector, tuple(fused), k)


def inv_lift(vector: Term, fused: tuple[int, ...], k: int | None = None) -> InvLift:
    return InvLift(vector, tuple(fused), k)


def kron(f: Term, g: Term) -> Kron:
    return Kron(f, g)


def llam(*vars: Var, body: Term) -> LLam:
    """``LLam`` — pass the bound :class:`Var`s positionally and ``body=`` as keyword."""
    return LLam(tuple(vars), body)


def lam(
    binders: tuple[tuple[str, Kind], ...],
    body: Term | Space | Type | Dim,
    term_vars: tuple[Var, ...] = (),
) -> Lam:
    return Lam(tuple(binders), body, tuple(term_vars))


def frame(var: Var, basis_body: Basis) -> Frame:
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
    "Solve",
    "Pinv",
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
    "solve",
    "pinv",
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
    "kron",
    "llam",
    "lam",
    "frame",
]
