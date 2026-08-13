"""Total covariant differentiation — symbolic forward-mode.

``diff(e, x, deps)`` differentiates a **closed** term ``e`` (no free space variables)
w.r.t. a single value variable ``x : X``, returning the total derivative as a
coordinate-free linear map ``X ⊸ space(e)`` (the Jacobian, space ``Dual(X) ⊗ space(e)``).

**Method — directional derivative + seed.** We compute the *directional derivative*
``De[h]`` — a **value** in ``space(e)`` that is linear in a fresh seed direction
``h : X`` — by forward-mode rules (product / chain / quotient), then wrap it in
``LLam((h,). De[h])``. The ``LLam``→coordinate-tensor lowering seeds ``h`` with the
identity basis and reads off the whole Jacobian. Working with *values* (not Jacobian
maps) throughout keeps every intermediate in the ordinary value algebra — no map is ever
re-applied, so the construction is layout-safe.

**Marked dependence.** ``deps = {var.uid: dval}`` marks x-dependent value vars. ``dval``
is the supplied derivative map ``∂val/∂x : X ⊸ space(val)`` (its directional derivative is
``dval·h``); ``None`` means "depends on x with an unknown derivative" — a fresh symbolic
map var (a Program input) is allocated. A var neither ``x`` nor in ``deps`` is treated as
x-independent (derivative ``0``).

**Covariant.** For a value living in a ``WithInner(s, g)`` (a tangent/cotangent space)
with an x-dependent metric, the flat derivative is corrected by the metric (Levi-Civita)
connection ``Γ`` built from ``∂g`` (see :func:`_covariant_correction`); a constant/flat
``g`` gives ``Γ = 0`` and the total derivative is recovered exactly.

**Scope.** Nonlinear bodies are fully supported (``sqrt``/``div``/inner/``Solve``/``Pinv``/
``Dagger``/…). The one exclusion: differentiating a **higher-order function** — a
``Lam``-as-function / SpaceFunction / ``Frame``, or any path where ``x`` flows into a
*space* position — is a :class:`DiffError`.
"""

from __future__ import annotations

from typing import assert_never

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from grassmann.core.space import (
    Scalar,
    Space,
    WithInner,
    dual,
    is_space,
    linear_map_space,
    spliced_oprod,
)
from grassmann.core.term import (
    Assert,
    Corestrict,
    Compose,
    Dagger,
    Det,
    Embed,
    Eval,
    External,
    Flat,
    Frame,
    Identity,
    Inner,
    InvLift,
    Kron,
    Lam,
    Lift,
    Lit,
    LLam,
    LogDet,
    NEval,
    OpaqueApply,
    OpaqueOp,
    OPlusV,
    OProdV,
    Permute,
    Pinv,
    PinvApply,
    Project,
    Remetric,
    Scale,
    Sharp,
    Solve,
    SolveApply,
    Sqrt,
    Sum,
    Term,
    Trace,
    TupleEmbed,
    TuplePermute,
    TupleProject,
    Var,
    Zero,
    is_term,
)


class DiffError(Exception):
    """Raised when a term cannot be differentiated (open term, higher-order fn, …)."""


_counter = itertools.count()


@dataclass
class _DCtx:
    """Differentiation context: the variable ``x`` and its seed direction ``h``."""

    x: Var
    h: Var  # the seed direction, a fresh Var of x.space
    deps: dict[int, Term | None]
    # memo: uid -> the directional-derivative VALUE of a marked var (so a ``None`` dep's
    # fresh map var is allocated once and shared across occurrences).
    _ddval: dict[int, Term] = field(default_factory=dict)

    @property
    def X(self) -> Space:
        return self.x.space


# ===========================================================================
# Derivative-input provenance + enumeration.
#
# A ``deps[uid] = None`` tracer differentiates into a fresh symbolic derivative-MAP input
# ``∂ⁿsource/∂xⁿ : X ⊸ …`` (a Program input pointwise samples).  These two registries give the
# inputs structured provenance — (base tracer var, derivative order) — so a caller can read off
# EXACTLY which derivatives a (possibly multiply-differentiated) term references, by var identity
# rather than by parsing the ``d_…`` name.  They also DEDUPE: the derivative of a given source
# w.r.t. a given ``x`` is allocated once and reused across repeated/nested ``diff`` calls, so the
# enumerated requirement set is a true set (no duplicate input per (source, order)).
_DERIV_INPUTS: dict[int, DerivInput] = {}            # deriv-var.uid -> provenance record
_DERIV_INPUT_BY_KEY: dict[tuple[int, int], Var] = {}  # (immediate-source.uid, x.uid) -> deriv var
_OPAQUE_JAC_BY_KEY: dict[tuple[str, Term], Var] = {}  # (op.name, arg) -> Jacobian Df input var


@dataclass(frozen=True)
class DerivInput:
    """Provenance of a symbolic derivative-map input allocated for a ``deps=None`` tracer.

    ``var`` is the Program-input map ``∂ᵒʳᵈᵉʳsource/∂xᵒʳᵈᵉʳ``; ``source`` is the base
    tracer ``Var`` it ultimately differentiates; ``order`` is the derivative order (1 = first).
    """

    var: Var
    source: Var
    order: int


def _tower_space(ctx: _DCtx, cod: Space) -> Space:
    """Build a derivative-tower var's space ``X ⊸ cod`` **FLAT**.

    This is THE spelling of such a var's space.  An order-``n`` derivative input is the
    order-``(n−1)`` one differentiated once more, so its space is ``X ⊸ (X ⊸ (… ⊸ V))``.
    :func:`linear_map_space` RIGHT-NESTS that, and each nesting level is one array axis, so it
    would realize as ``(|X|, |X|ⁿ⁻¹·|V|)`` — a matrix whose second
    axis fuses ``n−1`` seed slots with the value.  This builder SPLICES instead, giving one axis
    per differentiation plus the value's own axes:

    ==========  ==============================  ==========================
    order       right-nested spelling           flat spelling (this)
    ==========  ==============================  ==========================
    1           ``(d, 1)``                      ``(d, 1)``      — identical
    2           ``(d, d)``                      ``(d, d, 1)``
    3           ``(d, d²)``                     ``(d, d, d, 1)``
    ==========  ==============================  ==========================

    Flat is what makes the ``(|X|,…,|X|,k,n)`` layout this module documents true, and it is what
    carries rank ≥ 3.  ⊗ is associative but the two spellings are two LAYOUTS, and
    :func:`grassmann.infer.unify.unify` refuses to identify them (``AssociationMismatch``): from
    three leaves up, a nested tower cannot meet the flat ``tensor_power``/``Kron`` products it is
    contracted against, so a rank-3 covariant pullback of a jet is unrepresentable in the nested
    spelling.  Flat, it realizes exactly ``(A⁻¹)^{⊗k}``
    (``tests/test_rank_k_covariant_pullback.py``).  ``k = 2`` is blind to the difference: the
    ⊗-unit rewrite bridges exactly one level (``(d, d)`` vs ``(d, d, 1)``).

    Parameters
    ----------
    ctx
        The differentiation context; ``ctx.X`` is the seed space the tower differentiates along.
    cod
        The codomain — the space the order-``(n−1)`` value lives in.

    Returns
    -------
    Space
        ``Dual(X) ⊗ cod``, spliced flat.

    Notes
    -----
    ⚠ The tower is the ONLY thing spelled this way.  A genuine operator — an opaque op's Jacobian
    ``op.dom ⊸ op.cod``, ``Alt₃ : V⊗³ ⊸ V⊗³`` — stays a two-factor :func:`linear_map_space`
    MATRIX; splicing those hands the FFS/SVD seam a rank-``2k`` array where it expects a matrix.
    """
    return spliced_oprod(dual(ctx.X), cod)


def _alloc_deriv_input(source: Var, ctx: _DCtx, space: Space | None = None) -> Var:
    """Get-or-create the symbolic derivative-map input ``∂source/∂x``, recording its provenance.

    The codomain space is ``source.space`` by default, or ``space`` when the derivative lands
    elsewhere — e.g. a metric gradient ``M ⊸ M*⊗M*``.  Deduped by ``(source.uid, x.uid)`` so
    repeated/nested ``diff`` of the same source reuses one input.  When ``source`` is itself a
    derivative input, the new var inherits the base tracer and bumps the order (so an order-``n``
    chain reports ``(base, n)``).

    The space is the FLAT tower spelling (:func:`_tower_space`), so an order-``n`` input has ``n``
    seed axes followed by the value's own — ``(d, d, 1)`` at order 2, not ``(d, d)``.
    """
    key = (source.uid, ctx.x.uid)
    existing = _DERIV_INPUT_BY_KEY.get(key)
    if existing is not None:
        return existing
    cod = space if space is not None else source.space
    var = Var(f"d_{source.name}_{next(_counter)}", _tower_space(ctx, cod))
    parent = _DERIV_INPUTS.get(source.uid)
    base, order = (parent.source, parent.order + 1) if parent is not None else (source, 1)
    _DERIV_INPUTS[var.uid] = DerivInput(var=var, source=base, order=order)
    _DERIV_INPUT_BY_KEY[key] = var
    return var


def enumerate_derivative_inputs(term: Term) -> list[DerivInput]:
    """List the symbolic derivative-map inputs occurring in ``term``, with their provenance.

    These are the ``deps=None`` tracer derivatives, each carrying ``(base tracer var, derivative
    order)``.  A requirements pass reads this after lowering to sample exactly the chart/field
    derivatives actually used — and to confirm pruning unreferenced tracers leaves a valid
    program.  Deterministic order: by ``(source name, order, uid)``.
    """
    seen: dict[int, DerivInput] = {}

    def walk(n: object) -> None:
        if isinstance(n, Var) and n.uid in _DERIV_INPUTS:
            seen[n.uid] = _DERIV_INPUTS[n.uid]
        for c in _children(n):
            walk(c)

    walk(term)
    return sorted(seen.values(), key=lambda r: (r.source.name, r.order, r.var.uid))


def clear_deriv_inputs() -> None:
    """Drop all recorded derivative-input provenance and dedup keys.

    For test hygiene / a fresh tracer session.  Safe to omit — keys are globally unique Var uids,
    so stale reuse cannot occur.
    """
    _DERIV_INPUTS.clear()
    _DERIV_INPUT_BY_KEY.clear()
    _OPAQUE_JAC_BY_KEY.clear()


def diff(e: Term, x: Var, deps: dict[int, Term | None] | None = None) -> Term:
    """Total covariant derivative ``d e / d x : X ⊸ space(e)`` (the public entry point)."""
    if not is_term(e):
        raise DiffError(f"diff: expected a Term, got {e!r}")
    if not isinstance(x, Var):
        raise DiffError(f"diff: the differentiation variable must be a Var, got {x!r}")
    _guard_closed_and_first_order(e, x, deps or {})

    from grassmann.infer.synth import synth_space

    R = synth_space(e)
    h = Var(f"d{x.name}", x.space)
    ctx = _DCtx(x=x, h=h, deps=deps or {})
    cov = _covariant_section(e, ctx)
    if cov is not None:
        # A covariant section ``Remetric(sec, g)`` over a manifold with an x-dependent
        # metric: ∇sec = d(sec) + Γ(h)·sec, computed entirely in the bare (coordinate-free)
        # basis — the derivative never realizes a metric basis.
        sec, M, conn_builder = cov
        sspace = synth_space(sec)
        body = _add(_dd(sec, ctx), _connection_correction(sec, sspace, M, conn_builder, ctx), sspace)
        rspace = sspace
    else:
        body = _dd(e, ctx)
        rspace = R
    if isinstance(body, Zero):  # audit: ok EX-ISINSTANCE-LADDER — narrowing: Zero-derivative shape test, else LLam wrap [approved 2026-07-03]
        # e is x-independent: the derivative is the zero map (an LLam over an unused seed
        # would violate the linear discipline, so emit the zero map directly).
        return Zero(linear_map_space(ctx.X, rspace))
    return LLam((h,), body)


# ===========================================================================
# Directional derivative — De[h], a VALUE in space(e), linear in h.
# ===========================================================================


def _dd(e: Term, ctx: _DCtx) -> Term:
    """Differentiate ``e`` in direction ``ctx.h`` — a value in ``space(e)``."""
    match e:
        case Var():
            return _dd_var(e, ctx)
        case Zero() | Lit() | Identity() | TupleEmbed() | TupleProject():
            return _zero(e, ctx)
        case Inner() | Flat() | Sharp():
            # Musical maps of a *constant* space are constant. But if the operator's space is a
            # ``WithInner(s, g)`` whose metric ``g`` depends on x, the operator itself varies — its
            # derivative is nonzero (e.g. ``d⟨a,b⟩_g = aᵀ(∂g)b``).  The covariant correction only
            # intercepts ``Remetric`` sections, NOT a bare Inner/Flat/Sharp, so returning Zero here
            # would be silent-wrong; raise loudly instead (wrap the section in ``Remetric`` for the
            # connection, or differentiate the explicit ``WithInner``-metric form).
            if _xdep_metric_space(e.space, ctx):
                raise DiffError(
                    "diff: Inner/Flat/Sharp of a space with an x-dependent metric has a nonzero "
                    "operator derivative (not yet supported); use Remetric(sec, g) for the covariant "
                    "connection, or supply the metric-pairing in an explicit form"
                )
            return _zero(e, ctx)
        case External():
            return _dd_external(e, ctx)
        case Scale():
            return _dd_scale(e, ctx)
        case Sum(vs):
            return _sum([_dd(v, ctx) for v in vs], e, ctx)
        case OPlusV(vs):
            # d(⊕ᵢ vᵢ)[h] = ⊕ᵢ dvᵢ[h].  When every component is x-independent the whole
            # direct sum is, so collapse to the zero VALUE in space(e) — otherwise the caller
            # wraps an OPlusV-of-Zeros in an LLam over an unused seed (an invalid linear map).
            # Mirrors the all-zero collapse of the Sum / OProdV arms via _zero / _sum.
            ds = tuple(_dd(v, ctx) for v in vs)
            if all(isinstance(d, Zero) for d in ds):
                return _zero(e, ctx)
            return OPlusV(ds)
        case OProdV(vs):
            return _dd_multilinear_tuple(OProdV, vs, ctx, e)
        case Eval():
            return _dd_eval(e, ctx)
        case Compose():
            return _dd_compose(e, ctx)
        case Trace(f, i, j):
            return _wrap_linear(lambda d: Trace(d, i, j), _dd(f, ctx), e, ctx)
        case Det():
            return _dd_det(e, ctx)
        case LogDet():
            return _dd_logdet(e, ctx)
        case OpaqueApply():
            return _dd_opaque(e, ctx)
        case Solve():
            return _dd_solve(e, ctx)
        case Pinv():
            return _dd_pinv(e, ctx)
        case Sqrt():
            return _dd_sqrt(e, ctx)
        case SolveApply():
            return _dd_solve_apply(e, ctx)
        case PinvApply():
            return _dd_pinv_apply(e, ctx)
        case Dagger(f):
            return _wrap_linear(Dagger, _dd(f, ctx), e, ctx)
        case Project(v, s):
            return _dd_project(v, s, e, ctx)
        case Embed(v, s):
            return _dd_embed(v, s, e, ctx)
        case Assert(v, s):
            return _wrap_linear(lambda d: Assert(d, s), _dd(v, ctx), e, ctx)
        case Lift(v, fused, k):
            return _wrap_linear(lambda d: Lift(d, fused, k), _dd(v, ctx), e, ctx)
        case InvLift(v, fused, k):
            return _wrap_linear(lambda d: InvLift(d, fused, k), _dd(v, ctx), e, ctx)
        case Permute(v, perm):
            return _wrap_linear(lambda d: Permute(d, perm), _dd(v, ctx), e, ctx)
        case Remetric(v, g):
            return _wrap_linear(lambda d: Remetric(d, g), _dd(v, ctx), e, ctx)
        case Kron():
            return _dd_kron(e, ctx)
        case LLam(vars_, body):
            return _dd_llam(e, vars_, body, ctx)
        case NEval(fn, pos, args) if isinstance(fn, Lam):
            return _dd_neval_lam(e, fn, pos, args, ctx)
        case NEval() | Lam() | Frame():
            raise DiffError(
                f"diff: cannot differentiate a higher-order construct ({type(e).__name__}); "
                f"differentiation of SpaceFunctions / Lam-as-function / Frame is unsupported"
            )
        case TuplePermute():
            raise DiffError("diff: no derivative rule for TuplePermute")
        case Corestrict():
            # ``Corestrict(f) : dom(f) ⊸ Img(f)`` has a MOVING codomain: perturbing ``f``
            # moves the subspace ``Img(f)`` itself, so the difference quotient does not live
            # in a fixed space and there is no linear-map derivative to hand back.  Refuse
            # loudly (differentiate the ambient ``f`` and corestrict afterwards instead).
            raise DiffError(
                "diff: no derivative rule for Corestrict — its codomain Img(f) moves with f"
            )
        case _ as unreachable:
            assert_never(unreachable)


def _dd_var(e: Var, ctx: _DCtx) -> Term:
    if e.uid == ctx.x.uid:
        return ctx.h  # d x = h (the seed direction)
    if e.uid in ctx.deps:
        if e.uid not in ctx._ddval:
            dval = ctx.deps[e.uid]
            if dval is None:
                # Unknown derivative ⇒ a fresh symbolic map var ∂e/∂x : X ⊸ space(e); its
                # directional derivative is (∂e/∂x)·h.  Provenance-tracked + deduped so a
                # requirements pass can enumerate (source, order) — see _alloc_deriv_input.
                dmap = _alloc_deriv_input(e, ctx)
            else:
                dmap = dval
            ctx._ddval[e.uid] = Eval(dmap, (0,), (ctx.h,))
        return ctx._ddval[e.uid]
    if e.uid in _QR_FRAME_VARS:
        # A self-differentiating QR-frame tower element: its derivative is the next
        # tower var, applied to the seed — ``dQₙ = Eval(Qₙ₊₁, (0,), (h,))``.  Recursing through this
        # rule (``diff`` of ``Eval(Qₙ₊₁, h)`` hits ``Qₙ₊₁`` again) gives ARBITRARY differentiation
        # order from a single registry.
        return Eval(_qr_next(e, ctx), (0,), (ctx.h,))
    return Zero(e.space)  # x-independent variable


def _dd_external(e: External, ctx: _DCtx) -> Term:
    """Elementwise nonlinear ops + the Hadamard product — chain / product rules.

    Each unary op has an elementwise coefficient ``c = f'(u)`` (a value in ``e.space``); the
    derivative is ``c ⊙ du`` via :func:`_prod`, which is a plain ``Scale`` for a ``Scalar``
    operand (byte-identical to the original scalar arms) and the Hadamard pointwise product
    for any other Euclidean space — so these ops differentiate over vectors/matrices too.
    """
    name = getattr(e.data, "name", None)
    args = e.args
    sp = e.space

    # an argument-less External is a CONSTANT (e.g. a basis/volume-form vector inside a Hodge
    # star) ⇒ its derivative is zero.
    if not args:
        return Zero(sp)

    # unary ops: coefficient c = f'(u), then c ⊙ du -----------------------------------
    _UNARY = {
        "sqrt": lambda u: Scale(0.5, _recip(_sqrt(u, sp), sp)),  # 1/(2√u)
        "reciprocal": lambda u: Scale(-1.0, _recip(_prod(u, u, sp), sp)),  # −1/u²
        "abs": lambda u: _sign(u, sp),  # sgn(u)
        "exp": lambda u: _exp(u, sp),  # eᵘ
        "log": lambda u: _recip(u, sp),  # 1/u
        "square": lambda u: Scale(2.0, u),  # 2u
        "sin": lambda u: _cos(u, sp),  # cos(u)
        "cos": lambda u: Scale(-1.0, _sin(u, sp)),  # −sin(u)
        "tan": lambda u: _recip(_prod(_cos(u, sp), _cos(u, sp), sp), sp),  # 1/cos²(u)
        "sinh": lambda u: _cosh(u, sp),  # cosh(u)
        "cosh": lambda u: _sinh(u, sp),  # sinh(u)
        "tanh": lambda u: _recip(_prod(_cosh(u, sp), _cosh(u, sp), sp), sp),  # sech²(u)
        # derivatives needing the unit constant 1±g(u) (built via _ones, see above)
        "atan": lambda u: _recip(Sum((_ones(u, sp), _prod(u, u, sp))), sp),  # 1/(1+u²)
        "asin": lambda u: _recip(_sqrt(_one_minus_sq(u, sp), sp), sp),  # 1/√(1−u²)
        "acos": lambda u: Scale(-1.0, _recip(_sqrt(_one_minus_sq(u, sp), sp), sp)),  # −1/√(1−u²)
        # σ' = σ(1−σ);  softplus' = σ
        "sigmoid": lambda u: _prod(
            _sigmoid(u, sp), Sum((_ones(u, sp), Scale(-1.0, _sigmoid(u, sp)))), sp
        ),
        "softplus": lambda u: _sigmoid(u, sp),
    }
    if name in _UNARY:
        (u,) = args
        du = _dd(u, ctx)
        if isinstance(du, Zero):
            return _zero(e, ctx)
        return _prod(_UNARY[name](u), du, sp)
    if name in ("sign", "ones"):
        return _zero(e, ctx)
    if name == "div":
        u, w = args
        du, dw = _dd(u, ctx), _dd(w, ctx)
        parts = []
        if not isinstance(du, Zero):
            parts.append(_prod(_recip(w, sp), du, sp))  # du / w
        if not isinstance(dw, Zero):
            coeff = Scale(-1.0, _prod(u, _recip(_prod(w, w, sp), sp), sp))  # −u / w²
            parts.append(_prod(coeff, dw, sp))
        return _sum(parts, e, ctx)
    if name == "hadamard":
        # The pointwise linear op: bilinear, so d(a⊙b) = da⊙b + a⊙db.
        a, b = args
        da, db = _dd(a, ctx), _dd(b, ctx)
        parts = []
        if not isinstance(da, Zero):
            parts.append(_prod(b, da, sp))
        if not isinstance(db, Zero):
            parts.append(_prod(a, db, sp))
        return _sum(parts, e, ctx)
    raise DiffError(
        f"diff: no derivative rule for external op {name!r}; supply ∂/∂x via deps or "
        f"extend the rule table"
    )


def _dd_scale(e: Scale, ctx: _DCtx) -> Term:
    """``d(r·v) = (dr)·v + r·(dv)`` — product rule (``r`` may be a constant or a scalar term)."""
    r, v = e.scalar, e.vector
    dv = _dd(v, ctx)
    parts: list[Term] = []
    if not isinstance(dv, Zero):
        parts.append(Scale(r, dv))
    if is_term(r):
        dr = _dd(r, ctx)
        if not isinstance(dr, Zero):
            parts.append(Scale(dr, v))
    return _sum(parts, e, ctx)


def _dd_multilinear_tuple(
    ctor: Callable[[tuple[Term, ...]], Term], vs: tuple[Term, ...], ctx: _DCtx, e: Term
) -> Term:
    """Apply the Leibniz rule to an n-ary multilinear tuple node (``OProdV``).

    Differentiate one factor at a time, keeping the others.
    """
    parts: list[Term] = []
    for k in range(len(vs)):
        dvk = _dd(vs[k], ctx)
        if isinstance(dvk, Zero):
            continue
        new = list(vs)
        new[k] = dvk
        parts.append(ctor(tuple(new)))
    return _sum(parts, e, ctx)


def _dd_eval(e: Eval, ctx: _DCtx) -> Term:
    """``d(f(args)) = (df)(args) + Σₖ f(…, dargₖ, …)``.

    The product rule over the map and every argument (the multilinear-form case).
    """
    f, pos, args = e.f, e.positions, e.args
    parts: list[Term] = []
    df = _dd(f, ctx)
    if not isinstance(df, Zero):
        parts.append(Eval(df, pos, args))
    for k in range(len(args)):
        dak = _dd(args[k], ctx)
        if isinstance(dak, Zero):
            continue
        new = list(args)
        new[k] = dak
        parts.append(Eval(f, pos, tuple(new)))
    return _sum(parts, e, ctx)


def _dd_compose(e: Compose, ctx: _DCtx) -> Term:
    """``d(f ∘ gs) = (df) ∘ gs + Σₖ f ∘ (…, dgₖ, …)`` — chain + product rule."""
    f, pos, gs = e.f, e.positions, e.gs
    parts: list[Term] = []
    df = _dd(f, ctx)
    if not isinstance(df, Zero):
        parts.append(Compose(df, pos, gs))
    for k in range(len(gs)):
        dgk = _dd(gs[k], ctx)
        if isinstance(dgk, Zero):
            continue
        new = list(gs)
        new[k] = dgk
        parts.append(Compose(f, pos, tuple(new)))
    return _sum(parts, e, ctx)


def _subspace_defining_map(space: object) -> Term | None:
    """Give the defining map ``f`` of a term-dependent subspace, else ``None``.

    Callers pass ``sub_or_J`` (a Space OR the map Term itself); a non-Space is simply not a
    subspace node — this canonical probe is Space-strict.
    """
    from grassmann.core.space import is_space
    from grassmann.infer.subspace import ffs_defining_map

    return ffs_defining_map(space) if is_space(space) else None


# The four fundamental subspaces (FFS) of a linear map ``J : dom ⊸ cod`` and the AMBIENT space each
# embeds into.  ``represent`` realizes each in a deterministic orthonormal frame:
#   Img(J)   ⊂ cod, range frame qr_pos(Jᵀ)        — δ = rank
#   CoImg(J) ⊂ dom, range frame SVD V[:,:δ]       — δ = rank
#   Ker(J)   ⊂ dom, complement frame SVD V[:,δ:]  — δ = dom−rank
#   CoKer(J) ⊂ cod, complement frame SVD U[:,δ:]  — δ = cod−rank
# where U,_,Vt = svd(Jᵀ = Jval.T, full_matrices=True); V = Vt.T.  ``_ffs_kind`` reads the FFS kind +
# ambient off the subspace node so the gauge machinery picks the matching frame.
_FFS_AMBIENT = {"img": "cod", "coimg": "dom", "ker": "dom", "coker": "cod"}


def _ffs_kind(sub: object) -> str | None:
    """Give the FFS kind string of a moving-subspace node, else ``None``.

    Tolerates a Term (callers pass ``sub_or_J``) — a Term has no FFS kind.
    """
    from grassmann.core.space import is_space
    from grassmann.infer.subspace import ffs_kind

    return ffs_kind(sub) if is_space(sub) else None


def _ffs_ambient(kind: str, J: Term) -> Space:
    """Give the ambient space (``dom`` or ``cod`` of ``J``) the FFS ``kind`` embeds into."""
    from grassmann.infer.synth import synth_space

    dom, cod = _map_dom_cod(synth_space(J))
    if dom is None:
        raise DiffError(f"diff: FFS gauge needs a simple linear-map defining map; got {synth_space(J)!r}")
    return dom if _FFS_AMBIENT[kind] == "dom" else cod


def _ffs_frame_value(kind: str, Jval: np.ndarray, rank: int | None = None) -> np.ndarray:
    """Give the ambient orthonormal frame columns of the FFS ``kind`` for immersion ``Jval``.

    ``Jval`` is in the grassmann ``(dom=k, cod=n)`` layout; the result has shape
    ``(ambient_dim, δ)`` — Img via QR, the other three via a full SVD of ``Jm = Jval.T``
    (cod×dom).  ``represent`` lowers exactly these frames (probed to machine precision).

    ``rank`` PINS the slice width.  Defaults to the numeric ``matrix_rank(Jm)``, but a moving NON-full
    rank subspace (e.g. a non-injective ``Img(J)``) must pass the DECLARED rank: the Taylor-reconstructed
    ``Jm`` drifts off the rank-``r`` manifold at finite δ, so its numeric rank jumps (spurious small
    singular values) and the slice width would change between FD steps — fixing ``rank`` keeps δ
    constant.
    """
    Jm = Jval.T  # ambient immersion columns, cod×dom
    if rank is None:
        rank = np.linalg.matrix_rank(Jm)
    dom = Jm.shape[1]
    if kind == "img":
        if rank == dom:
            Q_amb, _ = _qr_pos(Jm)  # cod × rank (= dom), range of Jm; injective ⇒ QR (byte-identical)
            return Q_amb
        # Non-injective J: ``represent`` realizes Img(J) in the SVD LEFT-singular frame U[:,:rank]
        # (cod×rank), NOT qr_pos (which would span all dom columns).  Probed == represent to machine
        # precision; qr_pos here would be a SILENTLY WRONG (dom-dim) frame.
        U, _s, _Vt = np.linalg.svd(Jm, full_matrices=True)  # U cod×cod
        return U[:, :rank]  # cod × rank
    U, _s, Vt = np.linalg.svd(Jm, full_matrices=True)  # U cod×cod, V=Vt.T dom×dom
    V = Vt.T
    if kind == "coimg":
        return V[:, :rank]      # dom × rank   (row space)
    if kind == "ker":
        return V[:, rank:]      # dom × (dom−rank)  (null space)
    if kind == "coker":
        return U[:, rank:]      # cod × (cod−rank)  (left null space)
    raise DiffError(f"diff: unknown FFS kind {kind!r}")


# ---------------------------------------------------------------------------
# Gauge of a moving subspace.  Differentiating a section of a moving subspace needs a
# *connection* (a gauge): the realization frame ``Q`` of ``Img(J)`` is defined only up to ``O(k)``,
# and the derivative depends on the choice.  The gauge is a property of the SPACE — keyed here by the
# defining map's uid — read by ``diff`` to pick the frame ``(Q, dQ)``.  ``represent`` is untouched: it
# realizes the subspace in its basis and cobs the value to any output basis, exactly as before.
#
# Gauge-awareness lives ENTIRELY in ``diff`` + this registry.  The ``immersion`` gauge
# (``Q = J``, ``dQ = ∂J``) is the literal immersion frame.  ``polar``
# (``Q = J·G^{-1/2}``, via the SPD ``Sqrt`` primitive) is the closed-form orthonormal gauge.  An
# opaque (e.g. QR) gauge supplies ``dQ`` as a program-backed variable.
# uid -> (gauge_name, frame, frame_deriv).  frame/frame_deriv are used only by OPAQUE gauges (qr/…).
_SUBSPACE_GAUGE: dict[int, tuple[str, Term | None, Term | None]] = {}
# The DEFAULT gauge is ``qr`` — the realization frame ``Q`` of ``Img(J)`` that ``represent``
# actually lowers (the orthonormal ``ProjectedBasis = qr_pos(Jᵀ)``).  So an unregistered moving
# subspace differentiates in the SAME frame its value lives in, with no per-test pinning.  The
# opaque ``dQ`` is supplied through auto-created program-backed variables, seeded by
# :func:`qr_gauge_feed` — ``represent`` stays untouched.
_DEFAULT_GAUGE = "qr"


def set_subspace_gauge(
    defining_map: Term, gauge: str, *, frame: Term | None = None, frame_deriv: Term | None = None,
) -> None:
    """Register the AD gauge for the moving subspace ``Img(defining_map)``.

    Read by ``diff`` ONLY; ``represent`` is unaffected.

    * ``"immersion"``, ``"polar"`` — closed-form frames (no extra args).
    * ``"qr"`` (the DEFAULT) — the orthonormal ``ProjectedBasis`` frame ``Q = qr_pos(Jᵀ)``.  With no
      explicit ``frame``/``frame_deriv`` the opaque ``Q`` / ``∂Q/∂x`` are AUTO-created as
      program-backed variables (see :func:`_qr_gauge_vars`); seed them with :func:`qr_gauge_feed`.
    * any other OPAQUE gauge — pass ``frame`` (the realization frame ``Q`` as a term/var) and
      ``frame_deriv`` (``∂Q/∂x`` as a map term/var); ``diff`` forms the directional
      ``dQ = (∂Q/∂x)·h`` exactly as it does for any marked variable.
    """
    _SUBSPACE_GAUGE[defining_map.uid] = (gauge, frame, frame_deriv)


def clear_subspace_gauges() -> None:
    """Reset all registered subspace gauges to the default and drop the auto-created QR vars."""
    _SUBSPACE_GAUGE.clear()
    _QR_AUTO_VARS.clear()
    _QR_FRAME_VARS.clear()
    _QR_NEXT.clear()


# --- opaque QR gauge: the default frame is the lowered ``ProjectedBasis`` ------------------------
# ``represent`` realizes a moving ``Img(J)`` in the orthonormal ``ProjectedBasis = qr_pos(Jᵀ)``.
# The matching AD gauge frame ``Q`` (and its motion ``∂ⁿQ/∂xⁿ``) has NO closed form the substrate can
# thread, so we supply it as opaque program-backed variables, auto-created on demand and seeded
# numerically by :func:`qr_gauge_feed`.  The frame is a SELF-DIFFERENTIATING TOWER
# ``Q₀ = Q, Q₁ = ∂Q/∂x, Q₂ = ∂²Q/∂x², …``: ``diff`` of any tower element yields the next one,
# so the opaque QR gauge supports ARBITRARY differentiation order.  Each ``Qₙ`` lives
# in ``_tower_space(X, Qₙ₋₁.space)`` — the flat ``(|X|,…,|X|,k,n)`` layout of the n-th derivative
# tensor, one axis per differentiation (see :func:`_tower_space`).  ``diff`` treats ``Qₙ`` like a
# marked-variable derivative: ``dQₙ = Eval(Qₙ₊₁, (0,), (h,))``.
#
# ``_QR_AUTO_VARS``: ``(J.uid, kind) -> Q₀`` (the realization frame; one per (defining map, FFS kind)
#   — ``Img(J)`` and ``Ker(J)`` of the SAME ``J`` need DIFFERENT frames, so the kind is part of the
#   key).
# ``_QR_FRAME_VARS``: the set of ALL tower-var uids (Q₀, Q₁, …) — :func:`_dd_var` reads it to make
#   every tower element self-differentiating.
# ``_QR_NEXT``: ``Qₙ.uid -> Qₙ₊₁`` (the derivative var of each tower element, created lazily).
_QR_AUTO_VARS: dict[tuple[int, str], Var] = {}
_QR_FRAME_VARS: set[int] = set()
_QR_NEXT: dict[int, Var] = {}


def _qr_next(e: Var, ctx: _DCtx) -> Var:
    """Get-or-create ``e``'s derivative tower var ``∂e/∂x``.

    Its space is ``_tower_space(X, e.space)`` — the FLAT ``(|X|, …, |X|, k, n)`` spelling this
    section documents.  The new var is registered in ``_QR_FRAME_VARS`` (so it too is
    self-differentiating) and in ``_QR_NEXT``.  Drives the arbitrary-order QR frame tower.

    Parameters
    ----------
    e
        The tower element to differentiate.
    ctx
        The differentiation context; ``ctx.X`` is the seed space.

    Returns
    -------
    Var
        ``e``'s successor in the tower — the same var on every call for a given ``e``.
    """
    if e.uid not in _QR_NEXT:
        nxt = Var(f"_dQqr_{e.name}_{next(_counter)}", _tower_space(ctx, e.space))
        _QR_NEXT[e.uid] = nxt
        _QR_FRAME_VARS.add(nxt.uid)
    return _QR_NEXT[e.uid]


def _qr_pos(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give the positive-diagonal (``R``) QR of ``M``.

    The unique factorization ``M = Q·R`` with ``diag(R) > 0`` (so ``Q`` is a deterministic
    orthonormal frame, == grassmann ``ProjectedBasis``).
    """
    q, r = np.linalg.qr(M)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s, r * s[:, None]


def _qr_dQ(Q: np.ndarray, R: np.ndarray, dJm: np.ndarray) -> np.ndarray:
    """Give the analytic QR frame differential.

    For ``Q,R = _qr_pos(Jm)`` (``Jm`` of shape ``(n,k)``) and a column perturbation ``dJm``
    (``(n,k)``), the motion of the orthonormal factor is

        M = Qᵀ·dJm·R⁻¹ ,   Ω = tril(M,−1) − tril(M,−1)ᵀ ,   dQ = Q·Ω + (I − Q·Qᵀ)·dJm·R⁻¹ .

    Verified against a central finite difference of :func:`_qr_pos` (square and rectangular,
    full-column-rank ``Jm``).
    """
    Rinv = np.linalg.inv(R)
    Mt = Q.T @ dJm @ Rinv
    low = np.tril(Mt, -1)
    omega = low - low.T
    n = Q.shape[0]
    return Q @ omega + (np.eye(n) - Q @ Q.T) @ dJm @ Rinv


def _qr_gauge_vars(J: Term, sub: Space, kind: str, ctx: _DCtx) -> Var:
    """Get-or-create the opaque realization frame ``Q₀`` of the moving FFS subspace ``sub``.

    Keyed by ``(J.uid, kind)`` for defining map ``J`` — ``Img(J)`` and ``Ker(J)`` of the SAME ``J``
    get SEPARATE frames.  ``Q₀`` lives in the grassmann frame-map space ``coords ⊸ ambient`` (stored
    ``(δ, ambient_dim)``); it is registered in ``_QR_FRAME_VARS`` so it is self-differentiating —
    ``_dd(Q₀)`` grows the tower ``Q₁ = ∂Q/∂x, …`` on demand.

    For ``Img`` the coordinate space IS ``J``'s domain, so ``Q₀ : J.space`` exactly reproduces the
    prior Img path (byte-identical).  For ``CoImg/Ker/CoKer`` the coordinate space is the moving
    subspace ITSELF, so ``Q₀ : oprod(dual(sub), ambient)`` — the roundtrip ``Q·(Q⁺·embed) == embed``
    lowers (probed).
    """
    from grassmann.core.space import dual, oprod
    from grassmann.infer.properties import is_injective
    from grassmann.infer.synth import synth_space

    key = (J.uid, kind)
    if key not in _QR_AUTO_VARS:
        if kind == "img" and is_injective(J):
            # INJECTIVE Img: coords = dom(J), frame qr_pos(Jᵀ) (dim = dom); keep the prior path identical.
            frame_space = synth_space(J)
        else:
            # NON-injective Img realizes in the SVD frame U[:,:rank] (dim = rank < dom), exactly like
            # the other three FFS kinds — route it through the same subspace-coord path (coords = sub).
            frame_space = oprod(dual(sub), _ffs_ambient(kind, J))  # coords = the moving subspace
        n = next(_counter)
        name = getattr(J, "name", "J")
        Qvar = Var(f"_Q{kind}_{name}_{n}", frame_space)
        _QR_AUTO_VARS[key] = Qvar
        _QR_FRAME_VARS.add(Qvar.uid)
    return _QR_AUTO_VARS[key]


# Per-order step for the nested Richardson-extrapolated central difference (:func:`_nth_deriv_tensor`,
# O(eps⁴) per level): large enough that the ``order``-fold cancellation (error ~ macheps/eps^order)
# stays bounded.  With Richardson, order 1 cross-checks the analytic ``_qr_dQ`` to <1e-7 and orders
# 2–3 reach ~1e-6 (a sharp 3rd-order FD check, not a smoke test).
_NESTED_FD_EPS = {1: 1e-4, 2: 4e-3, 3: 8e-3}


def _nth_deriv_tensor(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    order: int,
    eps: float | None = None,
) -> np.ndarray:
    """Give the ``order``-th derivative tensor of ``f : R^nX → array`` at ``x0``.

    Computed by NESTED, **Richardson-extrapolated** central differences over the ``nX``
    coordinate directions.  Each level uses ``(4·D(h/2) − D(h))/3`` (``D`` the central
    difference), cancelling the ``O(eps²)``
    truncation ⇒ ``O(eps⁴)`` per level — sharp enough that the fed ``∂ⁿQ`` tower is accurate, not
    smoke-accurate.  Returns shape ``(nX,)*order + f(x0).shape`` (axis ``aᵢ`` = the i-th direction);
    mixed partials are symmetric, so it is symmetric in its first ``order`` axes.
    """
    if order == 0:
        return np.asarray(f(x0), dtype=float)
    if eps is None:
        eps = _NESTED_FD_EPS.get(order, 1e-2)
    nX = x0.shape[0]

    def g(x: np.ndarray) -> np.ndarray:  # the (order-1) tensor — itself Richardson-extrapolated
        return _nth_deriv_tensor(f, x, order - 1, eps)

    blocks = []
    for a in range(nX):  # differentiate the (order-1) tensor once more in direction a (Richardson)
        ea = np.zeros(nX)
        ea[a] = 1.0
        d_h = (g(x0 + eps * ea) - g(x0 - eps * ea)) / (2.0 * eps)
        d_h2 = (g(x0 + (eps / 2.0) * ea) - g(x0 - (eps / 2.0) * ea)) / eps  # = /(2·eps/2)
        blocks.append((4.0 * d_h2 - d_h) / 3.0)
    return np.stack(blocks, axis=0)


def ffs_gauge_feed(
    sub_or_J: Space | Term, J_deriv_tower: Sequence[Var] | Var, feed: dict
) -> dict:
    """Build the numeric values of the auto-created opaque gauge TOWER of a moving FFS subspace.

    This is the default ``qr`` gauge.  Fills the ENTIRE realized frame tower
    ``Q₀ = Q, Q₁ = ∂Q/∂x, …`` to whatever depth ``diff`` grew it — so arbitrary-order FFS AD
    lowers.

    ``sub_or_J`` selects the case: pass the moving-subspace NODE (``Img(J)/CoImg(J)/Ker(J)/CoKer(J)``)
    to feed that exact frame; passing the bare defining map ``J`` is the back-compat ``Img`` alias
    (:func:`qr_gauge_feed`).  ``J_deriv_tower`` is the list of J's derivative VARS ``[DJ, D²J, …]`` the
    caller marked for the nested diffs (a bare ``DJ`` is accepted).

    Reads the immersion ``Jval`` (``feed[J]``, grassmann ``(k,n)``) and the derivative-tower values
    (``feed[DⁿJ]``, ``(|X|,…,|X|,k,n)``), reconstructs ``J(x₀+δ) = Σ_m (1/m!) ∂ᵐJ · δ^⊗m``, sets
    ``Q(x₀+δ) = _ffs_frame_value(kind, J(x₀+δ))ᵀ`` (QR for ``img``, the SVD U/V slice for the others),
    and takes each ``∂ⁿQ/∂xⁿ`` tensor at ``δ=0`` by NESTED Richardson central differences.  Returns
    ``{Qₙ: value}`` for every realized tower var, shaped ``(|X|,)*n + (δ, ambient)`` — the derivative
    tensor's own layout, fed through unreshaped, because the tower var's space is the FLAT
    ``_tower_space`` spelling.

    SVD ordering must be consistent across the tiny FD steps — pick a generic ``J`` with well-separated
    singular values (degenerate spectra rotate the frame discontinuously).  Call after ``diff`` so the
    tower vars exist; merge the result into the program feed.
    """
    if isinstance(J_deriv_tower, Term):
        J_deriv_tower = [J_deriv_tower]
    J = _subspace_defining_map(sub_or_J) if _ffs_kind(sub_or_J) is not None else sub_or_J
    kind = _ffs_kind(sub_or_J) or "img"
    Q0 = _QR_AUTO_VARS[(J.uid, kind)]
    Jval = np.asarray(feed[f"{J.name}_{J.uid}"], dtype=float)  # (k, n) grassmann
    # J-tower values in grassmann (k,n) codomain layout: tower[m] has shape (|X|,)*m + (k,n).
    Jtower = [Jval]
    for DmJ in J_deriv_tower:
        Jtower.append(np.asarray(feed[f"{DmJ.name}_{DmJ.uid}"], dtype=float))
    if len(Jtower) < 2:
        raise DiffError("ffs_gauge_feed: supply at least J's first derivative DJ")
    nX = Jtower[1].shape[0]
    # PIN the frame slice width at the base immersion's numeric rank.  A moving non-full-rank subspace
    # (non-injective Img) reconstructs OFF its rank-r manifold at finite δ, so the per-δ numeric rank
    # would jump and the slice width with it — fixing it here keeps the δ-dimension constant across the
    # nested FD steps so the ∂ⁿQ tower has a stable shape.
    base_rank = int(np.linalg.matrix_rank(Jval.T))

    def J_at(delta: np.ndarray) -> np.ndarray:
        """Reconstruct J at x₀+δ from the Taylor tower, grassmann (k, n) layout."""
        acc = np.array(Jtower[0])  # (k, n)
        for m in range(1, len(Jtower)):
            # ∂ᵐJ · δ^⊗m / m!  — contract the m leading |X| axes with δ.
            term = Jtower[m]
            for _ in range(m):
                term = np.tensordot(delta, term, axes=([0], [0]))
            acc = acc + term / float(math.factorial(m))
        return acc  # (k, n)

    def Q_at(delta: np.ndarray) -> np.ndarray:
        """Give the grassmann (δ, ambient) realization frame Q₀ at x₀+δ for this FFS case."""
        return _ffs_frame_value(kind, J_at(delta), base_rank).T  # frame (ambient, δ) ⇒ (δ, ambient)

    # Walk the realized tower Q₀ → Q₁ → … via _QR_NEXT; fill each to its order.
    out: dict = {}
    x0 = np.zeros(nX)
    var = Q0
    order = 0
    while True:
        depth = len(Jtower) - 1  # highest J-derivative available bounds reconstruction accuracy
        if order > depth:
            raise DiffError(
                f"ffs_gauge_feed: tower needs ∂^{order}Q but only {depth} J-derivative(s) supplied; "
                f"mark J→DJ→…→D^{order}J in the nested diffs and pass them all"
            )
        # shape (|X|,)*order + (δ, ambient) — already the FLAT tower layout, so it
        # goes in unreshaped; order 0 is the bare frame (δ, ambient).
        out[f"{var.name}_{var.uid}"] = _nth_deriv_tensor(Q_at, x0, order)
        nxt = _QR_NEXT.get(var.uid)
        if nxt is None:
            break
        var = nxt
        order += 1
    return out


def qr_gauge_feed(J: Term, J_deriv_tower: Sequence[Var] | Var, feed: dict) -> dict:
    """Fill the opaque ``ProjectedBasis`` (QR) frame tower of ``Img(J)`` — the default gauge.

    The ``Img`` alias of :func:`ffs_gauge_feed`.  ``J`` is the bare defining map; see
    :func:`ffs_gauge_feed` for the general FFS cases (pass the subspace node).
    """
    return ffs_gauge_feed(J, J_deriv_tower, feed)


def _gauge_frame(J: Term, dJ: Term, sub: Space, kind: str, ctx: _DCtx) -> tuple[Term, Term]:
    """Give the realization frame ``Q`` of the moving FFS subspace ``sub`` and its motion ``dQ``.

    ``kind`` is ``"img"/"coimg"/"ker"/"coker"``; the frame is the one the subspace's gauge names.
    The DEFAULT gauge is ``qr`` — the orthonormal frame ``represent`` actually lowers — supplied via
    AUTO-created opaque program-backed vars.  For ``Img`` two closed-form gauges also
    exist: ``immersion`` ⇒ ``(J, ∂J)`` (the literal immersion); ``polar`` ⇒ ``(J·G^{-1/2},
    ∂(J·G^{-1/2}))``, ``G = JᵀJ`` (the ``Sqrt`` primitive).  An OPAQUE gauge with an explicit
    ``frame``/``frame_deriv`` ⇒ ``(frame, (∂Q/∂x)·h)``.  ``CoImg/Ker/CoKer`` have no such immersion
    closed form (the frame is an SVD slice with no substrate-threadable derivative) — they go through
    the default opaque ``qr`` tower only.
    """
    from grassmann.infer.properties import is_injective

    entry = _SUBSPACE_GAUGE.get(getattr(J, "uid", None))
    gauge, frame, frame_deriv = entry if entry is not None else (_DEFAULT_GAUGE, None, None)
    if gauge in ("immersion", "polar") and kind != "img":
        raise DiffError(
            f"diff: the closed-form {gauge!r} gauge is defined only for Img(J); the {kind!r} "
            f"subspace realizes in an SVD frame — use the default qr gauge (ffs_gauge_feed)"
        )
    if gauge in ("immersion", "polar") and kind == "img" and not is_injective(J):
        # The closed immersion/polar frames assume an INJECTIVE immersion (Q = J or J·G^{-1/2}, full
        # column rank).  For a non-injective J, Img(J) realizes in the SVD frame U[:,:rank] — these
        # closed forms have no valid frame there.  Use the default qr gauge (ffs_gauge_feed).
        raise DiffError(
            f"diff: the closed-form {gauge!r} gauge is defined only for an INJECTIVE Img(J); a "
            f"non-injective J realizes Img(J) in the SVD frame — use the default qr gauge "
            f"(ffs_gauge_feed)"
        )
    if gauge == "immersion":
        return J, dJ
    if gauge == "polar":
        gram = Compose(Dagger(J), (0,), (J,))         # G = JᵀJ
        q = Compose(J, (0,), (Pinv(Sqrt(gram)),))     # Q = J·G^{-1/2}
        return q, _dd(q, ctx)
    if frame is not None and frame_deriv is not None:  # explicit opaque gauge (qr / external)
        return frame, Eval(frame_deriv, (0,), (ctx.h,))  # dQ = (∂Q/∂x)·h  (program-backed)
    if gauge == "qr":  # default opaque QR/SVD: auto-create the self-differentiating frame tower (Q₀,…)
        Qvar = _qr_gauge_vars(J, sub, kind, ctx)
        return Qvar, _dd(Qvar, ctx)  # dQ = Eval(Q₁, h) via the tower; higher diffs recurse
    raise DiffError(
        f"diff: gauge {gauge!r} has no closed form — supply frame= and frame_deriv= (the opaque "
        f"frame Q and ∂Q/∂x) via set_subspace_gauge"
    )


def _dd_embed(v: Term, big: Space, e: Term, ctx: _DCtx) -> Term:
    """``d(embed(v, big))`` — embed IS ``di``, the differential of the embedding.

    When ``v``'s home space ``Sub(J)`` has an x-dependent defining map ``J`` the derivative is

        d(embed(v)) = embed(dv)  +  dQ·(Q⁺·embed(v)) ,

    where ``Q`` is the **gauge frame** of the subspace and ``dQ`` its motion — the realization is the
    frame's directional derivative applied to ``v``'s coordinates ``Q⁺·embed(v)`` (recovered through
    the ambient).  The gauge is read from the subspace; the **default `qr` gauge** uses the opaque
    ``ProjectedBasis`` frame, while ``immersion`` (``Q = J``, ``dQ = ∂J``) reproduces
    ``embed(dv) + ∂J·(J⁺·embed v)`` exactly.  A constant ``J`` ⇒ the ordinary linear rule
    (``embed(dv)``).
    """
    from grassmann.infer.synth import synth_space

    dv = _dd(v, ctx)
    frozen = None if isinstance(dv, Zero) else Embed(dv, big)
    sub = synth_space(v)
    J = _subspace_defining_map(sub)
    if J is not None and _depends_on_x(J, ctx):
        dJ = _dd(J, ctx)
        if not isinstance(dJ, Zero):
            kind = _ffs_kind(sub)
            q, dq = _gauge_frame(J, dJ, sub, kind, ctx)  # gauge frame + its motion
            realiz = Eval(dq, (0,), (Eval(Pinv(q), (0,), (Embed(v, big),)),))
            return _sum([p for p in (frozen, realiz) if p is not None], e, ctx)
    return frozen if frozen is not None else _zero(e, ctx)


def _dd_project(z: Term, sub: Space, e: Term, ctx: _DCtx) -> Term:
    """``d(project(z, Sub(J)))`` — project IS ``di⁺``, the co-immersion / pseudoinverse.

    ``di⁺`` is the pseudoinverse of the gauge frame ``Q``.  When the target subspace ``Sub(J)`` has
    an x-dependent defining map ``J``,

        d(project(z)) = project(dz)  +  ∂(Q⁺)·z ,

    with ``∂(Q⁺)`` the pseudoinverse-map derivative (:func:`_pinv_map_grad`) of the **gauge frame**.
    The **default `qr` gauge** uses the opaque ``ProjectedBasis`` frame; ``immersion`` reproduces
    ``project(dz) + ∂(J⁺)·z`` exactly.
    """
    dz = _dd(z, ctx)
    frozen = None if isinstance(dz, Zero) else Project(dz, sub)
    J = _subspace_defining_map(sub)
    if J is not None and _depends_on_x(J, ctx):
        dJ = _dd(J, ctx)
        if not isinstance(dJ, Zero):
            kind = _ffs_kind(sub)
            q, dq = _gauge_frame(J, dJ, sub, kind, ctx)  # gauge frame + its motion (default: qr/SVD)
            # ``∂(Q⁺)·z`` lands in the frame's *coordinate* space ``dom(Q)``; the ``project(dz)`` arm
            # lives in ``sub`` (the ``Img(J)`` tag).  Re-tag the realization into ``sub`` value-faith-
            # fully by routing it through the frame: ``project(Q·(∂(Q⁺)·z), sub) = Q⁺ Q (∂(Q⁺)·z)``,
            # and ``Q⁺Q = I`` for an injective frame (immersion ``J`` is inj; qr/polar are orthonormal).
            realiz_coords = Eval(_pinv_map_grad(q, dq), (0,), (z,))  # : dom(Q)
            realiz = Project(Eval(q, (0,), (realiz_coords,)), sub)   # re-tag into ``sub``
            return _sum([p for p in (frozen, realiz) if p is not None], e, ctx)
    return frozen if frozen is not None else _zero(e, ctx)


def _dd_kron(e: Kron, ctx: _DCtx) -> Term:
    df, dg = _dd(e.f, ctx), _dd(e.g, ctx)
    parts: list[Term] = []
    if not isinstance(df, Zero):
        parts.append(Kron(df, e.g))
    if not isinstance(dg, Zero):
        parts.append(Kron(e.f, dg))
    return _sum(parts, e, ctx)


def _dd_logdet_kernel(f: Term, ctx: _DCtx) -> Term | None:
    """``tr(f⁻¹·df)`` — the shared scalar core of ``d det`` and ``d logdet`` (``f`` square).

    Returns ``None`` when ``f`` is x-independent (``df = 0``).  ``f⁻¹·df`` is the map ``Solve(f)``
    composed with the directional ``df`` (a value in ``V ⊸ V``); ``Trace(·, 1, 0)`` contracts its
    primal/dual slots — the standard ``trace`` convention for an endomorphism.
    """
    df = _dd(f, ctx)
    if isinstance(df, Zero):
        return None
    return Trace(Compose(Solve(f), (0,), (df,)), 1, 0)  # tr(f⁻¹·df), a Scalar


def _dd_det(e: Det, ctx: _DCtx) -> Term:
    """``d det(X)[dX] = det(X)·tr(X⁻¹·dX)`` (Jacobi's formula; ``X`` square + bij)."""
    tr = _dd_logdet_kernel(e.f, ctx)
    if tr is None:
        return _zero(e, ctx)
    return Scale(Det(e.f), tr)  # det(X) · tr(X⁻¹·dX), a Scalar


def _dd_logdet(e: LogDet, ctx: _DCtx) -> Term:
    """``d logdet(X)[dX] = tr(X⁻¹·dX)`` — the clean form (no ``det`` factor; ``X`` square + bij)."""
    tr = _dd_logdet_kernel(e.f, ctx)
    if tr is None:
        return _zero(e, ctx)
    return tr


def _dd_opaque(e: OpaqueApply, ctx: _DCtx) -> Term:
    """``d f(arg)[h] = Df(arg) · d arg[h]`` — the chain rule for an opaque map ``f : dom → cod``.

    ``Df(arg)`` is the Jacobian of ``f`` *at* ``arg`` — a linear map ``dom ⊸ cod`` the engine
    auto-allocates as a symbolic tracer input (the user never declares it).  It is fed numerically
    later; :func:`enumerate_derivative_inputs` surfaces it with ``source.name == op.name``.
    """
    darg = _dd(e.arg, ctx)  # directional derivative of the argument, a value in op.dom
    if isinstance(darg, Zero):
        return _zero(e, ctx)
    dF = _alloc_opaque_jacobian(e.op, e.arg, ctx)  # the Jacobian Df at arg: a map op.dom ⊸ op.cod
    return Eval(dF, (0,), (darg,))  # Df(arg) · d arg  ∈ op.cod


def _alloc_opaque_jacobian(op: OpaqueOp, arg: Term, ctx: _DCtx) -> Var:
    """Get-or-create the opaque map's Jacobian input ``Df`` at ``arg``.

    ``Df`` is a symbolic linear map ``op.dom ⊸ op.cod`` (NOT ``X ⊸ cod``; the Jacobian lives in
    the op's own spaces and is then composed with ``d arg`` by the chain rule).  Deduped by
    ``(op.name, arg)`` so repeated
    ``f(arg)`` shares one Jacobian input.  Provenance: a synthetic source ``Var(op.name, …)`` (the
    opaque function symbol) so :func:`enumerate_derivative_inputs` reports ``source.name == op.name``,
    order 1.
    """
    key = (op.name, arg)
    existing = _OPAQUE_JAC_BY_KEY.get(key)
    if existing is not None:
        return existing
    jspace = linear_map_space(op.dom, op.cod)
    source = Var(op.name, jspace)  # the opaque function symbol f
    dF = Var(f"D{op.name}_{next(_counter)}", jspace)
    _DERIV_INPUTS[dF.uid] = DerivInput(var=dF, source=source, order=1)
    _OPAQUE_JAC_BY_KEY[key] = dF
    return dF


def _dd_solve(e: Solve, ctx: _DCtx) -> Term:
    """``d(f⁻¹) = −f⁻¹ (df) f⁻¹`` — the bare inverse-map derivative (``f`` square).

    Built term→term from ``Solve``/``Compose``: the seed in ``df`` flows through the einsum
    ``Compose``s, NEVER the bulk ``SolveOp`` (which only ever sees the seed-free ``f``), so it
    threads in both llam modes — unlike the *applied* ``Solve(f)·v`` where the seed enters the
    bulk op directly.
    """
    df = _dd(e.f, ctx)
    if isinstance(df, Zero):
        return _zero(e, ctx)
    inv = Solve(e.f)
    return Scale(-1.0, Compose(inv, (0,), (Compose(df, (0,), (inv,)),)))


def _dd_pinv(e: Pinv, ctx: _DCtx) -> Term:
    """``d(f⁺)`` — the bare pseudoinverse-map derivative (Golub–Pereyra).

    ``d(f⁺) = −f⁺ (df) f⁺ + (fᵀf)⁻¹ (df)ᵀ (I − f f⁺)``. The rank term vanishes for a square map
    (``f f⁺ = I``), so the same formula is exact for square *and* injective (full-column-rank)
    ``f``. Pure ``Pinv``/``Dagger``/``Compose`` — the seed threads through the einsum ``Compose``s
    (the bulk ``PinvOp`` only sees the seed-free ``f``).  This is what makes ``project`` through a
    moving ``Img(J)`` differentiate term→term (``project ≡ Eval(Pinv(J), ·)``).
    """
    df = _dd(e.f, ctx)
    if isinstance(df, Zero):
        return _zero(e, ctx)
    return _pinv_map_grad(e.f, df)


def _dd_sqrt(e: Sqrt, ctx: _DCtx) -> Term:
    """``d(√f)`` for an SPD endomorphism ``f : V ⊸ V``.

    The symmetric operator-square-root derivative.  ``dS`` (``S = √f``) is the unique solution of
    the Sylvester equation

        S·dS + dS·S = df ,

    which we solve term→term as the inverse of the Sylvester operator built on the **map space**
    ``(V⊸V) ⊸ (V⊸V)`` (NOT a fused ``V⊗V`` covector — that route can't un-fuse back to a map):

        L = Kron(Sᵈ, I_V) + Kron(I_{V*}, S)        (left-mult ``S·`` + right-mult ``·S``),

    with ``Sᵈ = ♭∘S∘♯ : V* ⊸ V*`` the action of ``S`` on covariant indices.  Then
    ``dS = Pinv(L)·df`` lands directly in ``V⊸V``.  ``L`` is SPD (eigenvalues ``√λᵢ + √λⱼ > 0``),
    and the seed in ``df`` enters only the final ``Eval`` einsum — ``L`` itself is seed-free.
    """
    from grassmann.core.space import dual
    from grassmann.infer.synth import synth_space

    df = _dd(e.f, ctx)
    if isinstance(df, Zero):
        return _zero(e, ctx)
    _dom, cod = _map_dom_cod(synth_space(e.f))  # f : V ⊸ V ⇒ V = cod (= dom)
    if cod is None:
        raise DiffError(f"diff: sqrt derivative needs a simple map space; got {synth_space(e.f)!r}")
    v = cod
    s = e  # Sqrt(e.f) : V ⊸ V
    # The value lowers sqrt(sym(f)) (``_SqrtSpdOp`` symmetrizes), so the honest derivative sees the
    # SYMMETRIC part of df — the antisymmetric part doesn't move sqrt(sym(f)).  No-op for the
    # immersed-metric use (``f = JᵀJ`` ⇒ df = dJᵀJ + JᵀdJ already symmetric); guards the rest.
    df_sym = Scale(0.5, Sum((df, Dagger(df))))
    s_dual = Compose(Flat(v), (0,), (Compose(s, (0,), (Sharp(v),)),))  # V* ⊸ V*, S on covectors
    syl = Sum((Kron(s_dual, Identity(v)), Kron(Identity(dual(v)), s)))  # (V⊸V) ⊸ (V⊸V)
    return Eval(Pinv(syl), (0,), (df_sym,))  # solve  syl · dS = df_sym  for dS : V ⊸ V


def _pinv_map_grad(f: Term, df: Term) -> Term:
    """Give the bare pseudoinverse-map derivative ``d(f⁺)`` (see :func:`_dd_pinv`)."""
    from grassmann.infer.synth import synth_space

    fp = Pinv(f)
    lead = Scale(-1.0, Compose(fp, (0,), (Compose(df, (0,), (fp,)),)))  # −f⁺ (df) f⁺
    dom, cod = _map_dom_cod(synth_space(f))
    if dom is not None and dom == cod:
        return lead  # square ⇒ I − f f⁺ = 0 ⇒ no rank term
    gram = Compose(Dagger(f), (0,), (f,))  # fᵀf : dom ⊸ dom
    im_proj = Sum((Identity(cod), Scale(-1.0, Compose(f, (0,), (fp,)))))  # I − f f⁺ : cod ⊸ cod
    rank = Compose(Pinv(gram), (0,), (Compose(Dagger(df), (0,), (im_proj,)),))
    return Sum((lead, rank))


def _map_dom_cod(space: Space) -> tuple[Space, Space] | tuple[None, None]:
    """Give ``(dom, cod)`` of a simple linear-map space ``OProd(Dual(dom), cod)``.

    Anything else gives ``(None, None)``.  The canonical strict peel: tolerating a non-``Dual``
    head would silently read a (2,0)-tensor as a map — the implicit-conversion class grassmann
    bans.
    """
    from grassmann.infer.subspace import split_map_space

    split = split_map_space(space)
    return split if split is not None else (None, None)


def _dd_solve_apply(e: SolveApply, ctx: _DCtx) -> Term:
    """``d(f⁻¹ v) = f⁻¹(dv) − f⁻¹ (df)(f⁻¹ v)``."""
    return _dd_inverse_apply(e, e.f, e.v, SolveApply, ctx)


def _dd_pinv_apply(e: PinvApply, ctx: _DCtx) -> Term:
    """``d(f† v) = f†(dv) − f† (df)(f† v)`` (leading term; square-bijective exact)."""
    return _dd_inverse_apply(e, e.f, e.v, PinvApply, ctx)


def _dd_inverse_apply(
    e: Term, f: Term, v: Term, apply_inv: Callable[[Term, Term], Term], ctx: _DCtx
) -> Term:
    """Apply the shared ``d((inv f) v)`` rule for ``Solve``/``Pinv``.

    The ``dv`` term is built with the **applied** inverse node
    (``apply_inv`` = ``SolveApply``/``PinvApply``) so it inverts ``f`` in
    the *same* operator convention as the primal ``e`` — routing through ``Eval(inv(f), …)``
    instead silently applies the *transpose* inverse (``Eval(f, ·)`` contracts the dom slot, so
    ``Eval(Solve(f), ·) = f⁻ᵀ·`` whereas ``SolveApply(f, ·) = f⁻¹·``).  Differentiating w.r.t. the
    inverted map itself (``df ≠ 0``) yields a higher tensor the substrate cannot yet thread
    through the bulk op — that case is rejected.
    """
    parts: list[Term] = []
    dv = _dd(v, ctx)
    if not isinstance(dv, Zero):
        parts.append(apply_inv(f, dv))
    df = _dd(f, ctx)
    if not isinstance(df, Zero):
        raise DiffError(
            f"diff: differentiating {type(e).__name__} w.r.t. the inverted map is not "
            f"supported in v1 (the seed would flow through the bulk inverse op); only the "
            f"applied vector may depend on x here"
        )
    return _sum(parts, e, ctx)


def _dd_neval_lam(e: NEval, lam: Lam, positions: tuple[int, ...], args: tuple, ctx: _DCtx) -> Term:
    """Apply the value-``Lam`` chain rule.

    ``NEval(Lam([yₖ], body), pos, args)`` is ``body[yₖ := args]``; its directional derivative
    differentiates the *body* with each bound var
    ``yₖ`` carrying the directional derivative of its argument, then re-binds the args:

        d(NEval(Lam([yₖ], body), pos, args)) = NEval(Lam([yₖ], d_body), pos, args)

    where ``d_body = _dd(body)`` in a context that sets ``∂yₖ/∂x · h = _dd(arg_k)`` (so a bound-var
    occurrence in the body contributes its argument's derivative — the chain rule).  The original
    ``yₖ`` occurrences that survive ``_dd``'s product rule are re-bound to ``args`` by the wrapping
    ``NEval``; the substituted derivatives reference only outer vars (``args`` cannot mention the
    binder), so the re-binding leaves them untouched.

    Only a **value**-producing ``Lam`` is differentiable: a Space/Type body (a SpaceFunction) or a
    space/nat binder/argument is genuinely higher-order and stays a :class:`DiffError`.
    """
    from grassmann.core.kinds import SpaceSort

    if not is_term(lam.body):
        raise DiffError("diff: cannot differentiate a Lam with a non-value (Space/Type/Nat) body")
    if any(not isinstance(sort, SpaceSort) for _n, sort in lam.binders):
        raise DiffError("diff: cannot differentiate a Lam with a Space/Nat binder (higher-order)")
    if any(not is_term(a) for a in args):
        raise DiffError("diff: cannot differentiate an NEval that applies a Space/Nat argument")
    value_slots = [k for k, (_n, sort) in enumerate(lam.binders) if isinstance(sort, SpaceSort)]
    slot_to_var = dict(zip(value_slots, lam.term_vars, strict=False))
    arg_at_slot = dict(zip(positions, args, strict=True))
    if any(v.uid == ctx.x.uid for v in slot_to_var.values()):
        raise DiffError("diff: the differentiation variable is bound by a Lam in the term")
    # Directional derivative of each argument (in the OUTER context — args are outer terms).
    darg: dict[int, Term] = {}
    for slot, yk in slot_to_var.items():
        if slot not in arg_at_slot:
            raise DiffError(f"diff: NEval value slot {slot} of the Lam has no supplied argument")
        darg[yk.uid] = _dd(arg_at_slot[slot], ctx)
    new_deps = dict(ctx.deps)
    new_ddval = dict(ctx._ddval)  # after computing darg ⇒ shares any outer None-dep allocations
    for yk in slot_to_var.values():
        new_deps[yk.uid] = None  # marked dependent; its value is supplied via _ddval (below)
        new_ddval[yk.uid] = darg[yk.uid]
    inner = _DCtx(x=ctx.x, h=ctx.h, deps=new_deps, _ddval=new_ddval)
    dbody = _dd(lam.body, inner)
    if isinstance(dbody, Zero):
        return _zero(e, ctx)
    return NEval(Lam(lam.binders, dbody, lam.term_vars), positions, args)


def _dd_llam(e: LLam, vars_: tuple[Var, ...], body: Term, ctx: _DCtx) -> Term:
    """Differentiate a (multi)linear map value: differentiate the body, keep the binder.

    ``x`` must not be one of the linear vars (those are not the differentiation variable).
    """
    if any(v.uid == ctx.x.uid for v in vars_):
        raise DiffError("diff: the differentiation variable is bound by an LLam in the term")
    dbody = _dd(body, ctx)
    if isinstance(dbody, Zero):
        return _zero(e, ctx)
    return LLam(vars_, dbody)


# ===========================================================================
# Covariant correction (the metric connection). Flat / constant metric ⇒ 0.
# ===========================================================================


def _covariant_section(
    e: Term, ctx: _DCtx
) -> tuple[Term, Space, Callable[[Var], Term]] | None:
    """Recognize a covariant section over an x-dependent metric, giving its connection builder.

    If ``e`` is a covariant *section* ``Remetric(sec, g)`` over a manifold ``M`` with an
    **x-dependent** metric, return ``(sec, M, conn)`` where ``conn(h)`` builds the connection map
    ``Γ(h) : M ⊸ M``; otherwise ``None``.

    Two metric sources are recognized:

    * a marked metric **Var** ``g`` (with gradient ``Dg`` via ``deps``) ⇒ the Levi-Civita
      connection from ``∂g`` (:func:`_connection_map`);
    * a **chart-induced** metric ``g = JᵀJ`` with ``J`` a marked chart immersion ⇒ the
      frame connection ``Γ(h) = J⁺·(∂J·h)`` (:func:`_frame_connection_map`), which needs no
      rank-3 metric gradient.

    A constant metric (``Dg``/``J`` not x-dependent) is *not* covariant — the plain total
    derivative is correct — so it falls through to the ordinary path.
    """
    from functools import partial

    from grassmann.infer.synth import synth_space

    if not isinstance(e, Remetric) or not is_term(e.metric):
        return None
    sec, g = e.vector, e.metric
    M = _manifold_of(synth_space(sec))
    if M is None:
        return None
    frame = _chart_frame_of(g, ctx)
    if frame is not None:  # chart immersion g = JᵀJ with J marked ⇒ frame connection
        J, DJ = frame
        conn = partial(_frame_connection_map, M, J, DJ)
    else:
        Dg = _metric_gradient(g, M, ctx)
        if Dg is None:  # constant / x-independent metric ⇒ flat ⇒ ordinary derivative
            return None
        conn = partial(_connection_map, M, g, Dg)
    if _strip_space(ctx.X) != M:
        raise DiffError(
            "diff: the metric connection requires the differentiation variable to live in "
            "the tangent space (X == the manifold base space M)"
        )
    return (sec, M, conn)


def _chart_jacobian_var(g: Term) -> Var | None:
    """Give the chart Jacobian ``J`` of a chart-induced metric ``JᵀJ``, else ``None``.

    Recognized shapes: ``Compose(Dagger(J),(0,),(J,))`` or ``Compose(J,(0,),(Dagger(J),))`` for a
    single map ``Var J``.
    """
    if isinstance(g, Compose) and len(g.gs) == 1:
        f, gg = g.f, g.gs[0]
        if isinstance(f, Dagger) and isinstance(f.f, Var) and gg == f.f:
            return f.f
        if isinstance(gg, Dagger) and isinstance(gg.f, Var) and f == gg.f:  # audit: ok EX-ISINSTANCE-LADDER — narrowing: one JtJ Compose/Dagger pattern, else None [approved 2026-07-03]
            return f
    return None


def _chart_frame_of(g: Term, ctx: _DCtx) -> tuple[Var, Term] | None:
    """Give ``(J, DJ)`` for a chart metric ``JᵀJ`` over an **x-dependent** marked immersion ``J``.

    ``DJ`` is the frame's derivative map ``∂J/∂x : X ⊸ (M ⊸ A)`` (from ``deps``, or a
    fresh symbolic map var for a ``None`` dep). ``None`` if ``g`` is not a chart metric or ``J``
    is x-independent (a constant chart ⇒ flat).
    """
    J = _chart_jacobian_var(g)
    if J is None or J.uid not in ctx.deps:
        return None
    dval = ctx.deps[J.uid]
    if dval is not None:
        return (J, dval)
    return (J, _alloc_deriv_input(J, ctx))  # provenance-tracked ∂J/∂x input


def _manifold_of(s: Space) -> Space | None:
    """Give the base manifold ``M`` of a (co)tangent (tensor) space, else ``None``.

    Recognized: ``M``, ``M*``, or a tensor product whose factors are all ``M`` / ``M*`` over a
    single ``M``.
    """
    from grassmann.core.space import Dual, OProd

    def base(f: Space) -> Space:
        return f.space if isinstance(f, Dual) else f

    if isinstance(s, OProd):
        bases = {base(f) for f in s.spaces}
        return next(iter(bases)) if len(bases) == 1 else None
    return base(s)


def _connection_correction(
    sec: Term, sspace: Space, M: Space, conn_builder: Callable[[Var], Term], ctx: _DCtx
) -> Term:
    """Build the Levi-Civita correction ``∇sec − d(sec)`` in direction ``h``, **basis-free**.

    The tensor-product/Leibniz rule: apply the connection map ``Γ(h) : M ⊸ M`` (built by
    ``conn_builder(h)`` — metric or frame source) to each slot of the section — ``+Γ`` on a
    contravariant (``M``) factor, ``−Γ`` on a covariant (``M*``) factor:

    - vector ``v : M``        →  ``Γ(h)·v``                       (Eval)
    - covector ``ω : M*``     →  ``−ωᵀ·Γ(h)``                     (precompose)
    - ``(1,1)`` tensor ``T : M ⊸ M`` → ``Γ(h)∘T − T∘Γ(h)``       (the commutator)

    Everything is built from ``Eval``/``Compose`` of the bare connection map, so no metric
    basis is ever realized (the lowering stays in standard coordinates).
    """
    from grassmann.core.space import OProd, dual

    conn = conn_builder(ctx.h)  # Γ(h) : M ⊸ M, bare
    factors = list(sspace.spaces) if isinstance(sspace, OProd) else [sspace]
    M_dual = dual(M)

    # Single contravariant slot (a tangent vector v : M): +Γ·v.
    if len(factors) == 1 and factors[0] == M:
        return Eval(conn, (0,), (sec,))

    # All-covariant tensor (covector, (0,p) form, the metric g itself): −Γ on every lower
    # index. Each covariant slot is a dual that `Compose` contracts the connection map into.
    if all(f == M_dual for f in factors):
        return _sum(
            [Scale(-1.0, Compose(sec, (a,), (conn,))) for a in range(len(factors))],
            sec, ctx,
        )

    # (1,1) tensor T : M* ⊗ M (a map M ⊸ M): +Γ on the M (cod), −Γ on the M* (dom).
    if len(factors) == 2 and factors[0] == M_dual and factors[1] == M:
        return Sum((Compose(conn, (0,), (sec,)), Scale(-1.0, Compose(sec, (0,), (conn,)))))

    raise DiffError(
        f"diff: covariant differentiation of {sspace!r} is not yet supported. A pure "
        f"contravariant slot in a non-map tensor ((2,0) etc.) needs per-slot application "
        f"that hits the nested-⊗ codomain issue. Supported: vector, "
        f"covector, all-covariant forms (incl. the metric, ∇g=0), and (1,1) maps."
    )


def _connection_map(M: Space, g: Term, Dg: Term, h: Var) -> Term:
    """Build the Levi-Civita connection map ``Γ(h) : M ⊸ M`` — **basis-free**.

    ``Γ(h)^k_m = g^{kl}·½(∂ₘg_{il} + ∂ᵢg_{ml} − ∂ₗg_{mi}) hⁱ``. Built as ``g⁻¹ ∘ Glower``,
    where ``Glower : M ⊸ M*`` is the first-kind symbol contracted with ``h`` (the ``Permute``
    supplies the one transpose the reshape-only ``Lift`` cannot). The inverse metric is the
    **bare** map ``Pinv(g)`` — ``g`` decodes as the flat map ``M ⊸ M*`` — so the connection
    carries no metric basis tag and the whole derivative stays coordinate-free.
    """
    t1 = Eval(Dg, (0,), (h,))  # ∂ᵢg_{jl} hⁱ, indexed (j, l)
    t2 = Eval(Dg, (1,), (h,))  # ∂ⱼg_{il} hⁱ, indexed (j, l)
    g_lower_map = Scale(0.5, Sum((t1, t2, Scale(-1.0, Permute(t2, (1, 0))))))  # M ⊸ M*
    return Compose(Pinv(g), (0,), (g_lower_map,))  # raise the cod index with g⁻¹ ⇒ M ⊸ M


def _frame_connection_map(M: Space, J: Term, DJ: Term, h: Var) -> Term:
    """Build the Levi-Civita connection of a **chart immersion** ``J : M ⊸ A`` (a moving frame).

    In direction ``h`` it is ``Γ(h) = J⁺·(∂J·h) : M ⊸ M``.

    For the induced (pullback) metric ``g = JᵀJ`` the Christoffel symbols are
    ``Γ^k_{mi} = (J⁺ ∂_i J)^k_m``, so ``Γ(h) = Pinv(J) ∘ Eval(DJ,(0,),h)`` — built from the bare
    pseudo-inverse and the **rank-2** directional derivative of the frame, with no rank-3 metric
    gradient (sidestepping the nested-⊗ substrate gap that ``∂(JᵀJ)`` would hit).  Numerically
    identical to :func:`_connection_map` fed ``g = JᵀJ``, ``Dg = ∂(JᵀJ)`` (verified), so it plugs
    into the same per-slot :func:`_connection_correction`.  Assumes a true chart Jacobian
    (``J = ∂Φ`` ⇒ ``∂J`` symmetric in direction ↔ tangent slot ⇒ a torsion-free connection).
    """
    return Compose(Pinv(J), (0,), (Eval(DJ, (0,), (h,)),))  # J⁺ ∘ (∂J·h) : M ⊸ M


def _metric_gradient(g: Term, M: Space, ctx: _DCtx) -> Term | None:
    """Give the metric gradient ``Dg : M ⊸ (M*⊗M*)`` (``∂g/∂x``) as a *primitive* map.

    ``None`` when the metric does not depend on ``x`` (a constant/flat metric).

    The metric must be a marked-dependent ``Var`` (its gradient supplied via ``deps``,
    or a fresh symbolic map var for ``None``); a constant metric ``Var`` (absent from
    ``deps``) yields ``None`` (the flat case). A more complex x-dependent metric expression
    is rejected — supply the metric (and its derivative) as a marked variable.
    """
    if isinstance(g, Var):
        if g.uid not in ctx.deps:
            return None  # metric does not depend on x ⇒ flat
        dval = ctx.deps[g.uid]
        if dval is not None:
            return dval
        from grassmann.core.space import dual, oprod

        cod = oprod(dual(M), dual(M))
        return _alloc_deriv_input(g, ctx, space=cod)  # provenance-tracked ∂g/∂x input
    if _depends_on_x(g, ctx):
        raise DiffError(
            "diff: x-dependent metric must be a marked variable so its gradient is known; "
            f"got a composite metric {type(g).__name__}"
        )
    return None


def _strip_space(s: Space) -> Space:
    return s.space if isinstance(s, WithInner) else s


def _depends_on_x(node: object, ctx: _DCtx) -> bool:
    if isinstance(node, Var):
        return node.uid == ctx.x.uid or node.uid in ctx.deps
    return any(_depends_on_x(c, ctx) for c in _children(node))


def _xdep_metric_space(space: Space, ctx: _DCtx) -> bool:
    """Report whether ``space`` carries a ``WithInner`` metric that depends on ``x``.

    Such a metric makes a musical operator of that space itself x-dependent.
    """
    return (
        isinstance(space, WithInner)
        and is_term(space.metric)
        and _depends_on_x(space.metric, ctx)
    )


# ===========================================================================
# Helpers.
# ===========================================================================


def _zero(e: Term, ctx: _DCtx) -> Zero:
    from grassmann.infer.synth import synth_space

    return Zero(synth_space(e))


def _sum(parts: list[Term], e: Term, ctx: _DCtx) -> Term:
    live = [p for p in parts if not isinstance(p, Zero)]
    if not live:
        return _zero(e, ctx)
    if len(live) == 1:
        return live[0]
    return Sum(tuple(live))


def _add(a: Term, b: Term, space: Space) -> Term:
    if isinstance(a, Zero):
        return b
    if isinstance(b, Zero):
        return a
    return Sum((a, b))


def _wrap_linear(ctor: Callable[[Term], Term], dv: Term, e: Term, ctx: _DCtx) -> Term:
    """Differentiate a node linear in one child — the same node over the child's derivative.

    Gives ``Zero`` when the child is x-independent.
    """
    if isinstance(dv, Zero):
        return _zero(e, ctx)
    return ctor(dv)


# --- scalar op builders (the nonlinear-rule coefficients) -------------------


def _ext(name: str, fn: Callable[..., np.ndarray], space: Space, *args: Term) -> External:
    from grassmann.lower.represent import ExternalOp

    return External(space, ExternalOp(name, fn), tuple(args))


def _sqrt(u: Term, space: Space = Scalar()) -> External:
    return _ext("sqrt", np.sqrt, space, u)


def _recip(u: Term, space: Space = Scalar()) -> External:
    return _ext("reciprocal", lambda a: np.divide(1.0, a), space, u)


def _sign(u: Term, space: Space = Scalar()) -> External:
    return _ext("sign", np.sign, space, u)


def _exp(u: Term, space: Space = Scalar()) -> External:
    return _ext("exp", np.exp, space, u)


def _sin(u: Term, space: Space = Scalar()) -> External:
    return _ext("sin", np.sin, space, u)


def _cos(u: Term, space: Space = Scalar()) -> External:
    return _ext("cos", np.cos, space, u)


def _sinh(u: Term, space: Space = Scalar()) -> External:
    return _ext("sinh", np.sinh, space, u)


def _cosh(u: Term, space: Space = Scalar()) -> External:
    return _ext("cosh", np.cosh, space, u)


def _sigmoid(u: Term, space: Space = Scalar()) -> External:
    from scipy.special import expit

    return _ext("sigmoid", expit, space, u)


def _ones(u: Term, space: Space) -> External:
    """Build a constant value of ``1`` in ``space``, shaped like ``u`` (elementwise ``ones_like``).

    The constant-scalar leaf ``Lit`` lowers inconsistently for in-context scalars (the
    ``ScalarBasis≅ℝ¹`` cob seam), so derivative arms that need ``1±g(u)`` build the unit
    constant from the operand's own shape — which also broadcasts to vectors/matrices for
    free. Its derivative is zero (the ``"ones"`` arm in :func:`_dd_external`).
    """
    return _ext("ones", lambda a: np.ones_like(np.asarray(a, dtype=float)), space, u)


def _one_minus_sq(u: Term, space: Space) -> Sum:
    """``1 − u²`` in ``space`` — the asin/acos denominator radicand."""
    return Sum((_ones(u, space), Scale(-1.0, _prod(u, u, space))))


def _hadamard(a: Term, b: Term, space: Space) -> External:
    """Build the pointwise (Hadamard) product ``a ⊙ b`` in ``space``."""
    return _ext("hadamard", np.multiply, space, a, b)


def _prod(a: Term, b: Term, space: Space) -> Term:
    """Elementwise product of two same-``space`` values.

    For a ``Scalar`` it is an ordinary ``Scale`` (scalar·scalar) — byte-identical to the
    pre-Hadamard arms; for any other (Euclidean) space it is the Hadamard product, so the
    elementwise chain rule ``f'(u) ⊙ du`` works on vectors/matrices, not just scalars.
    """
    if isinstance(space, Scalar):
        return Scale(a, b)
    return _hadamard(a, b, space)


# --- well-formedness guard --------------------------------------------------


def _guard_closed_and_first_order(e: Term, x: Var, deps: dict[int, Term | None]) -> None:
    """Reject an open term, a ``Frame`` on the path, or a non-subspace x-dependent space position.

    That is: a free space variable / unresolved metavar, a ``Frame``, or an x-dependent term in a
    *non-subspace* space position (a genuine space-function dependence).

    An x-dependent term in a **subspace** (``Img(J)/CoImg/Ker/CoKer`` with ``J`` x-dependent) is NOT
    rejected: ``embed`` IS ``di`` (the differential of the embedding), so the realizing arms
    differentiate it term→term — ``embed``/``project`` add the immersion-derivative term, and the
    musical maps raise their own specific error.  Only x flowing into a non-subspace space (an SVar
    / a space made to depend on x by a Frame/SpaceFunction) is genuinely higher-order.
    """
    from grassmann.core.space import is_space
    from grassmann.infer.subspace import is_ffs
    from grassmann.infer.synth import InferError, synth_space

    try:
        s = synth_space(e)
    except InferError as exc:
        raise DiffError(f"diff: term does not type-check / is not closed ({exc})") from exc
    if _mentions_space_meta(s):
        raise DiffError("diff: term has an unresolved space; it must be closed")

    def is_xdep(v: Var) -> bool:
        return v.uid == x.uid or v.uid in deps

    def walk(node: object, in_space: bool, in_subspace: bool) -> None:
        if isinstance(node, Frame):
            raise DiffError("diff: cannot differentiate through a Frame (basis binder)")
        if in_space and not in_subspace and isinstance(node, Var) and is_xdep(node):  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): Frame/x-dep rejections on generic recursion [approved 2026-07-03]
            raise DiffError(
                "diff: an x-dependent term flows into a non-subspace space position "
                "(a space made to depend on x — e.g. a SpaceFunction); differentiating through a "
                "space function is unsupported."
            )
        for child in _children(node):
            walk(child, in_space or is_space(child),
                 in_subspace or (is_space(child) and is_ffs(child)))

    walk(e, is_space(e), False)


def _mentions_space_meta(s: Space) -> bool:
    from grassmann.core.space import SpaceMeta, SVar

    found = False

    def walk(node: object) -> None:
        nonlocal found
        if isinstance(node, (SpaceMeta, SVar)):  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): meta/SVar leaf detect else generic recursion [approved 2026-07-03]
            found = True
            return
        for child in _children(node):
            walk(child)

    walk(s)
    return found


def _children(node: object) -> tuple[object, ...]:
    out: list[object] = []
    for fname in getattr(node, "__dataclass_fields__", {}):
        val = getattr(node, fname)
        items = val if isinstance(val, tuple) else (val,)
        for c in items:
            if is_term(c) or is_space(c):
                out.append(c)
    return tuple(out)


__all__ = [
    "diff",
    "DiffError",
    "set_subspace_gauge",
    "clear_subspace_gauges",
    "ffs_gauge_feed",
    "qr_gauge_feed",
]
