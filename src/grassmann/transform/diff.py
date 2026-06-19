"""Total covariant differentiation — symbolic forward-mode (`§50`).

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

import itertools
from dataclasses import dataclass, field

import numpy as np

from grassmann.core.space import Scalar, Space, WithInner, is_space, linear_map_space
from grassmann.core.term import (
    Assert,
    Compose,
    Dagger,
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
    NEval,
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
    Sum,
    Term,
    Trace,
    TupleEmbed,
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


def diff(e: Term, x: Var, deps: dict[int, Term | None] | None = None) -> Term:
    """Total covariant derivative ``d e / d x : X ⊸ space(e)`` (the public entry point)."""
    if not is_term(e):
        raise DiffError(f"diff: expected a Term, got {e!r}")
    if not isinstance(x, Var):
        raise DiffError(f"diff: the differentiation variable must be a Var, got {x!r}")
    _guard_closed_and_first_order(e, x)

    from grassmann.infer.synth import synth_space

    R = synth_space(e)
    h = Var(f"d{x.name}", x.space)
    ctx = _DCtx(x=x, h=h, deps=deps or {})
    cov = _covariant_section(e, ctx)
    if cov is not None:
        # A covariant section ``Remetric(sec, g)`` over a manifold with an x-dependent
        # metric: ∇sec = d(sec) + Γ(h)·sec, computed entirely in the bare (coordinate-free)
        # basis — the derivative never realizes a metric basis.
        sec, M, g, Dg = cov
        sspace = synth_space(sec)
        body = _add(_dd(sec, ctx), _connection_correction(sec, sspace, M, g, Dg, ctx), sspace)
        rspace = sspace
    else:
        body = _dd(e, ctx)
        rspace = R
    if isinstance(body, Zero):
        # e is x-independent: the derivative is the zero map (an LLam over an unused seed
        # would violate the linear discipline, so emit the zero map directly).
        return Zero(linear_map_space(ctx.X, rspace))
    return LLam((h,), body)


# ===========================================================================
# Directional derivative — De[h], a VALUE in space(e), linear in h.
# ===========================================================================


def _dd(e: Term, ctx: _DCtx) -> Term:
    """The directional derivative of ``e`` in direction ``ctx.h`` — a value in
    ``space(e)``."""
    match e:
        case Var():
            return _dd_var(e, ctx)
        case Zero() | Lit() | Identity() | TupleEmbed() | TupleProject():
            return _zero(e, ctx)
        case Inner() | Flat() | Sharp():
            # Musical maps of a *constant* space are constant; an x-dependent metric is
            # handled by the covariant correction, not by differentiating the operator.
            return _zero(e, ctx)
        case External():
            return _dd_external(e, ctx)
        case Scale():
            return _dd_scale(e, ctx)
        case Sum(vs):
            return _sum([_dd(v, ctx) for v in vs], e, ctx)
        case OPlusV(vs):
            return OPlusV(tuple(_dd(v, ctx) for v in vs))
        case OProdV(vs):
            return _dd_multilinear_tuple(OProdV, vs, ctx, e)
        case Eval():
            return _dd_eval(e, ctx)
        case Compose():
            return _dd_compose(e, ctx)
        case Trace(f, i, j):
            return _wrap_linear(lambda d: Trace(d, i, j), _dd(f, ctx), e, ctx)
        case Solve():
            return _dd_solve(e, ctx)
        case Pinv():
            return _dd_pinv(e, ctx)
        case SolveApply():
            return _dd_solve_apply(e, ctx)
        case PinvApply():
            return _dd_pinv_apply(e, ctx)
        case Dagger(f):
            return _wrap_linear(Dagger, _dd(f, ctx), e, ctx)
        case Project(v, s):
            return _wrap_linear(lambda d: Project(d, s), _dd(v, ctx), e, ctx)
        case Embed(v, s):
            return _wrap_linear(lambda d: Embed(d, s), _dd(v, ctx), e, ctx)
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
        case NEval() | Lam() | Frame():
            raise DiffError(
                f"diff: cannot differentiate a higher-order construct ({type(e).__name__}); "
                f"differentiation of SpaceFunctions / Lam-as-function / Frame is unsupported"
            )
        case _:
            raise DiffError(f"diff: no derivative rule for {type(e).__name__}")


def _dd_var(e: Var, ctx: _DCtx) -> Term:
    if e.uid == ctx.x.uid:
        return ctx.h  # d x = h (the seed direction)
    if e.uid in ctx.deps:
        if e.uid not in ctx._ddval:
            dval = ctx.deps[e.uid]
            if dval is None:
                # Unknown derivative ⇒ a fresh symbolic map var ∂e/∂x : X ⊸ space(e); its
                # directional derivative is (∂e/∂x)·h.
                dmap = Var(f"d_{e.name}_{next(_counter)}", linear_map_space(ctx.X, e.space))
            else:
                dmap = dval
            ctx._ddval[e.uid] = Eval(dmap, (0,), (ctx.h,))
        return ctx._ddval[e.uid]
    return Zero(e.space)  # x-independent variable


def _dd_external(e: External, ctx: _DCtx) -> Term:
    """Nonlinear scalar ops — chain / quotient rules over scalar values."""
    name = getattr(e.data, "name", None)
    args = e.args
    if name == "sqrt":
        (u,) = args
        du = _dd(u, ctx)
        if isinstance(du, Zero):
            return _zero(e, ctx)
        return Scale(Scale(0.5, _recip(_sqrt(u))), du)  # du / (2√u)
    if name == "reciprocal":
        (u,) = args
        du = _dd(u, ctx)
        if isinstance(du, Zero):
            return _zero(e, ctx)
        return Scale(Scale(-1.0, _recip(_mul(u, u))), du)  # −du / u²
    if name == "div":
        u, w = args
        du, dw = _dd(u, ctx), _dd(w, ctx)
        parts = []
        if not isinstance(du, Zero):
            parts.append(Scale(_recip(w), du))  # du / w
        if not isinstance(dw, Zero):
            parts.append(Scale(Scale(-1.0, _mul(u, _recip(_mul(w, w)))), dw))  # −u dw / w²
        return _sum(parts, e, ctx)
    if name == "abs":
        (u,) = args
        du = _dd(u, ctx)
        if isinstance(du, Zero):
            return _zero(e, ctx)
        return Scale(_sign(u), du)  # sgn(u) du
    if name == "sign":
        return _zero(e, ctx)
    raise DiffError(
        f"diff: no derivative rule for external op {name!r}; supply ∂/∂x via deps or "
        f"extend the rule table"
    )


def _dd_scale(e: Scale, ctx: _DCtx) -> Term:
    """``d(r·v) = (dr)·v + r·(dv)`` — product rule (``r`` may be a constant or a scalar
    term)."""
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


def _dd_multilinear_tuple(ctor, vs: tuple[Term, ...], ctx: _DCtx, e: Term) -> Term:
    """Leibniz rule for an n-ary multilinear tuple node (``OProdV``): differentiate one
    factor at a time, keeping the others."""
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
    """``d(f(args)) = (df)(args) + Σₖ f(…, dargₖ, …)`` — product rule over the map and
    every argument (the multilinear-form case)."""
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


def _dd_kron(e: Kron, ctx: _DCtx) -> Term:
    df, dg = _dd(e.f, ctx), _dd(e.g, ctx)
    parts: list[Term] = []
    if not isinstance(df, Zero):
        parts.append(Kron(df, e.g))
    if not isinstance(dg, Zero):
        parts.append(Kron(e.f, dg))
    return _sum(parts, e, ctx)


def _dd_solve(e: Solve, ctx: _DCtx) -> Term:
    """``d(f⁻¹) = −f⁻¹ (df) f⁻¹`` (the bare map's derivative).

    When ``f`` is x-independent this is the zero map. When ``f`` depends on ``x`` the
    derivative is a higher tensor (``∂f`` carries the seed *and* a full map codomain) that
    the substrate cannot yet thread through the bulk inverse op — rejected (v1); apply the
    map and differentiate ``Solve(f) v`` instead, where the seed flows through ``v``."""
    df = _dd(e.f, ctx)
    if isinstance(df, Zero):
        return _zero(e, ctx)
    raise DiffError(
        "diff: differentiating a bare Solve(f) w.r.t. x with x in f is not supported in "
        "v1; differentiate the applied form Solve(f)·v (seed through v) instead"
    )


def _dd_pinv(e: Pinv, ctx: _DCtx) -> Term:
    """``d(f†) = −f† (df) f†`` (leading term). Same v1 restriction as :func:`_dd_solve`:
    zero map when ``f`` is x-independent, otherwise rejected."""
    df = _dd(e.f, ctx)
    if isinstance(df, Zero):
        return _zero(e, ctx)
    raise DiffError(
        "diff: differentiating a bare Pinv(f) w.r.t. x with x in f is not supported in v1; "
        "differentiate the applied form Pinv(f)·v (seed through v) instead"
    )


def _dd_solve_apply(e: SolveApply, ctx: _DCtx) -> Term:
    """``d(f⁻¹ v) = f⁻¹(dv) − f⁻¹ (df)(f⁻¹ v)``.

    The applied form is routed through the inverse *map* ``Solve(f)`` so the seed only ever
    flows through ``Eval`` (einsum, free-label-aware), never the bulk ``SolveOp``."""
    return _dd_inverse_apply(e, e.f, e.v, Solve, ctx)


def _dd_pinv_apply(e: PinvApply, ctx: _DCtx) -> Term:
    """``d(f† v) = f†(dv) − f† (df)(f† v)`` (leading term; square-bijective exact)."""
    return _dd_inverse_apply(e, e.f, e.v, Pinv, ctx)


def _dd_inverse_apply(e: Term, f: Term, v: Term, inv, ctx: _DCtx) -> Term:
    """Shared ``d((inv f) v)`` rule for ``Solve``/``Pinv``. The seed must stay out of the
    bulk inverse op, so we apply the inverse *map* ``inv(f)`` via ``Eval``: the
    vector-position seed (``dv``) threads cleanly. Differentiating w.r.t. the inverted map
    itself (``df ≠ 0``) yields a higher tensor the substrate cannot yet thread through the
    bulk op — that case is rejected (v1)."""
    finv = inv(f)
    parts: list[Term] = []
    dv = _dd(v, ctx)
    if not isinstance(dv, Zero):
        parts.append(Eval(finv, (0,), (dv,)))
    df = _dd(f, ctx)
    if not isinstance(df, Zero):
        raise DiffError(
            f"diff: differentiating {type(e).__name__} w.r.t. the inverted map is not "
            f"supported in v1 (the seed would flow through the bulk inverse op); only the "
            f"applied vector may depend on x here"
        )
    return _sum(parts, e, ctx)


def _dd_llam(e: LLam, vars_: tuple[Var, ...], body: Term, ctx: _DCtx) -> Term:
    """Differentiate a (multi)linear map value: differentiate the body, keep the binder.
    ``x`` must not be one of the linear vars (those are not the differentiation variable)."""
    if any(v.uid == ctx.x.uid for v in vars_):
        raise DiffError("diff: the differentiation variable is bound by an LLam in the term")
    dbody = _dd(body, ctx)
    if isinstance(dbody, Zero):
        return _zero(e, ctx)
    return LLam(vars_, dbody)


# ===========================================================================
# Covariant correction (the metric connection). Flat / constant metric ⇒ 0.
# ===========================================================================


def _covariant_section(e: Term, ctx: _DCtx) -> tuple[Term, Space, Term, Term] | None:
    """If ``e`` is a covariant *section* ``Remetric(sec, g)`` over a manifold ``M`` with an
    **x-dependent** metric ``g``, return ``(sec, M, g, Dg)``; otherwise ``None``.

    A constant metric (``Dg`` ``None``) is *not* covariant — the plain total derivative is
    correct — so it falls through to the ordinary path. ``Remetric`` is the marker placing a
    value into the (co)tangent bundle of ``M``; ``g`` is ``M``'s metric (``M*⊗M*``)."""
    from grassmann.infer.synth import synth_space

    if not isinstance(e, Remetric) or not is_term(e.metric):
        return None
    sec, g = e.vector, e.metric
    M = _manifold_of(synth_space(sec))
    if M is None:
        return None
    Dg = _metric_gradient(g, M, ctx)
    if Dg is None:  # constant / x-independent metric ⇒ flat ⇒ ordinary derivative
        return None
    if _strip_space(ctx.X) != M:
        raise DiffError(
            "diff: the metric connection requires the differentiation variable to live in "
            "the tangent space (X == the manifold base space M)"
        )
    return (sec, M, g, Dg)


def _manifold_of(s: Space) -> Space | None:
    """The base manifold ``M`` of a (co)tangent (tensor) space: ``M``, ``M*``, or a tensor
    product whose factors are all ``M`` / ``M*`` over a single ``M``; else ``None``."""
    from grassmann.core.space import Dual, OProd

    def base(f: Space) -> Space:
        return f.space if isinstance(f, Dual) else f

    if isinstance(s, OProd):
        bases = {base(f) for f in s.spaces}
        return next(iter(bases)) if len(bases) == 1 else None
    return base(s)


def _connection_correction(
    sec: Term, sspace: Space, M: Space, g: Term, Dg: Term, ctx: _DCtx
) -> Term:
    """The Levi-Civita correction ``∇sec − d(sec)`` in direction ``h``, **basis-free**.

    The tensor-product/Leibniz rule: apply the connection map ``Γ(h) : M ⊸ M`` to each slot
    of the section — ``+Γ`` on a contravariant (``M``) factor, ``−Γ`` on a covariant
    (``M*``) factor:

    - vector ``v : M``        →  ``Γ(h)·v``                       (Eval)
    - covector ``ω : M*``     →  ``−ωᵀ·Γ(h)``                     (precompose)
    - ``(1,1)`` tensor ``T : M ⊸ M`` → ``Γ(h)∘T − T∘Γ(h)``       (the commutator)

    Everything is built from ``Eval``/``Compose`` of the bare connection map, so no metric
    basis is ever realized (the lowering stays in standard coordinates)."""
    from grassmann.core.space import OProd, dual

    conn = _connection_map(M, g, Dg, ctx.h)  # Γ(h) : M ⊸ M, bare
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
        f"diff: covariant differentiation of this tensor section ({sspace!r}) is not yet "
        f"supported (vector, covector, all-covariant forms, and (1,1) maps are). A pure "
        f"contravariant slot needs per-slot Kron (free-label threading) — see plans/80."
    )


def _connection_map(M: Space, g: Term, Dg: Term, h: Var) -> Term:
    """The Levi-Civita connection map ``Γ(h) : M ⊸ M`` — **basis-free**.

    ``Γ(h)^k_m = g^{kl}·½(∂ₘg_{il} + ∂ᵢg_{ml} − ∂ₗg_{mi}) hⁱ``. Built as ``g⁻¹ ∘ Glower``,
    where ``Glower : M ⊸ M*`` is the first-kind symbol contracted with ``h`` (the ``Permute``
    supplies the one transpose the reshape-only ``Lift`` cannot). The inverse metric is the
    **bare** map ``Pinv(g)`` — ``g`` decodes as the flat map ``M ⊸ M*`` — so the connection
    carries no metric basis tag and the whole derivative stays coordinate-free."""
    t1 = Eval(Dg, (0,), (h,))  # ∂ᵢg_{jl} hⁱ, indexed (j, l)
    t2 = Eval(Dg, (1,), (h,))  # ∂ⱼg_{il} hⁱ, indexed (j, l)
    g_lower_map = Scale(0.5, Sum((t1, t2, Scale(-1.0, Permute(t2, (1, 0))))))  # M ⊸ M*
    return Compose(Pinv(g), (0,), (g_lower_map,))  # raise the cod index with g⁻¹ ⇒ M ⊸ M


def _metric_gradient(g: Term, M: Space, ctx: _DCtx) -> Term | None:
    """The metric gradient ``Dg : M ⊸ (M*⊗M*)`` (``∂g/∂x``), as a *primitive* map, or
    ``None`` when the metric does not depend on ``x`` (a constant/flat metric).

    For v1 the metric must be a marked-dependent ``Var`` (its gradient supplied via ``deps``,
    or a fresh symbolic map var for ``None``); a constant metric ``Var`` (absent from
    ``deps``) yields ``None`` (the flat case). A more complex x-dependent metric expression
    is rejected — supply the metric (and its derivative) as a marked variable."""
    if isinstance(g, Var):
        if g.uid not in ctx.deps:
            return None  # metric does not depend on x ⇒ flat
        dval = ctx.deps[g.uid]
        if dval is not None:
            return dval
        from grassmann.core.space import dual, oprod

        cod = oprod(dual(M), dual(M))
        return Var(f"d_{g.name}_{next(_counter)}", linear_map_space(ctx.X, cod))
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


def _wrap_linear(ctor, dv: Term, e: Term, ctx: _DCtx) -> Term:
    """A node linear in one child: its derivative is the same node over the child's
    derivative (or ``Zero`` when the child is x-independent)."""
    if isinstance(dv, Zero):
        return _zero(e, ctx)
    return ctor(dv)


# --- scalar op builders (the nonlinear-rule coefficients) -------------------


def _ext(name: str, fn, *args: Term) -> External:
    from grassmann.lower.represent import ExternalOp

    return External(Scalar(), ExternalOp(name, fn), tuple(args))


def _sqrt(u: Term) -> External:
    return _ext("sqrt", np.sqrt, u)


def _recip(u: Term) -> External:
    return _ext("reciprocal", lambda a: np.divide(1.0, a), u)


def _sign(u: Term) -> External:
    return _ext("sign", np.sign, u)


def _mul(a: Term, b: Term) -> Scale:
    """Scalar·scalar product (both ``Scalar``-valued) as a ``Scale``."""
    return Scale(a, b)


# --- well-formedness guard --------------------------------------------------


def _guard_closed_and_first_order(e: Term, x: Var) -> None:
    """Reject an open term (free space variable / unresolved metavar) and any occurrence of
    ``x`` in a *space* position (a dependence position) — differentiating through a space
    function is unsupported."""
    from grassmann.core.space import SpaceMeta, SVar, is_space
    from grassmann.infer.synth import InferError, synth_space

    try:
        s = synth_space(e)
    except InferError as exc:
        raise DiffError(f"diff: term does not type-check / is not closed ({exc})") from exc
    if _mentions_space_meta(s):
        raise DiffError("diff: term has an unresolved space; it must be closed")

    # x must not flow into a space (a Frame/SpaceFunction would make a space depend on x).
    def walk(node: object) -> None:
        if isinstance(node, Frame):
            raise DiffError("diff: cannot differentiate through a Frame (basis binder)")
        for child in _children(node):
            walk(child)

    walk(e)
    _ = (SpaceMeta, SVar, is_space)


def _mentions_space_meta(s: Space) -> bool:
    from grassmann.core.space import SpaceMeta, SVar

    found = False

    def walk(node: object) -> None:
        nonlocal found
        if isinstance(node, (SpaceMeta, SVar)):
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


__all__ = ["diff", "DiffError"]
