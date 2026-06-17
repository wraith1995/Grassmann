"""Bidirectional type synthesis ``Γ ⊢ e ⇒ s`` (`25 §3`).

Synthesizes the Space (for vectors / linear maps) or Type (for ``Lam`` over a Term, i.e.
a ``Π``; or an arrow Kind for Space/Type Functions) of a core ``Term``. Application
(``Eval``/``NEval``), composition, trace, inverses, movement and the binders are typed
per the contraction-view rules of `25 §3`.

The companion :mod:`grassmann.infer.check` runs the ``⇐`` direction. Unification of
implicit space metavars and dimension equality live in :mod:`grassmann.infer.unify` /
:mod:`grassmann.infer.dims`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from grassmann.core.kinds import NAT, SPACE, STAR, Arrow, Kind, Sort, SpaceSort
from grassmann.core.space import (
    Dual,
    OPlus,
    OProd,
    Scalar,
    Space,
    dual,
    is_space,
    linear_map_space,
    oplus,
    oprod,
)
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
from grassmann.core.types import Pi, SpaceAsType, Type, is_type, space_as_type

from .properties import is_bijective, is_injective, is_surjective
from .subst import subst_type_space
from .unify import MetaStore, fresh_meta, unify


class InferError(Exception):
    """Raised on an inference failure that is not a raw unification mismatch."""


# Result of synthesizing a Term: either a Space (value / linear map), a Type (a Π for a
# Lam over a Term), or an arrow Kind (a Space/Type Function).
SynthResult = Space | Type | Kind


@dataclass
class Ctx:
    """Inference context ``Γ`` (`25 §0`): metavar store + type-var scope.

    ``space_vars`` records in-scope rigid space vars introduced by ``Lam`` over
    ``space`` (used only for completeness; the core already carries them as ``SVar``).
    """

    store: MetaStore = field(default_factory=MetaStore)
    space_vars: set[str] = field(default_factory=set)


def synth(term: Term, ctx: Ctx | None = None) -> SynthResult:
    """Synthesize the Space/Type/Kind of ``term`` (`25 §3`)."""
    if ctx is None:
        ctx = Ctx()
    if not is_term(term):
        raise InferError(f"synth: expected a Term, got {term!r}")
    return _synth(term, ctx)


def synth_space(term: Term, ctx: Ctx | None = None) -> Space:
    """Synthesize and require a Space result (resolved against the metastore)."""
    if ctx is None:
        ctx = Ctx()
    r = synth(term, ctx)
    if not is_space(r):
        raise InferError(f"expected a value/map (a Space), got type {r!r} for {term!r}")
    return ctx.store.resolve(r)  # type: ignore[arg-type]


# ===========================================================================
# Core synthesis dispatch.
# ===========================================================================


def _synth(e: Term, ctx: Ctx) -> SynthResult:
    if isinstance(e, Var):
        return e.space
    if isinstance(e, Lit):
        return Scalar()
    if isinstance(e, External):
        return e.space
    if isinstance(e, Zero):
        return e.space
    if isinstance(e, Identity):
        # Identity s : ~s * s  (a linear map s ⊸ s).
        return linear_map_space(e.space, e.space)
    if isinstance(e, Scale):
        return _synth_value(e.vector, ctx)
    if isinstance(e, Sum):
        return _synth_sum(e, ctx)
    if isinstance(e, OPlusV):
        return oplus(*(_synth_value(v, ctx) for v in e.vectors))
    if isinstance(e, OProdV):
        return oprod(*(_synth_value(v, ctx) for v in e.vectors))
    if isinstance(e, Eval):
        return _synth_eval(e, ctx)
    if isinstance(e, NEval):
        return _synth_neval(e, ctx)
    if isinstance(e, Compose):
        return _synth_compose(e, ctx)
    if isinstance(e, Trace):
        return _synth_trace(e, ctx)
    if isinstance(e, Solve):
        return _synth_solve(e, ctx)
    if isinstance(e, Pinv):
        return _synth_pinv(e, ctx)
    if isinstance(e, SolveApply):
        return _synth_solve_apply(e, ctx)
    if isinstance(e, PinvApply):
        return _synth_pinv_apply(e, ctx)
    if isinstance(e, Dagger):
        return _synth_dagger(e, ctx)
    if isinstance(e, Inner):
        # Inner(s) is the metric bilinear form on s — an element of s* ⊗ s* (≡ ~s ⊗ ~s),
        # with two covariant slots. Full contraction against two s-vectors ⇒ Scalar
        # (same representation as Flat: a metric g is both the form and the map s→s*).
        return oprod(dual(e.space), dual(e.space))
    if isinstance(e, Flat):
        return linear_map_space(e.space, dual(e.space))
    if isinstance(e, Sharp):
        return linear_map_space(dual(e.space), e.space)
    if isinstance(e, Project):
        return _synth_project(e, ctx)
    if isinstance(e, Embed):
        return _synth_embed(e, ctx)
    if isinstance(e, Assert):
        return _synth_assert(e, ctx)
    if isinstance(e, TupleEmbed):
        return _synth_tuple_embed(e, ctx)
    if isinstance(e, TupleProject):
        return _synth_tuple_project(e, ctx)
    if isinstance(e, Remetric):
        return _synth_remetric(e, ctx)
    if isinstance(e, Kron):
        return _synth_kron(e, ctx)
    if isinstance(e, (Lift, InvLift)):
        return _synth_lift(e, ctx)
    if isinstance(e, LLam):
        return _synth_llam(e, ctx)
    if isinstance(e, Lam):
        return _synth_lam(e, ctx)
    if isinstance(e, Frame):
        raise InferError("synth: Frame (basis binder) typing is out of IW3 scope")
    raise InferError(f"synth: unsupported term {e!r}")


def _synth_value(v: Term, ctx: Ctx) -> Space:
    r = _synth(v, ctx)
    if not is_space(r):
        raise InferError(f"expected a value (Space), got {r!r} for {v!r}")
    return r  # type: ignore[return-value]


# ===========================================================================
# Algebra.
# ===========================================================================


def _synth_sum(e: Sum, ctx: Ctx) -> Space:
    first = _synth_value(e.vectors[0], ctx)
    for v in e.vectors[1:]:
        s = _synth_value(v, ctx)
        unify(first, s, ctx.store)
    return ctx.store.resolve(first)


# ===========================================================================
# Application / composition / trace (contraction view, `25 §3`).
# ===========================================================================


def _as_factors(s: Space) -> tuple[Space, ...]:
    """A map's tensor factors (contraction view). A non-⊗ map is a single factor."""
    if isinstance(s, OProd):
        return s.spaces
    return (s,)


def _synth_eval(e: Eval, ctx: Ctx) -> Space:
    """``Eval(f, pos, args)`` — linear application (`25 §3`).

    ``f ⇒ ⊗(t…)``; for each j, ``t_{pos[j]} ~ Dual(s_j)`` with ``args[j] ⇐ s_j``
    (``s_j`` may be a fresh ``?α`` recovered from the arg); result = remaining factors.
    """
    fs = _synth_value(e.f, ctx)
    fs = ctx.store.resolve(fs)
    factors = list(_as_factors(fs))
    for pos, arg in zip(e.positions, e.args, strict=True):
        if pos >= len(factors):
            raise InferError(
                f"Eval: slot {pos} out of range for map with {len(factors)} factors"
            )
        sj = fresh_meta("arg")
        # The selected slot must be a Dual input slot of space s_j.
        unify(factors[pos], dual(sj), ctx.store)
        # Check the argument against s_j (recovers the metavar).
        _check(arg, sj, ctx)
    remaining = [factors[i] for i in range(len(factors)) if i not in set(e.positions)]
    remaining = [ctx.store.resolve(r) for r in remaining]
    if not remaining:
        # Fully applied (e.g. a scalar form contracted to nothing) ⇒ Scalar.
        return Scalar()
    if len(remaining) == 1:
        return remaining[0]
    return oprod(*remaining)


def _synth_neval(e: NEval, ctx: Ctx) -> Space:
    """``NEval(f, pos, args)`` — non-linear application of a ``Lam`` (`25 §3`).

    ``f ⇒ Π(b:σ).τ`` (curried per slot). A value arg is checked against σ; an implicit
    space param (σ=space) gets a fresh ``?α`` recovered from the arg/result; an explicit
    space/nat arg substitutes directly.
    """
    fty = _synth(e.f, ctx)
    supplied = dict(zip(e.positions, e.args, strict=True))
    result: object = fty
    slot = 0
    # Walk the Π chain by absolute slot index. A SPACE slot not explicitly supplied is
    # an IMPLICIT space param: instantiate a fresh ?α (recovered by unification as later
    # value args / the expected type flow in, `25 §5`).
    while isinstance(result, Pi):
        if slot in supplied:
            result = _apply_pi(result, supplied[slot], ctx)
        elif isinstance(result.domain, Sort) and result.domain == SPACE:
            result = _subst_codomain(result, fresh_meta(result.binder), ctx)
        else:
            # No more supplied args for this (value/nat) slot ⇒ partial application.
            break
        slot += 1
    result = ctx.store.resolve(result) if is_space(result) else result
    if isinstance(result, SpaceAsType):
        return ctx.store.resolve(result.space)
    if is_space(result):
        return ctx.store.resolve(result)  # type: ignore[arg-type]
    raise InferError(f"NEval: application did not reduce to a Space, got {result!r}")


def _apply_pi(fty: Pi, arg: object, ctx: Ctx) -> object:
    dom = fty.domain
    if isinstance(dom, SpaceSort):
        # Value domain: arg is a Term checked against the domain space.
        if not is_term(arg):
            raise InferError(f"NEval: expected a value argument, got {arg!r}")
        _check(arg, dom.space, ctx)  # type: ignore[arg-type]
        return _subst_codomain(fty, arg, ctx)
    if isinstance(dom, Sort) and dom == SPACE:
        # Space param: explicit space arg, or recover via a fresh metavar.
        sp: Space = arg if is_space(arg) else fresh_meta(fty.binder)  # type: ignore[assignment]
        return _subst_codomain(fty, sp, ctx)
    if isinstance(dom, Sort) and dom == NAT:
        # Nat param: must be explicit (nats are not unified, `25 §2`).
        from grassmann.core.space import is_dim

        if not is_dim(arg):
            raise InferError("NEval: nat parameter requires an explicit nat [n]")
        return _subst_codomain(fty, arg, ctx)
    raise InferError(f"NEval: unsupported Π domain {dom!r}")


def _subst_codomain(fty: Pi, value: object, ctx: Ctx) -> object:
    """Substitute the bound variable of ``fty`` by ``value`` in its codomain."""
    cod: Type = fty.codomain
    # SpaceSort domains bind a term Var by name; space/nat domains bind an SVar/DimVar.
    if isinstance(fty.domain, SpaceSort):
        return _subst_type_term(cod, fty.binder, value, ctx)
    if fty.domain == SPACE:
        return _subst_type_space(cod, fty.binder, value, ctx)
    if fty.domain == NAT:
        return _subst_type_nat(cod, fty.binder, value, ctx)
    raise InferError(f"_subst_codomain: unsupported domain {fty.domain!r}")


def _subst_type_space(t: Type, name: str, sp: Space, ctx: Ctx) -> Type:
    return subst_type_space(t, name, sp)  # type: ignore[return-value]


def _subst_type_term(t: Type, name: str, val: object, ctx: Ctx) -> Type:
    # Value substitution only affects embedded terms inside spaces (rare in these
    # examples). For ⟦s⟧ codomains independent of the value, this is the identity.
    return t


def _subst_type_nat(t: Type, name: str, n: object, ctx: Ctx) -> Type:
    return t


def _synth_compose(e: Compose, ctx: Ctx) -> Space:
    """``Compose(f, pos, gs)`` — contract each gₖ's codomain into f's selected slot."""
    fs = ctx.store.resolve(_synth_value(e.f, ctx))
    factors = list(_as_factors(fs))
    new_factors = list(factors)
    for pos, g in zip(e.positions, e.gs, strict=True):
        if pos >= len(factors):
            raise InferError(f"Compose: slot {pos} out of range")
        g_dom, g_cod = _map_dom_cod_resolved(g, ctx)
        # f's slot is a Dual(s); g's codomain must equal s, and the composed map takes
        # g's domain there instead.
        slot = fresh_meta("c")
        unify(factors[pos], dual(slot), ctx.store)
        unify(slot, g_cod, ctx.store)
        new_factors[pos] = dual(g_dom)
    new_factors = [ctx.store.resolve(f) for f in new_factors]
    if len(new_factors) == 1:
        return new_factors[0]
    return oprod(*new_factors)


def _synth_trace(e: Trace, ctx: Ctx) -> Space:
    """``Trace(f, i, j)`` — contract primal index i with dual index j (`25 §3`)."""
    fs = ctx.store.resolve(_synth_value(e.f, ctx))
    factors = list(_as_factors(fs))
    n = len(factors)
    if e.i >= n or e.j >= n:
        raise InferError(f"Trace: index out of range for {n}-factor map")
    si = factors[e.i]
    unify(factors[e.j], dual(si), ctx.store)
    remaining = [factors[k] for k in range(n) if k not in (e.i, e.j)]
    remaining = [ctx.store.resolve(r) for r in remaining]
    if not remaining:
        return Scalar()
    if len(remaining) == 1:
        return remaining[0]
    return oprod(*remaining)


# ===========================================================================
# Map domain/codomain decoding (for FFS source/target + inverses).
# ===========================================================================


def map_domain_codomain(f: object) -> tuple[Space, Space]:
    """Decode a map term ``f`` into ``(dom, cod)`` where ``f : dom ⊸ cod``.

    A map space is ``~dom * cod`` (an ``OProd`` of one ``Dual`` input factor and a
    primal tail), or any single-input/single-output linear-map space. Used by FFS
    source/target recovery (:mod:`grassmann.infer.dims`) and the inverse rules.
    """
    ctx = Ctx()
    return _map_dom_cod_resolved(f, ctx)


def _map_dom_cod_resolved(f: object, ctx: Ctx) -> tuple[Space, Space]:
    if not is_term(f):
        raise InferError(f"expected a map Term, got {f!r}")
    s = ctx.store.resolve(_synth_value(f, ctx))  # type: ignore[arg-type]
    return _decode_map_space(s)


def _decode_map_space(s: Space) -> tuple[Space, Space]:
    """``~dom * cod`` ⇒ ``(dom, cod)``. Leading ``Dual`` factors are inputs."""
    factors = list(_as_factors(s))
    duals = [f for f in factors if isinstance(f, Dual)]
    prims = [f for f in factors if not isinstance(f, Dual)]
    if len(duals) == 1 and len(prims) == 1:
        return duals[0].space, prims[0]
    if len(duals) == 1 and len(prims) >= 1:
        return duals[0].space, oprod(*prims) if len(prims) > 1 else prims[0]
    if len(duals) >= 1 and len(prims) == 1:
        dom = oprod(*(d.space for d in duals)) if len(duals) > 1 else duals[0].space
        return dom, prims[0]
    raise InferError(f"cannot decode map space (dom ⊸ cod): {s!r}")


# ===========================================================================
# Inverses & adjoint (`25 §3`).
# ===========================================================================


def _synth_solve(e: Solve, ctx: Ctx) -> Space:
    if not is_bijective(e.f):
        raise InferError("Solve requires the map to be asserted `bij` (25 §3/§7)")
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    return linear_map_space(cod, dom)


def _synth_pinv(e: Pinv, ctx: Ctx) -> Space:
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    # inj/surj sharpen the realization (lowering); the space is cod ⊸ dom regardless.
    _ = (is_injective(e.f), is_surjective(e.f))
    return linear_map_space(cod, dom)


def _synth_solve_apply(e: SolveApply, ctx: Ctx) -> Space:
    if not is_bijective(e.f):
        raise InferError("SolveApply requires the map to be asserted `bij`")
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    _check(e.v, cod, ctx)
    return dom


def _synth_pinv_apply(e: PinvApply, ctx: Ctx) -> Space:
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    _check(e.v, cod, ctx)
    return dom


def _synth_dagger(e: Dagger, ctx: Ctx) -> Space:
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    return linear_map_space(cod, dom)


# ===========================================================================
# Movement / coercion / reshape (`25 §3`).
# ===========================================================================


def _synth_project(e: Project, ctx: Ctx) -> Space:
    from .subspace import is_subspace

    s = ctx.store.resolve(_synth_value(e.vector, ctx))
    if not is_subspace(e.space, s):
        raise InferError(f"project: {e.space!r} is not a subspace of {s!r} (25 §6)")
    return e.space


def _synth_embed(e: Embed, ctx: Ctx) -> Space:
    from .subspace import is_subspace

    s = ctx.store.resolve(_synth_value(e.vector, ctx))
    if not is_subspace(s, e.space):
        raise InferError(f"embed: {s!r} is not a subspace of {e.space!r} (25 §6)")
    return e.space


def _synth_assert(e: Assert, ctx: Ctx) -> Space:
    s = _synth_value(e.vector, ctx)
    unify(s, e.space, ctx.store)
    return ctx.store.resolve(e.space)


def _synth_tuple_embed(e: TupleEmbed, ctx: Ctx) -> Space:
    s = e.space
    if not isinstance(s, OPlus):
        raise InferError(f"TupleEmbed: space must be a ⊕, got {s!r}")
    if e.i >= len(s.spaces):
        raise InferError(f"TupleEmbed: component {e.i} out of range")
    return linear_map_space(s.spaces[e.i], s)


def _synth_tuple_project(e: TupleProject, ctx: Ctx) -> Space:
    s = e.space
    if not isinstance(s, OPlus):
        raise InferError(f"TupleProject: space must be a ⊕, got {s!r}")
    if e.i >= len(s.spaces):
        raise InferError(f"TupleProject: component {e.i} out of range")
    return linear_map_space(s, s.spaces[e.i])


def _synth_remetric(e: Remetric, ctx: Ctx) -> Space:
    from grassmann.core.space import with_inner

    s = ctx.store.resolve(_synth_value(e.vector, ctx))
    return with_inner(s, e.metric)


def _synth_kron(e: Kron, ctx: Ctx) -> Space:
    a0, b0 = _map_dom_cod_resolved(e.f, ctx)
    a1, b1 = _map_dom_cod_resolved(e.g, ctx)
    return linear_map_space(oprod(a0, a1), oprod(b0, b1))


def _synth_lift(e: Lift | InvLift, ctx: Ctx) -> Space:
    """``Lift``/``InvLift`` — fuse the listed tensor factors of the vector's space.

    ``Lift`` fuses the listed (input/``Dual``) factors into one ``Dual`` of their
    ⊗; ``InvLift`` is the inverse reshape. The non-fused factors are kept in order, with
    the fused block inserted at ``k`` (default: in place of the first fused index).
    """
    s = ctx.store.resolve(_synth_value(e.vector, ctx))
    factors = list(_as_factors(s))
    n = len(factors)
    for idx in e.fused:
        if idx >= n:
            raise InferError(f"Lift/InvLift: index {idx} out of range for {n} factors")
    fused_factors = [factors[i] for i in e.fused]
    rest = [(i, factors[i]) for i in range(n) if i not in set(e.fused)]
    insert_at = e.k if e.k is not None else (min(e.fused) if e.fused else 0)

    if isinstance(e, Lift):
        # Fuse the (dual) input slots into one Dual(⊗(inputs)).
        inner = [
            f.space if isinstance(f, Dual) else f for f in fused_factors
        ]
        block: Space = dual(oprod(*inner)) if inner else oprod()
    else:
        # InvLift: fuse the listed factors primally, *reassociating* — a fused factor
        # that is itself an ⊗ (a nested ``OProd``, e.g. ``V⊗ᵏ``) flattens into the
        # block, so ``InvLift((V⊗ᵏ)⊗(V⊗ˡ), [0,1])`` reassociates to the flat
        # ``V⊗^{k+l}`` rather than keeping the nesting (`26 §2`, the explicit wedge
        # reassociation: the core never silently flattens nested ⊗).
        flat: list[Space] = []
        for f in fused_factors:
            flat.extend(_as_factors(f))
        block = oprod(*flat) if len(flat) != 1 else flat[0]

    out: list[Space] = [f for _, f in rest]
    out.insert(min(insert_at, len(out)), block)
    if len(out) == 1:
        return out[0]
    return oprod(*out)


# ===========================================================================
# Binders (`25 §3`).
# ===========================================================================


def _synth_llam(e: LLam, ctx: Ctx) -> Space:
    """``LLam([xᵢ:sᵢ]. e) ⇒ ⊗(Dual sᵢ) * s_e`` (the linear-map space, `25 §3`)."""
    s_e = ctx.store.resolve(_synth_value(e.body, ctx))
    dom_factors = [dual(v.space) for v in e.vars]
    return oprod(*dom_factors, s_e)


def _synth_lam(e: Lam, ctx: Ctx) -> SynthResult:
    """``Lam`` synthesis (`25 §3`):

    - Term body ⇒ a curried ``Π(bᵢ:σᵢ). Xbody``.
    - Space body ⇒ an arrow Kind ``σ → space``.
    - Type body ⇒ an arrow Kind ``σ → *``.
    """
    body = e.body
    if is_term(body):
        # Synthesize body type, then wrap binders right-to-left into Π.
        sub = Ctx(store=ctx.store, space_vars=set(ctx.space_vars))
        for _name, sort in e.binders:
            if sort == SPACE:
                sub.space_vars.add(_name)
        body_res = _synth(body, sub)  # type: ignore[arg-type]
        cod: Type = _as_type_result(body_res)
        for name, sort in reversed(e.binders):
            cod = Pi(name, sort, cod)
        return cod
    if is_space(body):
        return _arrow_kind(e.binders, SPACE)
    if is_type(body):
        return _arrow_kind(e.binders, STAR)
    # Dim body ⇒ nat-producing function.
    from grassmann.core.space import is_dim

    if is_dim(body):
        return _arrow_kind(e.binders, NAT)
    raise InferError(f"Lam: unsupported body {body!r}")


def _as_type_result(r: SynthResult) -> Type:
    if is_type(r):
        return r  # type: ignore[return-value]
    if is_space(r):
        return space_as_type(r)  # type: ignore[arg-type]
    raise InferError(f"Lam body produced a non-Type result {r!r}")


def _arrow_kind(binders: tuple[tuple[str, Kind], ...], result: Sort) -> Kind:
    k: Kind = result
    for _name, sort in reversed(binders):
        k = Arrow(sort, k)
    return k


# ===========================================================================
# Checking hook (forward to check.py; avoids an import cycle).
# ===========================================================================


def _check(term: object, expected: Space, ctx: Ctx) -> None:
    from .check import check_against

    check_against(term, expected, ctx)


__all__ = [
    "InferError",
    "Ctx",
    "SynthResult",
    "synth",
    "synth_space",
    "map_domain_codomain",
]
