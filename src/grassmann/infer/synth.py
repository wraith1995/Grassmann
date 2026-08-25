"""The synthesize direction of inference, ``Γ ⊢ e ⇒ s`` — reading a classifier off a term.

Synthesis walks a core ``Term`` and produces its classifier: a Space for a value or a
linear map, a Type — a ``Π`` — for a ``Lam`` over a term, or an arrow Kind for a space or
type function. The dispatch is an exhaustive ``match`` over the closed Term sum, so every
node has a rule and ``assert_never`` proves none was forgotten.

The rules are written in the contraction view: a linear map ``dom ⊸ cod`` is stored as the
tensor ``Dual(dom) ⊗ cod``, so applying it contracts a dual factor away and composing
substitutes one dual factor for another. Trace contracts a primal factor against a dual
one, the binders build the product of their duals against the body's space, and the
movement and reshape nodes rearrange factors without changing what they contract to.
Whatever a rule cannot read off the term it leaves as a fresh metavariable for unification
to solve, either from a sibling or from the expected space that :mod:`grassmann.infer.check`
pushes in.

Two obligations outlive a single rule. A ``Project`` with no stated target cannot be solved
in this direction — a space has many subspaces — so it returns a metavariable and defers the
``target ⊑ parent`` check until the top level, by which point the check direction should
have pinned it down. And an asserted map property implies dimension equalities that are
verified against the resolved space rather than unified into it, since dimensions are only
ever compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import assert_never

from grassmann.core.kinds import NAT, SPACE, STAR, Arrow, Kind, Sort, SpaceSort
from grassmann.core.space import (
    Dual,
    OPlus,
    OProd,
    Scalar,
    Space,
    SpaceMeta,
    SVar,
    dual,
    is_space,
    linear_map_space,
    oplus,
    oprod,
    spliced_oprod,
)
from grassmann.core.term import (
    Assert,
    Compose,
    Corestrict,
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
from grassmann.core.types import Pi, SpaceAsType, Type, is_type, space_as_type

from .properties import is_bijective, is_injective, is_surjective
from .subst import subst_type_space
from .unify import MetaStore, UnifyError, fresh_meta, unify


class InferError(Exception):
    """Raised when a term cannot be typed for a reason other than a space mismatch.

    A structural failure — a slot index out of range, a node used where its rule does not
    apply, an unlicensed inverse — as opposed to the two-spaces-disagree case, which is
    :class:`~grassmann.infer.unify.UnifyError`.
    """


# Result of synthesizing a Term: either a Space (value / linear map), a Type (a Π for a
# Lam over a Term), or an arrow Kind (a Space/Type Function).
SynthResult = Space | Type | Kind


@dataclass
class Ctx:
    """The inference context ``Γ`` threaded through every rule.

    It carries the metavariable store where unification records its solutions, the rigid
    space variables a ``Lam`` over ``space`` has brought into scope, and the subspace
    obligations an unannotated ``Project`` has deferred to the top level. The scope set is
    informational: the core already names those variables as ``SVar`` nodes.
    """

    store: MetaStore = field(default_factory=MetaStore)
    space_vars: set[str] = field(default_factory=set)
    # Deferred ``target ⊑ parent`` obligations from a ``Project(v, None)``. The target is a
    # metavariable the ⇐ direction must solve before the subspace relation can be decided, so
    # each entry — (target_meta, parent_space, vector) — waits until the top level.
    subspace_obligations: list[tuple[Space, Space, Term]] = field(default_factory=list)


def synth(term: Term, ctx: Ctx | None = None) -> SynthResult:
    """Synthesize the classifier of ``term`` — the *synthesize* direction ``Γ ⊢ e ⇒ s``.

    Reads a type off the term: a Space for a value or linear map, a Type (a ``Π``) for a
    :class:`~grassmann.core.term.Lam` over a term, or an arrow Kind for a Space/Type
    function. The companion :func:`~grassmann.infer.check.check` runs the ``⇐`` direction.

    Parameters
    ----------
    term
        The core term to type.
    ctx
        Inference context; a fresh one is created at the top level, where any deferred
        subspace obligations are also discharged.

    Returns
    -------
    SynthResult
        The synthesized Space, Type, or Kind.

    Raises
    ------
    InferError
        If ``term`` is not a term or cannot be typed.
    """
    top = ctx is None
    if ctx is None:
        ctx = Ctx()
    if not is_term(term):
        raise InferError(f"synth: expected a Term, got {term!r}")
    r = _synth(term, ctx)
    if top:
        _check_subspace_obligations(ctx)
    return r


def synth_space(term: Term, ctx: Ctx | None = None) -> Space:
    """Synthesize ``term`` and insist the result is a Space, fully resolved.

    This is :func:`synth` for a caller that needs a value or a linear map: a term
    classifying as a Type or a Kind is a function rather than a value and raises
    :class:`InferError`. The Space that comes back has the current substitution applied all
    the way down, so it mentions no solved metavariable.
    """
    top = ctx is None
    if ctx is None:
        ctx = Ctx()
    r = synth(term, ctx)
    if top:
        _check_subspace_obligations(ctx)
    if not is_space(r):
        raise InferError(f"expected a value/map (a Space), got type {r!r} for {term!r}")
    return ctx.store.resolve(r)  # type: ignore[arg-type]


def _check_subspace_obligations(ctx: Ctx) -> None:
    """Discharge the subspace obligations an unannotated ``Project`` deferred.

    By the time the whole surrounding term has been typed, the ⇐ direction should have
    solved each deferred target metavariable, and a solved target must be a subspace of the
    space it was projected from. A target still unsolved means nothing ever determined
    which subspace was meant, which is a genuine ambiguity: the caller has to name the
    target or supply an expected space.
    """
    from .subspace import is_subspace

    pending = ctx.subspace_obligations
    ctx.subspace_obligations = []
    for target, parent, vector in pending:
        resolved = ctx.store.resolve(target)
        if ctx.store.has_unsolved(resolved):
            raise InferError(
                f"project: target subspace of {vector!r} was omitted and could not be "
                f"inferred (no expected space flowed in); annotate it explicitly"
            )
        parent = ctx.store.resolve(parent)
        if not is_subspace(resolved, parent):
            raise InferError(
                f"project: inferred target {resolved!r} is not a subspace of "
                f"{parent!r}"
            )


# ===========================================================================
# Core synthesis dispatch.
# ===========================================================================


def _synth(e: Term, ctx: Ctx) -> SynthResult:
    match e:
        case Var():
            return e.space
        case Lit():
            return Scalar()
        case External():
            return _synth_external(e, ctx)
        case Zero():
            # An omitted space is a fresh metavar, solved by the surrounding context
            # (a Sum sibling, a check/Assert against an expected space, …).
            return e.space if e.space is not None else fresh_meta("zero")
        case Identity():
            # Identity s : ~s * s  (a linear map s ⊸ s).
            return linear_map_space(e.space, e.space)
        case Scale():
            return _synth_value(e.vector, ctx)
        case Sum():
            return _synth_sum(e, ctx)
        case OPlusV():
            return oplus(*(_synth_value(v, ctx) for v in e.vectors))
        case OProdV():
            return oprod(*(_synth_value(v, ctx) for v in e.vectors))
        case Eval():
            return _synth_eval(e, ctx)
        case NEval():
            return _synth_neval(e, ctx)
        case Compose():
            return _synth_compose(e, ctx)
        case Trace():
            return _synth_trace(e, ctx)
        case Det():
            return _synth_det(e, ctx)
        case LogDet():
            return _synth_det(e, ctx)
        case OpaqueApply():
            return _synth_opaque(e, ctx)
        case Solve():
            return _synth_solve(e, ctx)
        case Pinv():
            return _synth_pinv(e, ctx)
        case Corestrict():
            return _synth_corestrict(e, ctx)
        case SolveApply():
            return _synth_solve_apply(e, ctx)
        case PinvApply():
            return _synth_pinv_apply(e, ctx)
        case Sqrt():
            return _synth_sqrt(e, ctx)
        case Dagger():
            return _synth_dagger(e, ctx)
        case Inner():
            # Inner(s) is the metric bilinear form on s — an element of s* ⊗ s* (≡ ~s ⊗ ~s),
            # with two covariant slots. Full contraction against two s-vectors ⇒ Scalar
            # (same representation as Flat: a metric g is both the form and the map s→s*).
            return oprod(dual(e.space), dual(e.space))
        case Flat():
            return linear_map_space(e.space, dual(e.space))
        case Sharp():
            return linear_map_space(dual(e.space), e.space)
        case Project():
            return _synth_project(e, ctx)
        case Embed():
            return _synth_embed(e, ctx)
        case Assert():
            return _synth_assert(e, ctx)
        case TupleEmbed():
            return _synth_tuple_embed(e, ctx)
        case TupleProject():
            return _synth_tuple_project(e, ctx)
        case Remetric():
            return _synth_remetric(e, ctx)
        case Kron():
            return _synth_kron(e, ctx)
        case Lift() | InvLift():
            return _synth_lift(e, ctx)
        case Permute():
            return _synth_permute(e, ctx)
        case TuplePermute():
            return _synth_tuple_permute(e, ctx)
        case LLam():
            return _synth_llam(e, ctx)
        case Lam():
            return _synth_lam(e, ctx)
        case Frame():
            raise InferError("synth: Frame (basis binder) typing is not yet supported")
        case _ as unreachable:
            assert_never(unreachable)


def _synth_value(v: Term, ctx: Ctx) -> Space:
    r = _synth(v, ctx)
    if not is_space(r):
        raise InferError(f"expected a value (Space), got {r!r} for {v!r}")
    return r  # type: ignore[return-value]


# ===========================================================================
# Algebra.
# ===========================================================================


def _external_requires_standard(data: object) -> bool:
    """Report whether an ``External``'s payload declares that it needs standard coordinates.

    The payload is read structurally, without importing the lowering layer: an operator
    carrying a ``basis_req`` of ``"standard"`` is meaningful only in standard coordinates,
    while ``"equivariant"`` is basis-free. A payload with no such field imposes no
    constraint.
    """
    req = getattr(data, "basis_req", None)
    if req is None:
        return False  # not a basis-aware op payload ⇒ no constraint to surface.
    return getattr(req, "value", req) == "standard"


def _synth_external(e: External, ctx: Ctx) -> Space:
    """Type a foreign elementwise operator, and surface its basis contract.

    The result space is the one the node declares. What this rule adds is enforcement of
    the kernel's basis requirement: a ``STANDARD`` operator is only meaningful in standard
    coordinates, so its operands have to be standardizable before the componentwise kernel
    runs. An operand whose space is still a basis-polymorphic metavariable or ``SVar`` has
    no known realization and hence no standard basis, and is rejected here — the case that
    would otherwise miscompute silently under a non-standard basis at lowering.
    """
    if _external_requires_standard(e.data):
        for a in e.args:
            sp = ctx.store.resolve(_synth_value(a, ctx))
            if isinstance(sp, (SpaceMeta, SVar)):  # audit: ok EX-ISINSTANCE-LADDER — narrowing: peels unresolved SVar/SpaceMeta to reject [approved 2026-07-03]
                raise InferError(
                    f"External {getattr(e.data, 'name', e.data)!r} requires standard "
                    f"coordinates (basis_req=STANDARD), but operand {a!r} has a "
                    f"basis-polymorphic space {sp!r} that cannot be standardized; "
                    f"annotate/realize its space first"
                )
    else:
        # Still type-check operands (synthesizable) even when no basis constraint applies.
        for a in e.args:
            _synth_value(a, ctx)
    return e.space


def _synth_sum(e: Sum, ctx: Ctx) -> Space:
    first = _synth_value(e.vectors[0], ctx)
    for v in e.vectors[1:]:
        s = _synth_value(v, ctx)
        unify(first, s, ctx.store)
    return ctx.store.resolve(first)


# ===========================================================================
# Application / composition / trace (contraction view).
# ===========================================================================


def _as_factors(s: Space) -> tuple[Space, ...]:
    """Split a map space into its tensor factors; a non-⊗ map is a single factor."""
    if isinstance(s, OProd):
        return s.spaces
    return (s,)


def _synth_eval(e: Eval, ctx: Ctx) -> Space:
    """Type linear application: contract the selected dual factors away.

    ``f`` synthesizes to a product of factors, and each selected slot must be a ``Dual``
    input of some space ``s_j``. The corresponding argument is checked against ``s_j``,
    which recovers it when it was only a fresh metavariable. What is left over after the
    selected factors are removed is the result: no factors at all means the map was fully
    contracted, so the result is ``Scalar``.
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
    """Type non-linear application by walking the ``Π`` chain one slot at a time.

    ``f`` synthesizes to a curried ``Π``, one binder per slot. A value argument is checked
    against its domain space; an explicit space or nat argument is substituted into the
    codomain directly. A space slot with no supplied argument is an *implicit* parameter: it
    gets a fresh metavariable, which later value arguments or the expected type will solve.
    Running out of supplied arguments at a value or nat slot leaves a partial application.
    """
    fty = _synth(e.f, ctx)
    supplied = dict(zip(e.positions, e.args, strict=True))
    result: object = fty
    slot = 0
    # Walk the Π chain by absolute slot index. A space slot with no supplied argument is an
    # implicit parameter, so instantiate a fresh ?α; unification recovers it as later value
    # arguments and the expected type flow in.
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
    match dom:
        case SpaceSort():
            # Value domain: arg is a Term checked against the domain space.
            if not is_term(arg):
                raise InferError(f"NEval: expected a value argument, got {arg!r}")
            _check(arg, dom.space, ctx)  # type: ignore[arg-type]
            return _subst_codomain(fty, arg, ctx)
        case Sort() if dom == SPACE:
            # Space param: explicit space arg, or recover via a fresh metavar.
            sp: Space = arg if is_space(arg) else fresh_meta(fty.binder)  # type: ignore[assignment]
            return _subst_codomain(fty, sp, ctx)
        case Sort() if dom == NAT:
            # Nat param: must be explicit (nats are not unified).
            from grassmann.core.space import is_dim

            if not is_dim(arg):
                raise InferError("NEval: nat parameter requires an explicit nat [n]")
            return _subst_codomain(fty, arg, ctx)
        case _:
            raise InferError(f"NEval: unsupported Π domain {dom!r}")


def _subst_codomain(fty: Pi, value: object, ctx: Ctx) -> object:
    """Substitute the bound variable of ``fty`` by ``value`` in its codomain."""
    cod: Type = fty.codomain
    # SpaceSort domains bind a term Var by name; space/nat domains bind an SVar/DimVar.
    match fty.domain:
        case SpaceSort():
            return _subst_type_term(cod, fty.binder, value, ctx)
        case _ if fty.domain == SPACE:
            return _subst_type_space(cod, fty.binder, value, ctx)
        case _ if fty.domain == NAT:
            return _subst_type_nat(cod, fty.binder, value, ctx)
        case _:
            raise InferError(f"_subst_codomain: unsupported domain {fty.domain!r}")


def _subst_type_space(t: Type, name: str, sp: Space, ctx: Ctx) -> Type:
    return subst_type_space(t, name, sp)  # type: ignore[return-value]


def _subst_type_term(t: Type, name: str, val: object, ctx: Ctx) -> Type:
    # A codomain does not depend on the value bound at its slot, so substituting a value
    # into a classifier changes nothing.
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
    """``Trace(f, i, j)`` — contract primal index i with dual index j."""
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


def _synth_det(e: Det | LogDet, ctx: Ctx) -> Space:
    """``Det(f)``/``LogDet(f)`` — scalar determinant of a *square* endomorphism ``f : V ⊸ V``.

    Requires ``f``'s source space to equal its target space (mirrors :func:`_synth_sqrt`).
    """
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    if dom != cod:
        raise InferError(
            f"{type(e).__name__} requires a square endomorphism (V ⊸ V); got {dom!r} ⊸ {cod!r}"
        )
    return Scalar()


def _synth_opaque(e: OpaqueApply, ctx: Ctx) -> Space:
    """``OpaqueApply(op, arg)`` — apply opaque ``op : dom → cod`` to ``arg``.

    The argument's synthesized space must unify with ``op.dom``; the result is ``op.cod``.
    """
    arg_space = _synth_value(e.arg, ctx)
    try:
        unify(arg_space, e.op.dom, ctx.store)
    except UnifyError as exc:
        raise InferError(
            f"opaque op {e.op.name!r} expects domain {e.op.dom!r}, got argument in {arg_space!r}"
        ) from exc
    return e.op.cod


# ===========================================================================
# Map domain/codomain decoding (for FFS source/target + inverses).
# ===========================================================================


def map_domain_codomain(f: object) -> tuple[Space, Space]:
    """Give the endpoints of a map term — the ``dom`` and ``cod`` of ``f : dom ⊸ cod``.

    The term is synthesized and its map space decoded: a map is stored as the tensor
    ``Dual(dom) ⊗ cod``, so the dual factors give the domain and the primal ones the
    codomain. This is how a fundamental subspace recovers the ambient it sits in when its
    node does not name one, and how the inverse rules learn which way round to build the
    inverse's space.
    """
    ctx = Ctx()
    return _map_dom_cod_resolved(f, ctx)


def _map_dom_cod_resolved(f: object, ctx: Ctx) -> tuple[Space, Space]:
    if not is_term(f):
        raise InferError(f"expected a map Term, got {f!r}")
    s = ctx.store.resolve(_synth_value(f, ctx))  # type: ignore[arg-type]
    return _decode_map_space(s)


def _decode_map_space(s: Space) -> tuple[Space, Space]:
    """Split a map space into ``(dom, cod)``, reading ``Dual`` factors as the inputs.

    A map is stored as ``Dual(dom) ⊗ cod``, so the dual factors are the domain and the
    primal ones the codomain, however many there are of each. A space that is all dual —
    a map into a dual codomain, ``M ⊸ M*`` being ``Dual(M) ⊗ Dual(M)`` — is ambiguous on
    that reading, and is resolved by the positional convention
    :func:`~grassmann.core.space.linear_map_space` builds with: factor 0 is the domain and
    the rest are the codomain.
    """
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
    # An all-dual space is ambiguous under the dual-means-input reading, so fall back to the
    # positional convention ``linear_map_space`` builds with.
    if len(prims) == 0 and len(factors) >= 2 and isinstance(factors[0], Dual):
        cod = factors[1] if len(factors) == 2 else oprod(*factors[1:])
        return factors[0].space, cod
    raise InferError(f"cannot decode map space (dom ⊸ cod): {s!r}")


# ===========================================================================
# Inverses & adjoint.
# ===========================================================================


def _synth_solve(e: Solve, ctx: Ctx) -> Space:
    from .assertions import check_assert_property_consistency

    if not is_bijective(e.f):
        raise InferError("Solve requires the map to be asserted `bij`")
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    # `bij` guarantees |dom| ≡ |cod|; reject an asserted-bij map between unequal-dim
    # spaces up front (asserted facts inform/validate the inverse's space).
    check_assert_property_consistency(e.f, dom, cod)
    return linear_map_space(cod, dom)


def _synth_pinv(e: Pinv, ctx: Ctx) -> Space:
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    # inj/surj sharpen the realization (lowering); the space is cod ⊸ dom regardless.
    _ = (is_injective(e.f), is_surjective(e.f))
    return linear_map_space(cod, dom)


def _synth_corestrict(e: Corestrict, ctx: Ctx) -> Space:
    """Type the corestriction ``f~ : dom(f) ⊸ Img(f)`` — the first-isomorphism-theorem iso.

    The corestriction exists to make the identification ``dom(f) ≅ Img(f)`` explicit, and
    that identification holds exactly when ``ker f = 0``. A map not asserted ``inj`` is
    therefore refused rather than typed: for a general map the induced isomorphism runs out
    of ``CoImg(f)``, a different domain, which this node does not cover.

    The codomain is the bare ``Img(f)``, with endpoints left unset. That is the space
    ``G.img(f)`` builds, and fundamental subspaces unify on the defining map alone, so a
    caller's ``Img(f, source=…, target=…)`` still unifies with it.
    """
    from grassmann.core.space import Img

    if not is_injective(e.f):
        raise InferError(
            "Corestrict requires the map to be asserted `inj` (props=inj): the corestriction "
            "f~ : dom(f) ⊸ Img(f) is the first-isomorphism-theorem isomorphism only when "
            "ker f = 0.  For a general map the induced iso is CoImg(f) ⊸ Img(f), which this "
            f"node deliberately does NOT cover.  Got {e.f!r}"
        )
    dom, _cod = _map_dom_cod_resolved(e.f, ctx)
    return linear_map_space(dom, Img(e.f))


def _synth_sqrt(e: Sqrt, ctx: Ctx) -> Space:
    # The operator square root is an endomorphism of the same (square) space (V ⊸ V).
    dom, cod = _map_dom_cod_resolved(e.f, ctx)
    if dom != cod:
        raise InferError("Sqrt requires a square endomorphism (V ⊸ V)")
    return linear_map_space(dom, cod)


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
# Movement / coercion / reshape.
# ===========================================================================


def _synth_project(e: Project, ctx: Ctx) -> Space:
    from .subspace import is_subspace

    s = ctx.store.resolve(_synth_value(e.vector, ctx))
    if e.space is None:
        # Target subspace omitted. A subspace is not unique, so we cannot solve the
        # target by unification in the ⇒ direction; return a fresh metavar (to be solved
        # by the ⇐ check direction) and defer the ``target ⊑ s`` obligation, validated
        # once the metavar is resolved (see ``_check_subspace_obligations``).
        target = fresh_meta("proj")
        ctx.subspace_obligations.append((target, s, e.vector))
        return target
    if not is_subspace(e.space, s):
        raise InferError(f"project: {e.space!r} is not a subspace of {s!r}")
    return e.space


def _synth_embed(e: Embed, ctx: Ctx) -> Space:
    from .subspace import is_subspace

    s = ctx.store.resolve(_synth_value(e.vector, ctx))
    if not is_subspace(s, e.space):
        raise InferError(f"embed: {s!r} is not a subspace of {e.space!r}")
    return e.space


def _synth_assert(e: Assert, ctx: Ctx) -> Space:
    """Type an assertion: unify the asserted space in, then check it against the properties.

    Unifying ``v``'s synthesized space with the asserted one solves the implicit space
    metavariables ``v`` still carries, which is what makes an assertion useful to inference
    beyond documentation. When ``v`` is a property-bearing map, the dimension equalities its
    ``inj``/``surj``/``bij``/``rank`` guarantee are then checked against the resolved space,
    so a bijection asserted between spaces of unequal dimension is rejected here rather than
    accepted.
    """
    from .assertions import check_assert_property_consistency
    from .properties import map_props

    s = _synth_value(e.vector, ctx)
    unify(s, e.space, ctx.store)
    resolved = ctx.store.resolve(e.space)
    # A property-bearing map validates its own resolved space: the dimension relations its
    # properties guarantee must hold of the decoded domain and codomain.
    if map_props(e.vector) is not None:
        try:
            dom, cod = _decode_map_space(resolved)
        except InferError:
            dom = cod = None  # not a map space ⇒ no map-property facts apply
        if dom is not None and cod is not None:
            check_assert_property_consistency(e.vector, dom, cod)
    return resolved


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


def _spliced_oprod(*parts: Space) -> Space:
    """``Kron``'s endpoint spelling: :func:`grassmann.core.space.spliced_oprod`.

    A ``kron`` chain nests its endpoints — ``kron(f, kron(g, h))`` has domain
    ``oprod(a0, oprod(a1, a2))`` — while ``tensor_power(V, k)`` builds the same space flat.  The
    two meet wherever a Λᵏ ambient is unified against an antisymmetrizer's domain, and ``unify``
    refuses to identify two ⊗ layouts, so ``Kron``'s endpoints are spliced to the flat spelling
    here.  (At ``k = 2`` a chain has no nesting to disagree about; from ``k = 3`` up it does.)

    Splicing the endpoints is layout-neutral, which is why it is done here rather than in
    ``oprod`` itself: the nesting sits inside the ``Dual(dom)`` / ``cod`` factor of the map space
    ``_synth_kron`` returns, and ``_shape_of_space`` counts that factor as one axis whatever its
    interior, so the map keeps its ``(dom, cod)`` matrix shape either way.

    Parameters
    ----------
    *parts
        The endpoint factors to multiply.

    Returns
    -------
    Space
        The flat product, as :func:`grassmann.core.space.spliced_oprod` builds it.
    """
    return spliced_oprod(*parts)


def _synth_kron(e: Kron, ctx: Ctx) -> Space:
    a0, b0 = _map_dom_cod_resolved(e.f, ctx)
    a1, b1 = _map_dom_cod_resolved(e.g, ctx)
    return linear_map_space(_spliced_oprod(a0, a1), _spliced_oprod(b0, b1))


def _synth_lift(e: Lift | InvLift, ctx: Ctx) -> Space:
    """Fuse the listed tensor factors of the vector's space into one block.

    ``Lift`` fuses input factors, taking the ``Dual`` of their product so that ``n``
    separate arguments become one tupled argument — currying, in the contraction view.
    ``InvLift`` is the inverse reshape and fuses primally, flattening a factor that is
    itself a product so that ``(V⊗ᵏ)⊗(V⊗ˡ)`` reassociates to ``V⊗ᵏ⁺ˡ`` rather than staying
    nested. Either way the untouched factors keep their order and the fused block is
    inserted at ``k``, or at the first fused index when ``k`` is not given.
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
        # ``V⊗^{k+l}`` rather than keeping the nesting (the explicit wedge
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


def _synth_permute(e: Permute, ctx: Ctx) -> Space:
    """``Permute(v, perm)`` — reorder ``v``'s tensor factors.

    Result factor ``i`` is ``v``'s factor ``perm[i]``.
    """
    s = ctx.store.resolve(_synth_value(e.vector, ctx))
    factors = list(_as_factors(s))
    n = len(factors)
    if sorted(e.perm) != list(range(n)):
        raise InferError(
            f"Permute: perm {e.perm!r} must be a permutation of range({n}) for a "
            f"{n}-factor space, got space {s!r}"
        )
    out = [factors[e.perm[i]] for i in range(n)]
    if len(out) == 1:
        return out[0]
    return oprod(*out)


def _synth_tuple_permute(e: TuplePermute, ctx: Ctx) -> Space:
    """``TuplePermute(v, perm)`` — reorder ``v``'s ``⊕`` components.

    Result component ``i`` is ``v``'s component ``perm[i]`` (the direct-sum analogue of
    :func:`_synth_permute`).
    """
    s = ctx.store.resolve(_synth_value(e.vector, ctx))
    if not isinstance(s, OPlus):
        raise InferError(f"TuplePermute: value must have a ⊕ space, got {s!r}")
    comps = list(s.spaces)
    n = len(comps)
    if sorted(e.perm) != list(range(n)):
        raise InferError(
            f"TuplePermute: perm {e.perm!r} must be a permutation of range({n}) for a "
            f"{n}-component ⊕, got space {s!r}"
        )
    return oplus(*(comps[e.perm[i]] for i in range(n)))


# ===========================================================================
# Binders.
# ===========================================================================


def _synth_llam(e: LLam, ctx: Ctx) -> Space:
    """Type a linear binder as the map space ``⊗(Dual sᵢ) ⊗ s_body``, and check its resources.

    The space is the obvious one: one dual factor per bound variable, tensored against the
    space of the body. The rule also enforces the discipline that makes the binder *linear*
    — each bound variable must be consumed exactly once along every path of the body, as
    counted by :mod:`grassmann.infer.linearity`. That is why ``x ↦ x ⊕ 0`` is well formed:
    the ``Zero`` slot consumes nothing, so ``x`` is used once.
    """
    from .linearity import check_llam_resources

    check_llam_resources(e)
    s_e = ctx.store.resolve(_synth_value(e.body, ctx))
    dom_factors = [dual(v.space) for v in e.vars]
    return oprod(*dom_factors, s_e)


def _synth_lam(e: Lam, ctx: Ctx) -> SynthResult:
    """Type a general binder by what its body is: a term gives a ``Π``, a space a Kind.

    A term body makes the binder a function between values, classified by a curried ``Π``
    over the binders' sorts. A space, type or dimension body makes it a space, type or nat
    function instead, classified by the corresponding arrow Kind.
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
