"""Term→Term **rewriting**: β-reduction, algebraic simplification, and a
**sum-of-products** normal form (a new pass in the :mod:`grassmann.transform`
family, alongside :mod:`~grassmann.transform.diff` / :mod:`~grassmann.transform.pullback`).

Three public entry points, in increasing strength:

- :func:`beta` — β-reduce applied binders only: ``Eval(LLam …)`` (linear) and
  ``NEval(Lam …)`` (non-linear, value / space binders), incl. partial application.
- :func:`simplify` — β + the always-valid algebraic cleanups (scalar/zero/identity
  laws, sum flattening, involutions). Strictly shrinks a term; never distributes.
- :func:`sum_of_products` (alias :func:`normal_form`) — :func:`simplify` plus the
  **linearity / multilinearity distribution** rules that push every linear operator
  through ``+`` and pull scalars to the front, then collect like terms. The result is a
  flat ``Sum`` of monomials ``c · atom`` where no ``atom`` hides a distributable ``+``.

Everything is **semantics-preserving**: each rule is an equality of the coordinate-free
algebra, so a rewritten term lowers to the same numbers (the test-suite checks this by
running both forms through :func:`grassmann.run`).

The engine itself is a tiny innermost (bottom-up) normalizer (:func:`_normalize`): it
rewrites children to a fixpoint first, then applies the local rules at the root until none
fire. Rules are ``Term -> Term | None`` (``None`` = "I don't apply"). They must be
strictly reducing under the chosen measure; a global step cap (:data:`_MAX_STEPS`) turns a
runaway non-confluent set into a :class:`RewriteError` rather than a hang.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace

from grassmann.core.kinds import NAT, SPACE, SpaceSort
from grassmann.core.space import OProd, Scalar, Space, dual, is_dim, is_space, oplus
from grassmann.core.term import (
    Assert,
    Compose,
    Dagger,
    Embed,
    Eval,
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
    Project,
    Remetric,
    Scale,
    Sharp,
    Solve,
    Sum,
    Term,
    Trace,
    TupleEmbed,
    TupleProject,
    Var,
    Zero,
    is_term,
)

Rule = Callable[[Term], Term | None]


class RewriteError(Exception):
    """Raised when a rewrite cannot proceed (e.g. the engine fails to terminate)."""


# A backstop against a non-terminating / non-confluent rule set: the number of *local*
# rule firings allowed at a single node before we declare non-termination.
_MAX_STEPS = 100_000


@dataclass(frozen=True)
class SoPConfig:
    """Switches that select how aggressively :func:`sum_of_products` normalizes (and what
    :func:`is_normalized_sop` then certifies). Each flag turns a rule group on/off; the
    surface grows as the build ladder lands (contraction-push, ⊕-dissolve, up-to-iso ⊗/⊕
    ordering, …). Confluence is *not* assumed — the engine just runs the chosen rules to a
    fixpoint (`plans/90`)."""

    beta: bool = True            # β-reduce applied binders (Eval(LLam …) / NEval(Lam …))
    distribute: bool = True      # push linear/multilinear ops through + (incl. zero-padded ⊕)
    push_contractions: bool = True  # localize Eval/Compose into the ⊗ factor (x(c)·y)
    collect: bool = True         # collect like monomials (v + v → 2·v, cancellation)
    order_sum: bool = True       # sort + summands by term_key — value-safe (commutative)
    order_otimes: bool = False   # canonical ⊗ factor order — UP-TO-ISO (emits a Permute 𝒞)
    order_oplus: bool = False    # canonical ⊕ component order — UP-TO-ISO (synthesized 𝒞; heavy)
    dissolve_oplus: bool = False  # dissolve ⊕ into a sum of injections (⊕ disappears; + outermost)
    canon_scalars: bool = False  # canonicalize scalar PRODUCTS (commutative ·); partial rational NF
    canon_binders: bool = True   # α-canonicalize bound vars (α-variant terms → identical NF)


# The default preset: the conservative, value-EXACT end of the ladder — only the commutative
# `+` is reordered, so every rewrite is layout-identical (no `𝒞` correction is ever emitted).
DEFAULT_SOP = SoPConfig()

# The aggressive preset: also canonicalizes ⊗ factor order up-to-iso, inserting a `Permute`
# output correction `𝒞` so the value/layout is still restored exactly.
AGGRESSIVE_SOP = SoPConfig(order_otimes=True)


# ===========================================================================
# Generic term traversal (rebuild + substitution).
# ===========================================================================


def _child_terms(node: Term) -> tuple[Term, ...]:
    """The immediate *Term* children of ``node`` (a ``Scale`` scalar may be one)."""
    out: list[Term] = []
    for fld in fields(node):
        val = getattr(node, fld.name)
        for c in (val if isinstance(val, tuple) else (val,)):
            if is_term(c):
                out.append(c)
    return tuple(out)


def _rebuild(node: Term, f: Callable[[Term], Term]) -> Term:
    """Reconstruct ``node`` with ``f`` applied to each *Term* child (spaces, dims, ints,
    strings, the opaque ``External`` payload, and binder sorts pass through untouched).

    Returns ``node`` itself when nothing changed, so identity rewrites don't churn."""
    changed = False
    new_fields: dict[str, object] = {}
    for fld in fields(node):
        val = getattr(node, fld.name)
        if isinstance(val, tuple):
            new_tuple = tuple(f(c) if is_term(c) else c for c in val)
            if any(a is not b for a, b in zip(new_tuple, val, strict=True)):
                changed = True
            new_fields[fld.name] = new_tuple
        elif is_term(val):
            nv = f(val)
            if nv is not val:
                changed = True
            new_fields[fld.name] = nv
        else:
            new_fields[fld.name] = val
    if not changed:
        return node
    return replace(node, **new_fields)


def subst(term: Term, mapping: dict[int, Term]) -> Term:
    """Capture-avoiding substitution of value variables: replace every ``Var`` whose
    ``uid`` is a key of ``mapping`` by the mapped Term.

    Capture-avoidance is automatic: binder variables carry globally fresh ``uid``s, so an
    inner binder never shares a ``uid`` with the (outer) variable being substituted, and a
    substituted-in Term cannot mention a variable bound by a binder we pass under.
    """

    def go(node: Term) -> Term:
        if isinstance(node, Var):
            return mapping.get(node.uid, node)
        return _rebuild(node, go)

    return go(term)


def _subst_space_in_term(term: Term, name: str, repl: Space) -> Term:
    """Substitute the rigid space variable ``SVar(name) := repl`` through every space that
    ``term`` embeds (used by space-binder β). Recurses into Term children and applies
    :func:`grassmann.infer.subst.subst_space` to every Space-typed field."""
    from grassmann.infer.subst import subst_space

    def go(node: Term) -> Term:
        new_fields: dict[str, object] = {}
        changed = False
        for fld in fields(node):
            val = getattr(node, fld.name)
            if isinstance(val, tuple):
                new = tuple(
                    go(c) if is_term(c) else (subst_space(c, name, repl) if is_space(c) else c)
                    for c in val
                )
                if any(a is not b for a, b in zip(new, val, strict=True)):
                    changed = True
                new_fields[fld.name] = new
            elif is_term(val):
                nv = go(val)
                if nv is not val:
                    changed = True
                new_fields[fld.name] = nv
            elif is_space(val):
                nv = subst_space(val, name, repl)
                if nv is not val:
                    changed = True
                new_fields[fld.name] = nv
            else:
                new_fields[fld.name] = val
        return replace(node, **new_fields) if changed else node

    return go(term)


def _subst_nat_in_term(term: Term, name: str, repl: object) -> Term:
    """Substitute the nat parameter ``DimVar(name) := repl`` through every dimension that
    ``term`` embeds (used by nat-binder β). A single generic structural walk over the whole
    Term/Space/Dim tree — every ``DimVar(name)`` (in an ``OPlusRep`` count, an ``App``/``NEval``
    nat arg, a ``DimSum``/``DimProd``/…) is replaced. (Capture under a shadowing inner nat
    binder is not handled in v1 — nat binders are rare; noted in plans/90.)"""
    from grassmann.core.space import DimVar

    def go(node: object) -> object:
        if isinstance(node, DimVar):
            return repl if node.name == name else node
        if not (is_term(node) or is_space(node) or is_dim(node)):
            return node
        new_fields: dict[str, object] = {}
        changed = False
        for fld in fields(node):
            val = getattr(node, fld.name)
            if isinstance(val, tuple):
                new = tuple(
                    go(c) if (is_term(c) or is_space(c) or is_dim(c)) else c for c in val
                )
                if any(a is not b for a, b in zip(new, val, strict=True)):
                    changed = True
                new_fields[fld.name] = new
            elif is_term(val) or is_space(val) or is_dim(val):
                nv = go(val)
                if nv is not val:
                    changed = True
                new_fields[fld.name] = nv
            else:
                new_fields[fld.name] = val
        return replace(node, **new_fields) if changed else node

    return go(term)  # type: ignore[return-value]


def alpha_canon(term: Term) -> Term:
    """α-canonicalize **value-binder** bound variables (``LLam.vars`` / ``Lam.term_vars``):
    rename each to ``%k`` with a canonical *negative* ``uid`` assigned in deterministic
    traversal order. Free variables (non-negative ``uid``s) are untouched and never collide.

    Two α-equivalent terms map to the **identical** term, so it is the basis of an
    α-invariant :func:`term_key` and of `is_normalized_sop`'s binder check. Idempotent.
    (Space/nat binder *names* — ``SVar``/``DimVar`` in ``Lam`` annotations — are left for a
    later refinement; value binders are the common case.)"""
    counter = [0]

    def fresh(v: Var) -> Var:
        counter[0] += 1
        return Var(f"%{counter[0]}", v.space, v.props, uid=-counter[0])

    def go(node: Term, env: dict[int, Var]) -> Term:
        if isinstance(node, Var):
            return env.get(node.uid, node)
        if isinstance(node, LLam):
            env2 = dict(env)
            nv = []
            for v in node.vars:
                cv = fresh(v)
                env2[v.uid] = cv
                nv.append(cv)
            return LLam(tuple(nv), go(node.body, env2))
        if isinstance(node, Lam):
            env2 = dict(env)
            ntv = []
            for v in node.term_vars:
                cv = fresh(v)
                env2[v.uid] = cv
                ntv.append(cv)
            body = go(node.body, env2) if is_term(node.body) else node.body
            return Lam(node.binders, body, tuple(ntv))
        return _rebuild(node, lambda c: go(c, env))

    return go(term, {})


def _has_binder(t: Term) -> bool:
    if isinstance(t, (LLam, Lam, Frame)):
        return True
    return any(_has_binder(c) for c in _child_terms(t))


# ===========================================================================
# Spaces & scalars.
# ===========================================================================


def _space(t: Term) -> Space | None:
    """The synthesized space of ``t`` (or ``None`` if it does not type-check yet) — used
    to give a fresh ``Zero`` its space. ``Zero(None)`` is a valid B1 placeholder, so a
    failure degrades gracefully rather than aborting the rewrite."""
    from grassmann.infer.synth import synth_space

    try:
        return synth_space(t)
    except Exception:
        return None


def _is_num(x: object) -> bool:
    return isinstance(x, (int, float))


def _mul_scalar(a: float | Term, b: float | Term) -> float | Term:
    """Scalar·scalar, collapsing numeric factors and the 0/1 identities. A genuinely
    symbolic product of two scalar *Terms* is represented as ``Scale(a, b)`` (both are
    ``Scalar``-valued, so this is the scalar product)."""
    if _is_num(a) and _is_num(b):
        return a * b
    if _is_num(a):
        if a == 1:
            return b
        if a == 0:
            return 0.0
        return Scale(a, b)  # type: ignore[arg-type]  num · term
    if _is_num(b):
        if b == 1:
            return a
        if b == 0:
            return 0.0
        return Scale(b, a)  # type: ignore[arg-type]  term · num → put the number in the scalar slot
    return Scale(a, b)  # type: ignore[arg-type]  term · term (scalar product)


def _split_num_scale(v: Term) -> tuple[float, Term]:
    """Peel a leading run of *numeric* ``Scale``s off ``v`` → ``(coeff, atom)``. A symbolic
    scalar stops the peel (it stays part of the atom)."""
    c = 1.0
    while isinstance(v, Scale) and _is_num(v.scalar):
        c *= float(v.scalar)
        v = v.vector
    return c, v


# ===========================================================================
# β-reduction rules.
# ===========================================================================


def rule_beta_eval(node: Term) -> Term | None:
    """``Eval(LLam([x₀,…,xₙ₋₁]. body), pos, args)`` — linear β.

    Slot ``p < n`` is the dual input of linear var ``xₚ``; applying ``args[k]`` there
    substitutes ``xₚ := args[k]``. Full application returns the substituted body; a partial
    one returns a residual ``LLam`` over the still-unapplied vars. Positions into the
    *codomain* (``p ≥ n``) are not var redexes and are left alone."""
    if not isinstance(node, Eval) or not isinstance(node.f, LLam):
        return None
    vars_ = node.f.vars
    n = len(vars_)
    if any(p >= n for p in node.positions):
        return None
    mapping = {vars_[p].uid: a for p, a in zip(node.positions, node.args, strict=True)}
    applied = set(node.positions)
    new_body = subst(node.f.body, mapping)
    remaining = tuple(vars_[i] for i in range(n) if i not in applied)
    if not remaining:
        return new_body
    return LLam(remaining, new_body)


def _term_var_for_binder(lam: Lam, idx: int) -> Var | None:
    """The ``Var`` introduced by the ``idx``-th binder, if it is a value (``SpaceSort``)
    binder. ``term_vars`` lists those vars in binder order, skipping space/nat sorts."""
    j = 0
    for i, (_name, sort) in enumerate(lam.binders):
        if isinstance(sort, SpaceSort):
            if i == idx:
                return lam.term_vars[j] if j < len(lam.term_vars) else None
            j += 1
    return None


def rule_beta_neval(node: Term) -> Term | None:
    """``NEval(Lam([b:σ]. body), pos, args)`` — non-linear β.

    Each supplied slot substitutes into the body: a **value** binder (``SpaceSort``)
    replaces its term var by the argument Term; a **space** binder replaces ``SVar(name)``
    by the argument Space throughout the body's spaces; a **nat** binder replaces
    ``DimVar(name)`` by the argument ``Dim`` throughout the body's dimensions. Unapplied
    binders survive as a residual ``Lam`` (partial application)."""
    if not isinstance(node, NEval) or not isinstance(node.f, Lam):
        return None
    lam = node.f
    n = len(lam.binders)
    if any(p >= n for p in node.positions):
        return None
    supplied = dict(zip(node.positions, node.args, strict=True))

    term_map: dict[int, Term] = {}
    body: Term = lam.body  # type: ignore[assignment]
    if not is_term(body):
        return None  # SpaceFunction / TypeFunction body — not a term β redex here.
    consumed: set[int] = set()
    for p, arg in supplied.items():
        name, sort = lam.binders[p]
        if isinstance(sort, SpaceSort):
            v = _term_var_for_binder(lam, p)
            if v is None or not is_term(arg):
                return None
            term_map[v.uid] = arg
            consumed.add(p)
        elif sort == SPACE and is_space(arg):
            body = _subst_space_in_term(body, name, arg)
            consumed.add(p)
        elif sort == NAT and is_dim(arg):
            body = _subst_nat_in_term(body, name, arg)
            consumed.add(p)
        else:
            return None  # unsupported argument kind — leave for later.
    if term_map:
        body = subst(body, term_map)

    remaining = tuple(
        (name, sort) for i, (name, sort) in enumerate(lam.binders) if i not in consumed
    )
    if not remaining:
        return body
    # Rebuild the residual Lam, keeping only the term vars of the surviving value binders.
    rem_term_vars = tuple(
        v
        for i, (_n, sort) in enumerate(lam.binders)
        if i not in consumed
        and isinstance(sort, SpaceSort)
        and (v := _term_var_for_binder(lam, i)) is not None
    )
    return Lam(remaining, body, rem_term_vars)


# ===========================================================================
# Algebraic simplification rules (always strictly reducing).
# ===========================================================================


def rule_scale(node: Term) -> Term | None:
    """Scalar laws: ``0·v → 0``, ``1·v → v``, ``r·0 → 0``, ``r·(s·w) → (r·s)·w``."""
    if not isinstance(node, Scale):
        return None
    r, v = node.scalar, node.vector
    if _is_num(r) and r == 0:
        return Zero(_space(v))
    if _is_num(r) and r == 1:
        return v
    if isinstance(v, Zero):
        return v
    if isinstance(v, Scale):
        # Collapse a nested scale into one coefficient — but a *symbolic scalar product*
        # (inner vector also Scalar-valued) is owned by rule_scalar_product, so defer it to
        # avoid the two rules fighting over where the numeric coefficient nests.
        if (_is_num(r) and _is_num(v.scalar)) or not isinstance(_space(v.vector), Scalar):
            return Scale(_mul_scalar(r, v.scalar), v.vector)
        return None
    return None


def rule_sum(node: Term) -> Term | None:
    """Flatten nested ``Sum``s and drop ``Zero`` summands; collapse to the single summand
    (or a ``Zero``) when that is all that remains."""
    if not isinstance(node, Sum):
        return None
    flat: list[Term] = []
    changed = False
    for v in node.vectors:
        if isinstance(v, Sum):
            flat.extend(v.vectors)
            changed = True
        elif isinstance(v, Zero):
            changed = True
        else:
            flat.append(v)
    if not flat:
        return Zero(_space(node.vectors[0]))
    if len(flat) == 1:
        return flat[0]
    return Sum(tuple(flat)) if changed else None


def _gather_scalar_product(t: Term, num: float, syms: list[Term]) -> float:
    """Flatten a product of scalars (nested ``Scale`` whose operands are all ``Scalar``-valued)
    into a numeric coefficient ``num`` and a list of symbolic scalar-atom factors ``syms``."""
    if isinstance(t, Lit):
        return num * t.value
    if isinstance(t, Scale) and isinstance(_space(t.vector), Scalar):
        if _is_num(t.scalar):
            num *= t.scalar
        else:
            num = _gather_scalar_product(t.scalar, num, syms)  # type: ignore[arg-type]
        return _gather_scalar_product(t.vector, num, syms)
    syms.append(t)  # an atomic scalar factor (a dot-product Eval, External scalar, scalar Var…)
    return num


def rule_scalar_product(node: Term) -> Term | None:
    """Canonicalize a **product of scalars** (the commutative multiplicative monoid): flatten
    nested ``Scale``s whose operands are all ``Scalar``-valued, fold the numeric factor, and
    sort the symbolic factors by ``term_key``. So ``⟨c,x⟩·⟨d,y⟩`` and ``⟨d,y⟩·⟨c,x⟩`` reach the
    same form. Opt-in via ``canon_scalars``.

    This is the *multiplicative* half of a scalar normal form; the full rational ``P/Q`` (sums
    of scalar products + ``div``/``reciprocal`` with cancellation) is left as future work."""
    if not isinstance(node, Scale) or not isinstance(_space(node.vector), Scalar):
        return None  # not a scalar product (a genuine vector scaling, or untypable)
    syms: list[Term] = []
    num = _gather_scalar_product(node, 1.0, syms)
    syms.sort(key=term_key)
    if num == 0:
        return Lit(0.0)
    # Rebuild right-nested with the numeric coefficient (if ≠ 1) at the front.
    if not syms:
        return Lit(num)
    body: Term = syms[-1]
    for s in reversed(syms[:-1]):
        body = Scale(s, body)
    canon = body if num == 1 else Scale(num, body)
    return canon if canon != node else None


def rule_collect(node: Term) -> Term | None:
    """Collect like monomials in a ``Sum``: group summands by atom (after peeling a numeric
    scalar) and add their coefficients. ``v + v → 2·v``; ``2·v + (−2)·v → 0``."""
    if not isinstance(node, Sum):
        return None
    order: list[Term] = []
    coeff: dict[Term, float] = {}
    for v in node.vectors:
        c, atom = _split_num_scale(v)
        if atom in coeff:
            coeff[atom] += c
        else:
            coeff[atom] = c
            order.append(atom)
    if len(order) == len(node.vectors) and all(
        _split_num_scale(v)[0] == coeff[_split_num_scale(v)[1]] for v in node.vectors
    ):
        return None  # nothing merged
    out: list[Term] = []
    for atom in order:
        c = coeff[atom]
        if c == 0:
            continue
        out.append(atom if c == 1 else Scale(c, atom))
    if not out:
        return Zero(_space(node.vectors[0]))
    return out[0] if len(out) == 1 else Sum(tuple(out))


def rule_eval_identity(node: Term) -> Term | None:
    """``Identity(s)·v → v`` (applying the identity map to a single vector)."""
    if (
        isinstance(node, Eval)
        and isinstance(node.f, Identity)
        and node.positions == (0,)
        and len(node.args) == 1
    ):
        return node.args[0]
    return None


def rule_eval_zero(node: Term) -> Term | None:
    """``0·… → 0`` and ``f(…,0,…) → 0`` — a linear application with a zero map or a zero
    argument vanishes."""
    if isinstance(node, Eval) and (
        isinstance(node.f, Zero) or any(isinstance(a, Zero) for a in node.args)
    ):
        return Zero(_space(node))
    return None


def rule_compose(node: Term) -> Term | None:
    """Identity / zero laws for composition: drop ``Identity`` operands, ``Identity ∘ g →
    g``, and ``0 ∘ … → 0`` / ``… ∘ 0 → 0``."""
    if not isinstance(node, Compose):
        return None
    if isinstance(node.f, Identity) and node.positions == (0,) and len(node.gs) == 1:
        return node.gs[0]
    if isinstance(node.f, Zero) or any(isinstance(g, Zero) for g in node.gs):
        return Zero(_space(node))
    keep_pos: list[int] = []
    keep_gs: list[Term] = []
    dropped = False
    for p, g in zip(node.positions, node.gs, strict=True):
        if isinstance(g, Identity):
            dropped = True
        else:
            keep_pos.append(p)
            keep_gs.append(g)
    if not dropped:
        return None
    if not keep_gs:
        return node.f
    return Compose(node.f, tuple(keep_pos), tuple(keep_gs))


def rule_involution(node: Term) -> Term | None:
    """``(fᵀ)ᵀ → f`` (``Dagger``) and ``(f⁻¹)⁻¹ → f`` (``Solve``) — involutive operators."""
    if isinstance(node, Dagger) and isinstance(node.f, Dagger):
        return node.f.f
    if isinstance(node, Solve) and isinstance(node.f, Solve):
        return node.f.f
    return None


def rule_dagger_push(node: Term) -> Term | None:
    """Push the metric-transpose adjoint ``Dagger`` toward the leaves so it only ever wraps a
    primitive map (the grammar's ``atom``): ``Id† → Id``, ``(f⊗g)† → f†⊗g†`` (``Kron``),
    ``(f∘g)† → g†∘f†`` (``Compose``, single slot). Value-exact (verified). ``(f†)† → f`` is
    handled by :func:`rule_involution`; ``Dagger`` over ``+``/``Scale`` by :func:`dist_unary`.

    Not done for ``Solve``/``Pinv``: ``Solve``'s argument must be a ``bij``-asserted *var* (it
    can never be a ``Compose``/``Kron``), and ``Pinv`` does **not** distribute over composition
    (the pseudo-inverse of a product is not the product of pseudo-inverses)."""
    if not isinstance(node, Dagger):
        return None
    f = node.f
    if isinstance(f, Identity):
        return f
    if isinstance(f, Kron):
        return Kron(Dagger(f.f), Dagger(f.g))
    if isinstance(f, Compose) and f.positions == (0,) and len(f.gs) == 1:
        return Compose(Dagger(f.gs[0]), (0,), (Dagger(f.f),))
    return None


def rule_musical(node: Term) -> Term | None:
    """Canonicalize the musical isomorphisms to a single metric primitive ``g = Inner(s)``:
    ``♭ = Flat(s) → Inner(s)`` (same space ``s*⊗s*`` and same data), ``♯ = Sharp(s) →
    Pinv(Inner(s))`` (the inverse metric), and cancel an inverse-metric pair in a composition
    — ``g⁻¹∘g → Id(s)``, ``g∘g⁻¹ → Id(s*)`` (a metric is non-degenerate, so ``Pinv(Inner)`` is
    a true inverse). Direction: collapse musical maps toward ``Inner``/``Identity`` (value-exact)."""
    if isinstance(node, Flat):
        return Inner(node.space)
    if isinstance(node, Sharp):
        return Pinv(Inner(node.space))
    if isinstance(node, Compose) and node.positions == (0,) and len(node.gs) == 1:
        f, g = node.f, node.gs[0]
        if isinstance(f, Pinv) and isinstance(f.f, Inner) and isinstance(g, Inner) and f.f.space == g.space:
            return Identity(g.space)
        if isinstance(f, Inner) and isinstance(g, Pinv) and isinstance(g.f, Inner) and f.space == g.f.space:
            return Identity(dual(f.space))
    return None


def rule_permute(node: Term) -> Term | None:
    """Collapse reshape ``Permute`` corrections: an identity permutation vanishes, and two
    nested permutations compose. Value-exact (a permutation of tensor axes). Keeps the
    up-to-iso ``𝒞`` corrections (Step C) from stacking up into non-canonical
    ``Permute(Permute(…))`` towers."""
    if not isinstance(node, Permute):
        return None
    if list(node.perm) == list(range(len(node.perm))):
        return node.vector
    if isinstance(node.vector, Permute):
        p, q = node.vector.perm, node.perm  # Permute(Permute(v, p), q)_i = v[p[q[i]]]
        return Permute(node.vector.vector, tuple(p[i] for i in q))
    return None


# ===========================================================================
# Linearity / multilinearity distribution (toward sum-of-products).
# ===========================================================================


def _push_sum_then_scale(node: Term, slots: list[tuple[Term, Callable[[Term], Term]]]) -> Term | None:
    """Generic distribution over a node that is **multilinear** in the given ``slots``
    (each ``slot`` is a child plus a rebuild-with-new-child closure).

    Pushes a ``Sum`` outward (preferred — moves ``+`` toward the root) before pulling a
    scalar outward (``Scale``). Doing one slot per call lets the engine iterate to a
    fixpoint; trying sums across all slots first keeps the result a clean SoP."""
    for child, rebuild in slots:
        if isinstance(child, Sum):
            return Sum(tuple(rebuild(c) for c in child.vectors))
    for child, rebuild in slots:
        if isinstance(child, Scale):
            return Scale(child.scalar, rebuild(child.vector))
    return None


def dist_scale(node: Term) -> Term | None:
    """``r·(a+b) → r·a + r·b`` and ``(p+q)·v → p·v + q·v`` — ``Scale`` is bilinear."""
    if not isinstance(node, Scale):
        return None
    r, v = node.scalar, node.vector
    if isinstance(v, Sum):
        return Sum(tuple(Scale(r, x) for x in v.vectors))
    if is_term(r) and isinstance(r, Sum):
        return Sum(tuple(Scale(p, v) for p in r.vectors))
    return None


def dist_eval(node: Term) -> Term | None:
    """Multilinearity of linear application in the map and every argument."""
    if not isinstance(node, Eval):
        return None
    slots: list[tuple[Term, Callable[[Term], Term]]] = [
        (node.f, lambda g: Eval(g, node.positions, node.args))
    ]
    for k in range(len(node.args)):
        def rb(a: Term, k: int = k) -> Term:
            new = list(node.args)
            new[k] = a
            return Eval(node.f, node.positions, tuple(new))

        slots.append((node.args[k], rb))
    return _push_sum_then_scale(node, slots)


def dist_compose(node: Term) -> Term | None:
    """Multilinearity of composition in ``f`` and every ``gₖ``."""
    if not isinstance(node, Compose):
        return None
    slots: list[tuple[Term, Callable[[Term], Term]]] = [
        (node.f, lambda h: Compose(h, node.positions, node.gs))
    ]
    for k in range(len(node.gs)):
        def rb(g: Term, k: int = k) -> Term:
            new = list(node.gs)
            new[k] = g
            return Compose(node.f, node.positions, tuple(new))

        slots.append((node.gs[k], rb))
    return _push_sum_then_scale(node, slots)


def dist_oprod(node: Term) -> Term | None:
    """``⊗`` is multilinear: distribute over a ``+`` in any factor, pull a scalar from any
    factor, and vanish if any factor is ``0``."""
    if not isinstance(node, OProdV):
        return None
    if any(isinstance(v, Zero) for v in node.vectors):
        return Zero(_space(node))
    slots: list[tuple[Term, Callable[[Term], Term]]] = []
    for k in range(len(node.vectors)):
        def rb(x: Term, k: int = k) -> Term:
            new = list(node.vectors)
            new[k] = x
            return OProdV(tuple(new))

        slots.append((node.vectors[k], rb))
    return _push_sum_then_scale(node, slots)


def dist_oplus(node: Term) -> Term | None:
    """Pull a ``+`` out of one ``⊕`` (``OPlusV``) slot by **zero-padding** — ``⊕`` is a
    *stack*, not additive per component, so the other slots must be ZEROED in the extra
    copies, not duplicated: ``(a+b)⊕c → a⊕c + b⊕0`` (NOT ``a⊕c + b⊕c``, which doubles ``c``).

    The first addend keeps the original siblings; each further addend gets a copy with that
    slot's value and ``Zero`` everywhere else. Keeps ``⊕`` as a layer whose components are
    ``+``-free (the v2 ``+ > ⊕ > ⊗`` grammar)."""
    if not isinstance(node, OPlusV):
        return None
    for k, v in enumerate(node.vectors):
        if isinstance(v, Sum):
            parts: list[Term] = []
            first = list(node.vectors)
            first[k] = v.vectors[0]
            parts.append(OPlusV(tuple(first)))
            for addend in v.vectors[1:]:
                slot = [Zero(_space(s)) for s in node.vectors]
                slot[k] = addend
                parts.append(OPlusV(tuple(slot)))
            return Sum(tuple(parts))
    return None


def dist_kron(node: Term) -> Term | None:
    """``Kron`` (map ⊗) is bilinear in its two maps; vanishes if either is ``0``."""
    if not isinstance(node, Kron):
        return None
    if isinstance(node.f, Zero) or isinstance(node.g, Zero):
        return Zero(_space(node))
    slots = [
        (node.f, lambda f: Kron(f, node.g)),
        (node.g, lambda g: Kron(node.f, g)),
    ]
    return _push_sum_then_scale(node, slots)


# ---------------------------------------------------------------------------
# Contraction-push: localize Eval/Compose into the ⊗ factor it acts on.
#
# An `Eval`/`Compose` on `OProdV(g₀,…,gₘ)` indexes ONE slot per value-factor (slot i ↔ gᵢ
# as a whole). For the term to type-check, a contracted slot must be a single Dual factor —
# so an Eval'd factor is always fully consumed to a SCALAR (the `x(c)·y` rule), and a
# Compose'd factor stays a covector. No flattened-arity bookkeeping is needed.
# ---------------------------------------------------------------------------


def dist_eval_oprod(node: Term) -> Term | None:
    """Push a contraction into a tensor product: ``Eval(⊗(g₀,…,gₘ), pos, args)`` contracts
    each selected factor fully (it is a single dual ⇒ a scalar ``gᵢ(argᵢ)``) and pulls it
    out as a coefficient — the ``Eval(x⊗y, c) → x(c)·y`` rule. Bails if a contracted factor
    does not reduce to a scalar (an atypical / untypable slot)."""
    if not isinstance(node, Eval) or not isinstance(node.f, OProdV):
        return None
    gs = node.f.vectors
    contracted = dict(zip(node.positions, node.args, strict=True))
    if any(p >= len(gs) for p in contracted):
        return None
    coeffs: list[Term] = []
    factors: list[Term] = []
    for i, g in enumerate(gs):
        if i in contracted:
            cf = Eval(g, (0,), (contracted[i],))
            if not isinstance(_space(cf), Scalar):
                return None
            coeffs.append(cf)
        else:
            factors.append(g)
    if not factors:  # every factor contracted → the product of the scalar coefficients
        body: Term = coeffs[0]
        for cf in coeffs[1:]:
            body = Scale(cf, body)
        return body
    body = OProdV(tuple(factors)) if len(factors) > 1 else factors[0]
    for cf in coeffs:
        body = Scale(cf, body)
    return body


def dist_trace_oprod(node: Term) -> Term | None:
    """Push a ``Trace`` into a tensor product. In ``⊗(g₀,…,gₘ)`` each ``Trace`` index is a
    whole factor, so contracting primal slot ``i`` with dual slot ``j`` is a contraction
    *between* two atoms — represented as ``Eval(gⱼ, gᵢ)`` (the user's choice) — yielding a
    scalar coefficient times the remaining product: ``Trace(x⊗c⊗z, 0, 1) → ⟨c,x⟩·z``."""
    if not isinstance(node, Trace) or not isinstance(node.f, OProdV):
        return None
    gs = node.f.vectors
    i, j = node.i, node.j
    if i >= len(gs) or j >= len(gs) or i == j:
        return None
    coeff = Eval(gs[j], (0,), (gs[i],))  # j is the dual slot, i the primal (Trace convention)
    if not isinstance(_space(coeff), Scalar):
        coeff = Eval(gs[i], (0,), (gs[j],))  # tolerate swapped primal/dual roles
        if not isinstance(_space(coeff), Scalar):
            return None
    rest = [g for k, g in enumerate(gs) if k not in (i, j)]
    if not rest:
        return coeff
    body = OProdV(tuple(rest)) if len(rest) > 1 else rest[0]
    return Scale(coeff, body)


def dist_compose_oprod(node: Term) -> Term | None:
    """Push a composition into a tensor product: ``Compose(⊗(g₀,…,gₘ), pos, hs)`` precomposes
    each selected factor (a single dual input slot) with its map ``hₖ``, leaving the other
    factors untouched."""
    if not isinstance(node, Compose) or not isinstance(node.f, OProdV):
        return None
    gs = node.f.vectors
    composed = dict(zip(node.positions, node.gs, strict=True))
    if any(p >= len(gs) for p in composed):
        return None
    factors = [
        Compose(g, (0,), (composed[i],)) if i in composed else g for i, g in enumerate(gs)
    ]
    return OProdV(tuple(factors)) if len(factors) > 1 else factors[0]


def _unary_linear(node: Term, child: Term, rebuild: Callable[[Term], Term]) -> Term | None:
    """Distribution for a node **linear in one child**: ``L(a+b) → L(a)+L(b)``,
    ``L(r·v) → r·L(v)``, ``L(0) → 0``."""
    if isinstance(child, Zero):
        return Zero(_space(node))
    if isinstance(child, Sum):
        return Sum(tuple(rebuild(c) for c in child.vectors))
    if isinstance(child, Scale):
        return Scale(child.scalar, rebuild(child.vector))
    return None


def dist_unary(node: Term) -> Term | None:
    """Single-child linear operators: ``Dagger`` / ``Trace`` / the movement & reshape nodes
    (``Project``/``Embed``/``Assert``/``Lift``/``InvLift``/``Permute``/``Remetric``)."""
    if isinstance(node, Dagger):
        return _unary_linear(node, node.f, Dagger)
    if isinstance(node, Trace):
        return _unary_linear(node, node.f, lambda g: Trace(g, node.i, node.j))
    if isinstance(node, Project):
        return _unary_linear(node, node.vector, lambda v: Project(v, node.space))
    if isinstance(node, Embed):
        return _unary_linear(node, node.vector, lambda v: Embed(v, node.space))
    if isinstance(node, Assert):
        return _unary_linear(node, node.vector, lambda v: Assert(v, node.space))
    if isinstance(node, Lift):
        return _unary_linear(node, node.vector, lambda v: Lift(v, node.fused, node.k))
    if isinstance(node, InvLift):
        return _unary_linear(node, node.vector, lambda v: InvLift(v, node.fused, node.k))
    if isinstance(node, Permute):
        return _unary_linear(node, node.vector, lambda v: Permute(v, node.perm))
    if isinstance(node, Remetric):
        return _unary_linear(node, node.vector, lambda v: Remetric(v, node.metric))
    return None


# ===========================================================================
# Rule sets and the normalization engine.
# ===========================================================================

_BETA_RULES: tuple[Rule, ...] = (rule_beta_eval, rule_beta_neval)

# The always-valid algebraic cleanups (scalar/zero/identity laws, sum flattening,
# involutions) — strictly reducing, run in every preset.
_CLEANUP_RULES: tuple[Rule, ...] = (
    rule_eval_identity,
    rule_eval_zero,
    rule_compose,
    rule_involution,
    rule_dagger_push,
    rule_musical,
    rule_permute,
    rule_scale,
    rule_sum,
)

_SIMPLIFY_RULES: tuple[Rule, ...] = (*_BETA_RULES, *_CLEANUP_RULES)

_DISTRIBUTE_RULES: tuple[Rule, ...] = (
    dist_scale,
    dist_eval,
    dist_compose,
    dist_oprod,
    dist_oplus,
    dist_kron,
    dist_unary,
)

# Contraction-push: localize Eval/Compose/Trace into the ⊗ factor that owns each slot.
_CONTRACTION_RULES: tuple[Rule, ...] = (dist_eval_oprod, dist_compose_oprod, dist_trace_oprod)


# ===========================================================================
# Sum-of-products normal form: the canonical ordering key and its predicate.
#
# Grammar (what :func:`sum_of_products` aims to produce, and what
# :func:`is_normalized_sop` certifies):
#
#   nf     ::= Zero                              -- the additive unit
#           |  mono                              -- a single monomial
#           |  Sum(mono₁, …, monoₖ)              -- k ≥ 2; the monomials' atoms are
#                                                --   DISTINCT and sorted by `term_key`
#   mono   ::= atom                              -- coefficient 1 (implicit)
#           |  Scale(coeff, atom)                -- coeff ∉ {0, 1}
#   coeff  ::= r ∈ ℝ∖{0,1}                        -- a numeric scalar, or
#           |  s                                 -- a Scalar-valued atom (no top-level +)
#   atom   ::= a Term whose head is NOT Sum / Scale / Zero, is NOT a β-redex
#              (Eval(LLam …) / NEval(Lam …)) nor Eval(Identity …), and ALL of whose
#              subterms are themselves fully reduced — no distributable `+` and no
#              pullable scalar sits in any linear position.
#
# `term_key` is the total order that makes the form canonical (α-equal terms ⇒ equal
# key ⇒ identical sorted output). It is `repr`-based today — deterministic and total —
# and is the single hook step 1 swaps for a structural key if needed.
# ===========================================================================


def term_key(t: Term) -> str:
    """A total, deterministic, **α-invariant** ordering key on Terms (used to sort ``Sum``
    summands canonically). ``repr`` is injective over these frozen dataclasses; for terms
    that contain a binder we key on the α-canonical form so α-equivalent atoms sort
    identically (the common no-binder case takes the fast path)."""
    return repr(alpha_canon(t) if _has_binder(t) else t)


def _atom_of(mono: Term) -> Term:
    """The atom of a monomial (strip a leading ``Scale`` coefficient)."""
    return mono.vector if isinstance(mono, Scale) else mono


def rule_sort_sum(node: Term) -> Term | None:
    """Order ``Sum`` summands into canonical order by ``term_key`` of their atoms. Sorting
    by the *atom* (not the whole monomial) is well-defined once :func:`rule_collect` has made
    atoms distinct. Reordering ``+`` is value-safe — addition is commutative and same-space."""
    if not isinstance(node, Sum):
        return None
    monos = node.vectors
    order = sorted(range(len(monos)), key=lambda i: term_key(_atom_of(monos[i])))
    if order == list(range(len(monos))):
        return None
    return Sum(tuple(monos[i] for i in order))


def _top_arity(g: Term) -> int | None:
    """Number of top-level tensor factors of ``g``'s space (1 for a non-⊗), or ``None`` if
    untypable."""
    sp = _space(g)
    if sp is None:
        return None
    return len(sp.spaces) if isinstance(sp, OProd) else 1


def rule_order_otimes(node: Term) -> Term | None:
    """Canonicalize the factor order of a tensor product **up to isomorphism**, restoring the
    original layout with a ``Permute`` correction (the ``𝒞`` of ``plans/90``):

        ``OProdV(factors)  ≡  Permute(OProdV(sorted), σ⁻¹)``

    where ``sorted[i] = factors[σ[i]]`` (``σ`` = argsort by ``term_key``). Reordering ``⊗``
    factors changes the space (``s_a⊗s_b ≠ s_b⊗s_a``) and the array axes, so it is *not*
    value-safe on its own — the ``Permute`` swaps the axes back, making the whole rewrite
    layout-exact (verified by run-equality).

    Conservative: only fires for an unsorted product of **single-axis** factors that are
    already clean atoms (no ``Sum``/``Scale``/``Zero`` to distribute first)."""
    if not isinstance(node, OProdV):
        return None
    facs = node.vectors
    if any(isinstance(f, (Sum, Scale, Zero)) for f in facs):
        return None  # let distribution/scalar-pull run first
    keys = [term_key(f) for f in facs]
    sigma = sorted(range(len(facs)), key=lambda i: keys[i])
    if sigma == list(range(len(facs))):
        return None  # already canonical
    if any((_top_arity(f) or 2) != 1 for f in facs):
        return None  # multi-axis factor — Permute-on-blocks deferred (Step C+)
    sorted_facs = tuple(facs[i] for i in sigma)
    inv = [0] * len(sigma)  # perm = σ⁻¹ : Permute(OProdV(sorted), inv) reconstructs the original
    for position, i in enumerate(sigma):
        inv[i] = position
    return Permute(OProdV(sorted_facs), tuple(inv))


def rule_order_oplus(node: Term) -> Term | None:
    """Canonicalize the component order of a ``⊕`` stack **up to isomorphism**, restoring the
    original block layout with a synthesized ``⊕``-block-permutation map (no IR reshape node
    exists for ``⊕``, so the correction is built from ``TupleEmbed``/``TupleProject``):

        ``OPlusV(comps)  ≡  P · OPlusV(sorted)``,
        ``P = Σᵢ ιᵢ^{orig} ∘ π_{σ⁻¹[i]}^{sorted} : (⊕ sorted) ⊸ (⊕ orig)``

    where ``sorted[k] = comps[σ[k]]``. Heavyweight (the correction is a sum of
    inject∘project maps, which then distributes) — hence opt-in via ``order_oplus``.
    Conservative: clean components only (no ``Sum``/``Scale``/``Zero`` to distribute first)."""
    if not isinstance(node, OPlusV):
        return None
    comps = node.vectors
    if any(isinstance(c, (Sum, Scale, Zero)) for c in comps):
        return None
    spaces = [_space(c) for c in comps]
    if any(s is None for s in spaces):
        return None
    sigma = sorted(range(len(comps)), key=lambda i: term_key(comps[i]))
    if sigma == list(range(len(comps))):
        return None  # already canonical
    orig_space = oplus(*spaces)  # type: ignore[arg-type]
    sorted_space = oplus(*[spaces[i] for i in sigma])  # type: ignore[arg-type]
    inv = [0] * len(sigma)
    for k, i in enumerate(sigma):
        inv[i] = k
    perm_map = Sum(
        tuple(
            Compose(TupleEmbed(orig_space, i), (0,), (TupleProject(sorted_space, inv[i]),))
            for i in range(len(comps))
        )
    )
    return Eval(perm_map, (0,), (OPlusV(tuple(comps[i] for i in sigma)),))


def rule_dissolve_oplus(node: Term) -> Term | None:
    """Dissolve a ``⊕`` stack into a sum of single-slot injections — the *most aggressive*
    ``⊕`` treatment: ``OPlusV(c₀,…,cₘ) → Σᵢ ιᵢ(cᵢ)`` where ``ιᵢ = Eval(TupleEmbed(⊕space, i))``
    zero-pads component ``i`` into the full space. ``⊕`` then disappears as a former and ``+``
    is strictly outermost (value-preserving). Opt-in via ``dissolve_oplus`` (mutually
    exclusive in spirit with ``order_oplus``)."""
    if not isinstance(node, OPlusV):
        return None
    comps = node.vectors
    spaces = [_space(c) for c in comps]
    if any(s is None for s in spaces):
        return None
    full = oplus(*spaces)  # type: ignore[arg-type]
    return Sum(tuple(Eval(TupleEmbed(full, i), (0,), (comps[i],)) for i in range(len(comps))))


def _sop_rules(cfg: SoPConfig) -> tuple[Rule, ...]:
    """Assemble the rule tuple selected by ``cfg`` (the always-valid algebraic cleanups run
    unconditionally; everything else is gated)."""
    rules: list[Rule] = []
    if cfg.beta:
        rules += _BETA_RULES
    rules += _CLEANUP_RULES
    if cfg.distribute:
        rules += _DISTRIBUTE_RULES
    if cfg.push_contractions:
        rules += _CONTRACTION_RULES
    if cfg.canon_scalars:
        rules.append(rule_scalar_product)
    if cfg.collect:
        rules.append(rule_collect)
    if cfg.order_otimes:
        rules.append(rule_order_otimes)
    if cfg.order_oplus:
        rules.append(rule_order_oplus)
    if cfg.dissolve_oplus:
        rules.append(rule_dissolve_oplus)
    if cfg.order_sum:
        rules.append(rule_sort_sum)
    return tuple(rules)


def _is_atom_head(a: Term) -> bool:
    """``a``'s *head* is a legal atom head: not additive/scalable/zero, and not a redex the
    normal form forbids (β or ``Id·v``)."""
    if isinstance(a, (Sum, Scale, Zero)):
        return False
    if isinstance(a, Eval) and isinstance(a.f, (LLam, Identity)):
        return False
    if isinstance(a, NEval) and isinstance(a.f, Lam):
        return False
    return True


def _is_monomial(mono: Term) -> bool:
    """``mono`` is ``atom`` or ``Scale(coeff, atom)`` with ``coeff ∉ {0, 1}`` (numeric) or a
    sum-free scalar term (symbolic), and ``atom`` a legal atom head."""
    if isinstance(mono, (Sum, Zero)):
        return False
    if isinstance(mono, Scale):
        c = mono.scalar
        if _is_num(c):
            if c == 0 or c == 1:
                return False
        elif isinstance(c, Sum):
            return False  # a distributable scalar sum must have been pushed out
        return _is_atom_head(mono.vector)
    return _is_atom_head(mono)


def _fully_reduced(t: Term, rules: tuple[Rule, ...]) -> bool:
    """No rule in ``rules`` fires anywhere in ``t`` (the operational "no redex remains" half
    of the normal-form check; the additive *shape* is checked separately by
    :func:`is_normalized_sop`)."""
    for rule in rules:
        out = rule(t)
        if out is not None and out != t:
            return False
    return all(_fully_reduced(c, rules) for c in _child_terms(t))


def is_normalized_sop(t: Term, config: SoPConfig = DEFAULT_SOP) -> bool:
    """Decide whether ``t`` is in **sum-of-products normal form** under ``config`` (the
    grammar above).

    Two independent checks: (1) the additive *shape* — ``Zero``, a lone monomial, or a
    ``Sum`` of ≥2 monomials whose atoms are distinct (when ``collect``) and sorted by
    :func:`term_key` (when ``order_sum``); and (2) *full reduction* — no enabled rule fires
    anywhere. Together they make ``is_normalized_sop(sum_of_products(t, c), c)`` the
    acceptance test for the normalizer.
    """
    if not is_term(t):
        return False
    if isinstance(t, Zero):
        return True
    if isinstance(t, Sum):
        monos = t.vectors
        if len(monos) < 2 or not all(_is_monomial(m) for m in monos):
            return False
        keys = [term_key(_atom_of(m)) for m in monos]
        if config.collect and len(set(keys)) != len(keys):  # atoms not collected
            return False
        if config.order_sum and keys != sorted(keys):  # not in canonical order
            return False
    elif not _is_monomial(t):
        return False
    if config.canon_binders and _has_binder(t) and alpha_canon(t) != t:
        return False  # bound variables are not in α-canonical form
    return _fully_reduced(t, _sop_rules(config))


def _normalize(term: Term, rules: tuple[Rule, ...]) -> Term:
    """Innermost (bottom-up) normalization to a fixpoint of ``rules``."""
    node = _rebuild(term, lambda c: _normalize(c, rules))
    steps = 0
    while True:
        for rule in rules:
            new = rule(node)
            if new is not None and new != node:
                # The rule's result may have un-normalized children → renormalize.
                node = _rebuild(new, lambda c: _normalize(c, rules))
                steps += 1
                if steps > _MAX_STEPS:
                    raise RewriteError(
                        f"rewrite did not terminate after {_MAX_STEPS} steps at {type(node).__name__}; "
                        f"the rule set may be non-confluent"
                    )
                break
        else:
            return node


# ===========================================================================
# Public API.
# ===========================================================================


def rewrite(term: Term, rules: tuple[Rule, ...]) -> Term:
    """Normalize ``term`` to a fixpoint of a custom ``rules`` tuple (the open-ended entry
    point; :func:`simplify` / :func:`sum_of_products` are the curated presets)."""
    if not is_term(term):
        raise RewriteError(f"rewrite: expected a Term, got {term!r}")
    return _normalize(term, rules)


def beta(term: Term) -> Term:
    """β-reduce applied binders (``Eval(LLam …)`` / ``NEval(Lam …)``) to a fixpoint, with
    no other simplification."""
    return rewrite(term, _BETA_RULES)


def simplify(term: Term) -> Term:
    """β-reduce and apply the always-valid algebraic cleanups (scalar/zero/identity laws,
    sum flattening, involutions). Strictly shrinks the term; never distributes ``+``."""
    return rewrite(term, _SIMPLIFY_RULES)


def sum_of_products(term: Term, config: SoPConfig = DEFAULT_SOP) -> Term:
    """Rewrite to **sum-of-products normal form** under ``config``: β + the algebraic
    cleanups, distribute every linear/multilinear operator through ``+`` (incl. the
    zero-padded ``⊕`` split), pull scalars to the front, collect like terms, and sort the
    summands canonically. The result is a ``Zero``, a single monomial, or a flat, canonically
    ordered ``Sum`` of monomials ``c · atom`` with no distributable ``+`` left inside."""
    out = rewrite(term, _sop_rules(config))
    if config.canon_binders and _has_binder(out):
        out = alpha_canon(out)
    return out


# A readable alias.
normal_form = sum_of_products


__all__ = [
    "Rule",
    "RewriteError",
    "SoPConfig",
    "DEFAULT_SOP",
    "AGGRESSIVE_SOP",
    "subst",
    "rewrite",
    "beta",
    "simplify",
    "sum_of_products",
    "normal_form",
    "term_key",
    "is_normalized_sop",
    "alpha_canon",
]
