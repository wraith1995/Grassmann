"""Rewriting a term into an equal but simpler one, up to a sum-of-products normal form.

Every rule here is an equality of the coordinate-free algebra, so a rewritten term denotes
the same value and lowers to the same numbers. What varies is how far the rewriting goes.
Three entry points sit on a ladder of increasing strength::

    beta              β-reduce applied binders and nothing else
      │                   Eval(LLam …), NEval(Lam …), partial application included
      ▼
    simplify          β + the always-valid cleanups
      │                   scalar/zero/identity laws, Sum flattening, involutions,
      │                   η, inverse cancellation, musical collapse
      │               strictly shrinks the term; never distributes a +
      ▼
    sum_of_products   simplify + distribution, collection and ordering
                          push every linear and multilinear operator through +,
                          pull scalars to the front, add like monomials, sort
                      result: Zero, one monomial, or a flat Sum of c·atom with no
                      distributable + left anywhere inside

Above ``simplify`` the behaviour is configurable rather than fixed. A :class:`SoPConfig`
turns each rule group on or off, from the value-exact default through ordering the ⊗
factors and dissolving ``⊕`` into injections, and :func:`is_normalized_sop` certifies
exactly what the enabled groups were supposed to produce. That predicate is the acceptance
test: ``is_normalized_sop(sum_of_products(t, cfg), cfg)`` is what "normalized" means here,
since the rule set is not assumed confluent.

Some of the stronger rules are canonical only **up to isomorphism**. Reordering the factors
of a ``⊗`` changes the space and the array axes, so it is not value-safe on its own; the
rule emits a ``Permute`` that swaps the axes back, and the pair is exact. The ``⊕``
analogue emits a ``TuplePermute``. A reordering that could not be corrected this way is not
performed.

The engine is a small innermost normalizer (:func:`_normalize`): it rewrites a node's
children to a fixpoint, then applies the local rules at the root until none fire. A rule is
a ``Term -> Term | None``, where ``None`` means it does not apply, and each is strictly
reducing under its own measure. Confluence across a whole rule set is not assumed, so a
global step cap (:data:`_MAX_STEPS`) turns a rule set that would loop into a
:class:`RewriteError` rather than a hang.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import assert_never

from grassmann.core.kinds import NAT, SPACE, SpaceSort
from grassmann.core.space import (
    OProd,
    Scalar,
    Space,
    dual,
    is_dim,
    is_space,
    linear_map_space,
    oplus,
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
    TuplePermute,
    TupleProject,
    Var,
    Zero,
    is_term,
)

Rule = Callable[[Term], Term | None]


class RewriteError(Exception):
    """Raised when normalization cannot finish — most often the step cap being hit.

    Reaching :data:`_MAX_STEPS` local firings at one node means the enabled rules cycle
    against each other rather than reducing, so the failure is reported instead of hung on.
    """


# The backstop against a rule set that cycles instead of reducing: the number of local rule
# firings allowed at a single node before normalization is declared non-terminating.
_MAX_STEPS = 100_000


@dataclass(frozen=True)
class SoPConfig:
    """How far :func:`sum_of_products` goes — one flag per rule group.

    The flags form a ladder rather than a set of independent switches. The defaults are the
    value-exact rungs: distribution, contraction-push, collection and sorting the
    commutative ``+`` all preserve the layout exactly, so nothing has to be corrected
    afterwards. The opt-in rungs go further and pay for it. Ordering ⊗ factors or ⊕
    components is canonical only up to isomorphism, so those rules emit a ``Permute`` or
    ``TuplePermute`` correction that restores the original layout; dissolving ``⊕`` removes
    the former entirely in favour of a sum of injections; and the advanced η rule folds a
    seed-linear lambda into the covector or map it denotes.

    Whatever is enabled, :func:`is_normalized_sop` certifies exactly that: it checks the
    additive shape and that no enabled rule still fires. The engine runs the chosen rules to
    a fixpoint and assumes no confluence, so the certificate, not the rule set, is what
    defines the normal form.
    """

    beta: bool = True            # β-reduce applied binders (Eval(LLam …) / NEval(Lam …))
    distribute: bool = True      # push linear/multilinear ops through + (incl. zero-padded ⊕)
    push_contractions: bool = True  # localize Eval/Compose into the ⊗ factor (x(c)·y)
    collect: bool = True         # collect like monomials (v + v → 2·v, cancellation)
    order_sum: bool = True       # sort + summands by term_key — value-safe (commutative)
    order_otimes: bool = False   # canonical ⊗ factor order — up to iso; emits a Permute 𝒞
    order_oplus: bool = False    # canonical ⊕ component order — up to iso; heavier 𝒞
    dissolve_oplus: bool = False  # dissolve ⊕ into a sum of injections (⊕ disappears; + outermost)
    canon_scalars: bool = False  # canonicalize scalar PRODUCTS (commutative ·); partial rational NF
    canon_binders: bool = True   # α-canonicalize bound vars (α-variant terms → identical NF)
    eta_advanced: bool = False   # fold a seed-linear λ into its covector/map (Riesz/adjoint η)


# The default preset: the conservative, value-EXACT end of the ladder — only the commutative
# `+` is reordered, so every rewrite is layout-identical (no `𝒞` correction is ever emitted).
DEFAULT_SOP = SoPConfig()

# The aggressive preset: also canonicalizes ⊗ factor order up-to-iso, inserting a `Permute`
# output correction `𝒞` so the value/layout is still restored exactly.
AGGRESSIVE_SOP = SoPConfig(order_otimes=True)

# The "pretty derivative" preset: collapse seed-linear lambdas (a `diff` result) into the
# lambda-free covector / linear map they denote (Riesz / adjoint η). Value-exact; opt-in.
RIESZ_SOP = SoPConfig(eta_advanced=True)


# ===========================================================================
# Generic term traversal (rebuild + substitution).
# ===========================================================================


def _child_terms(node: Term) -> tuple[Term, ...]:
    """Give the immediate *Term* children of ``node`` (a ``Scale`` scalar may be one)."""
    out: list[Term] = []
    for fld in fields(node):
        val = getattr(node, fld.name)
        for c in (val if isinstance(val, tuple) else (val,)):
            if is_term(c):
                out.append(c)
    return tuple(out)


def _rebuild(node: Term, f: Callable[[Term], Term]) -> Term:
    """Reconstruct ``node`` with ``f`` applied to each *Term* child.

    Spaces, dims, ints, strings, the opaque ``External`` payload and binder sorts pass through
    untouched. Returns ``node`` itself when nothing changed, so identity rewrites don't churn.
    """
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
    """Capture-avoiding substitution of value variables.

    Replace every ``Var`` whose ``uid`` is a key of ``mapping`` by the mapped Term.

    Capture-avoidance is automatic: binder variables carry globally fresh ``uid`` values, so
    an inner binder never shares a ``uid`` with the variable being substituted, and a
    substituted-in term cannot mention a variable bound by a binder the traversal passes
    under.
    """

    def go(node: Term) -> Term:
        if isinstance(node, Var):
            return mapping.get(node.uid, node)
        return _rebuild(node, go)

    return go(term)


def map_spaces_in_term(term: Term, on_space: Callable[[Space], Space]) -> Term:
    """Apply a space rewrite to every space a term embeds, however deeply annotated.

    ``on_space`` is any ``Space -> Space`` rewrite, typically a
    :func:`~grassmann.lower.space_basis.map_space` closure, so one rewrite can be applied to
    a space and to a term's embedded spaces without being written twice. This is the general
    form of :func:`_subst_space_in_term`, which performs the same traversal but is hardwired
    to ``SVar(name) := repl``.

    The traversal has to reach annotations, not just value positions, because a space can
    hide entirely inside one. ``Λᵏ(Img J) = Img(alt)`` keeps its ``Img(J)`` on the binder
    variable ``Var('x', Dual(Img(J)))`` of ``alt``'s ``LLam``: :func:`subst` visits value
    ``Var`` nodes and never sees it, and rewriting the value's space does not reach it
    either. A rewrite that missed it would work on a 1-form, where the immersion is a leaf of
    the value space, and silently do nothing for a form of degree two or more.

    The rewrite is structural and unconditional — it does not know what the caller is
    swapping — and returns ``term`` itself, not a copy, when no embedded space changes.
    """
    return _map_spaces(term, on_space)


def _map_spaces(term: Term, on_space: Callable[[Space], Space]) -> Term:
    """Run the shared traversal.

    Rebuild `term`, applying `on_space` to every Space-typed field and recursing into every Term-typed one.
    Identity-preserving when nothing changes.
    """
    def go(node: Term) -> Term:
        new_fields: dict[str, object] = {}
        changed = False
        for fld in fields(node):
            val = getattr(node, fld.name)
            if isinstance(val, tuple):
                new = tuple(
                    go(c) if is_term(c) else (on_space(c) if is_space(c) else c) for c in val
                )
                if any(a is not b for a, b in zip(new, val, strict=True)):
                    changed = True
                new_fields[fld.name] = new
            elif is_term(val):
                nv: object = go(val)
                if nv is not val:
                    changed = True
                new_fields[fld.name] = nv
            elif is_space(val):
                nv = on_space(val)
                if nv is not val:
                    changed = True
                new_fields[fld.name] = nv
            else:
                new_fields[fld.name] = val
        return replace(node, **new_fields) if changed else node

    return go(term)


def _subst_space_in_term(term: Term, name: str, repl: Space) -> Term:
    """Substitute the rigid space variable ``SVar(name) := repl`` through every space ``term`` embeds.

    Used by space-binder β. Recurses into Term children and applies
    :func:`grassmann.infer.subst.subst_space` to every Space-typed field.
    """
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
    """Substitute the nat parameter ``DimVar(name) := repl`` through every dimension ``term`` embeds.

    Used by nat-binder β. A single generic structural walk over the whole Term/Space/Dim tree —
    every ``DimVar(name)`` (in an ``OPlusRep`` count, an ``App``/``NEval`` nat arg, a
    ``DimSum``/``DimProd``/…) is replaced. Capture under a shadowing inner nat binder is not
    handled — nat binders are rare.
    """
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


def _rename_svar(node: object, old: str, new: str) -> object:
    """Rename the rigid space variable ``SVar(old) → SVar(new)`` through a Term / Space / Dim."""
    from grassmann.core.space import SVar
    from grassmann.infer.subst import subst_space

    repl = SVar(new)
    if is_space(node):
        return subst_space(node, old, repl)  # handles a top-level SVar too
    if is_term(node):
        return _subst_space_in_term(node, old, repl)
    return node  # a Dim carries no SVar


def _rename_dimvar(node: object, old: str, new: str) -> object:
    """Rename the nat variable ``DimVar(old) → DimVar(new)`` through a Term / Space / Dim."""
    from grassmann.core.space import DimVar

    if is_term(node) or is_space(node) or is_dim(node):
        return _subst_nat_in_term(node, old, DimVar(new))  # generic walker, handles top DimVar
    return node


def alpha_canon(term: Term) -> Term:
    """α-canonicalize bound variables so two α-equivalent terms map to the **identical** term.

    The basis of an α-invariant :func:`term_key` and of :func:`is_normalized_sop`'s binder check.

    Covers BOTH binders fully: ``LLam.vars`` and ``Lam.term_vars`` are renamed to ``%k`` (with a
    canonical negative ``uid``); a ``Lam``'s binder **names** are canonicalized too — a value
    (``SpaceSort``) binder's name follows its term var, and a ``SPACE``/``NAT`` binder's name is
    renamed throughout the body's ``SVar``/``DimVar`` references. Free variables (non-negative
    ``uid`` values) are untouched and never collide. Idempotent.
    """
    counter = [0]

    def fresh(v: Var) -> Var:
        counter[0] += 1
        return Var(f"%{counter[0]}", v.space, v.props, uid=-counter[0])

    def go(node: Term, env: dict[int, Var]) -> Term:
        match node:
            case Var():
                return env.get(node.uid, node)
            case LLam():
                env2 = dict(env)
                nv = []
                for v in node.vars:
                    cv = fresh(v)
                    env2[v.uid] = cv
                    nv.append(cv)
                return LLam(tuple(nv), go(node.body, env2))
            case Lam():
                env2 = dict(env)
                new_binders: list[tuple[str, object]] = []
                ntv: list[Var] = []
                tv_iter = iter(node.term_vars)
                body: object = node.body
                for name, sort in node.binders:
                    counter[0] += 1
                    cname = f"%{counter[0]}"
                    new_binders.append((cname, sort))
                    if isinstance(sort, SpaceSort):
                        v = next(tv_iter)
                        env2[v.uid] = Var(cname, v.space, v.props, uid=-counter[0])
                        ntv.append(env2[v.uid])
                    elif sort == SPACE:
                        body = _rename_svar(body, name, cname)
                    elif sort == NAT:
                        body = _rename_dimvar(body, name, cname)
                body = go(body, env2) if is_term(body) else body  # type: ignore[arg-type]
                return Lam(tuple(new_binders), body, tuple(ntv))  # type: ignore[arg-type]
            case _:  # audit: ok EX-WILDCARD — bail-default (§0): generic structural recursion [approved 2026-07-03]
                return _rebuild(node, lambda c: go(c, env))

    return go(term, {})


def _has_binder(t: Term) -> bool:
    """Report whether ``t`` contains a binder (``LLam``/``Lam``/``Frame``) anywhere.

    Lets :func:`term_key` take the α-canonicalization fast path only when it is needed.
    """
    match t:
        case LLam() | Lam() | Frame():
            return True
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): generic structural recursion [approved 2026-07-03]
            return any(_has_binder(c) for c in _child_terms(t))


# ===========================================================================
# Spaces & scalars.
# ===========================================================================


def _space(t: Term) -> Space | None:
    """Synthesize the space of ``t``, or give ``None`` if it does not type-check yet.

    Used to give a fresh ``Zero`` its space. ``Zero(None)`` is a valid placeholder, so a failure degrades
    gracefully rather than aborting the rewrite.
    """
    from grassmann.infer.synth import synth_space

    try:
        return synth_space(t)
    except Exception:
        return None


def normalize_space(s: Space) -> Space:
    """Canonical representative of a space's **isomorphism class**.

    Space normalization *modulo* term normalization:

    - ``⊗`` factors and ``⊕`` components are sorted by canonical key (symmetric-monoidal: two
      spaces are iso iff their normal forms are equal), recursing through ``Dual``/``WithInner``;
    - any **term** a space embeds (``Ker``/``Img``/``CoKer``/``CoImg`` map, ``WithInner`` metric,
      ``App`` args) is normalized with :func:`sum_of_products`, so space equality is decided
      modulo term normalization.

    Two values live in iso spaces iff ``normalize_space`` agrees; the up-to-iso term corrections
    (``Permute`` for ``⊗``, ``TuplePermute`` for ``⊕``) are exactly the witnesses of that iso.
    """
    from grassmann.core.space import (
        App,
        CoImg,
        CoKer,
        Dual,
        Img,
        Ker,
        OPlus,
        OPlusRep,
        OProd,
        SpaceMeta,
        SVar,
        WithInner,
        with_inner,
    )

    match s:
        case OProd():
            return OProd(tuple(sorted((normalize_space(f) for f in s.spaces), key=repr)))
        case OPlus():
            return OPlus(tuple(sorted((normalize_space(c) for c in s.spaces), key=repr)))
        case Dual():
            return Dual(normalize_space(s.space))
        case WithInner():
            return with_inner(normalize_space(s.space), sum_of_products(s.metric))
        case Ker() | Img() | CoKer() | CoImg():
            src = normalize_space(s.source) if s.source is not None else None
            tgt = normalize_space(s.target) if s.target is not None else None
            return type(s)(sum_of_products(s.f), src, tgt)
        case App():
            args = tuple(
                normalize_space(a) if is_space(a) else (sum_of_products(a) if is_term(a) else a)
                for a in s.args
            )
            return App(sum_of_products(s.g), s.positions, args)
        case Scalar() | SVar() | OPlusRep() | SpaceMeta():
            return s  # Scalar / SVar / OPlusRep (base) / SpaceMeta — atomic
        case _ as unreachable:
            assert_never(unreachable)


def space_key(s: Space) -> str:
    """Give a total, iso-invariant ordering key on Spaces.

    ``repr`` of the :func:`normalize_space` form (so iso spaces share a key).
    """
    return repr(normalize_space(s))


def spaces_iso(a: Space, b: Space) -> bool:
    """Report whether two spaces are isomorphic.

    Equal up to ``⊗``/``⊕`` reordering and term normalization — i.e.
    ``normalize_space(a) == normalize_space(b)``.
    """
    return normalize_space(a) == normalize_space(b)


def _static_space_dim(s: Space) -> int | None:
    """Give the static (concrete) dimension of a space, or ``None`` if it is symbolic."""
    from grassmann.core.space import (
        App,
        CoImg,
        CoKer,
        DimLit,
        Dual,
        Img,
        Ker,
        OPlus,
        OPlusRep,
        OProd,
        Scalar,
        SpaceMeta,
        SVar,
        WithInner,
    )

    match s:
        case Scalar():
            return 1
        case Dual() | WithInner():
            return _static_space_dim(s.space)
        case OPlusRep():
            inner = _static_space_dim(s.space)
            return s.count.value * inner if isinstance(s.count, DimLit) and inner is not None else None
        case OProd():
            prod = 1
            for f in s.spaces:
                d = _static_space_dim(f)
                if d is None:
                    return None
                prod *= d
            return prod
        case OPlus():
            total = 0
            for c in s.spaces:
                d = _static_space_dim(c)
                if d is None:
                    return None
                total += d
            return total
        case SVar() | Ker() | Img() | CoKer() | CoImg() | App() | SpaceMeta():
            return None  # symbolic — no static (concrete) dimension
        case _ as unreachable:
            assert_never(unreachable)


def _is_num(x: object) -> bool:
    return isinstance(x, (int, float))


def _mul_scalar(a: float | Term, b: float | Term) -> float | Term:
    """Scalar·scalar, collapsing numeric factors and the 0/1 identities.

    A genuinely symbolic product of two scalar *Terms* is represented as ``Scale(a, b)`` (both are
    ``Scalar``-valued, so this is the scalar product).
    """
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
    """Peel a leading run of *numeric* ``Scale`` nodes off ``v`` → ``(coeff, atom)``.

    A symbolic scalar stops the peel (it stays part of the atom).
    """
    c = 1.0
    while isinstance(v, Scale) and _is_num(v.scalar):
        c *= float(v.scalar)
        v = v.vector
    return c, v


# ===========================================================================
# β-reduction rules.
# ===========================================================================


def rule_beta_eval(node: Term) -> Term | None:
    """β-reduce a linear binder applied to arguments — the ``Eval(LLam …)`` redex.

    Slot ``p < n`` is the dual input of linear var ``xₚ``; applying ``args[k]`` there
    substitutes ``xₚ := args[k]``. Full application returns the substituted body; a partial
    one returns a residual ``LLam`` over the still-unapplied vars. Positions into the
    *codomain* (``p ≥ n``) are not var redexes and are left alone.
    """
    match node:
        case Eval(f=LLam(vars=vars_, body=body)):
            n = len(vars_)
            if any(p >= n for p in node.positions):
                return None
            mapping = {vars_[p].uid: a for p, a in zip(node.positions, node.args, strict=True)}
            applied = set(node.positions)
            new_body = subst(body, mapping)
            remaining = tuple(vars_[i] for i in range(n) if i not in applied)
            if not remaining:
                return new_body
            return LLam(remaining, new_body)
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def _term_var_for_binder(lam: Lam, idx: int) -> Var | None:
    """Give the ``Var`` introduced by the ``idx``-th binder, if it is a value (``SpaceSort``) binder.

    ``term_vars`` lists those vars in binder order, skipping space/nat sorts.
    """
    j = 0
    for i, (_name, sort) in enumerate(lam.binders):
        if isinstance(sort, SpaceSort):
            if i == idx:
                return lam.term_vars[j] if j < len(lam.term_vars) else None
            j += 1
    return None


def rule_beta_neval(node: Term) -> Term | None:
    """β-reduce a non-linear binder applied to arguments — the ``NEval(Lam …)`` redex.

    Each supplied slot substitutes into the body: a **value** binder (``SpaceSort``)
    replaces its term var by the argument Term; a **space** binder replaces ``SVar(name)``
    by the argument Space throughout the body's spaces; a **nat** binder replaces
    ``DimVar(name)`` by the argument ``Dim`` throughout the body's dimensions. Unapplied
    binders survive as a residual ``Lam`` (partial application).
    """
    match node:
        case NEval(f=Lam() as lam):
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
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


# ===========================================================================
# Algebraic simplification rules (always strictly reducing).
# ===========================================================================


def rule_scale(node: Term) -> Term | None:
    """Scalar laws: ``0·v → 0``, ``1·v → v``, ``r·0 → 0``, ``r·(s·w) → (r·s)·w``."""
    match node:
        case Scale(scalar=r, vector=v) if _is_num(r) and r == 0:
            return Zero(_space(v))
        case Scale(scalar=r, vector=v) if _is_num(r) and r == 1:
            return v
        case Scale(vector=Zero() as v):
            return v
        case Scale(scalar=r, vector=Scale() as v):
            # Collapse a nested scale into one coefficient — but a *symbolic scalar product*
            # (inner vector also Scalar-valued) is owned by rule_scalar_product, so defer it to
            # avoid the two rules fighting over where the numeric coefficient nests.
            if (_is_num(r) and _is_num(v.scalar)) or not isinstance(_space(v.vector), Scalar):
                return Scale(_mul_scalar(r, v.scalar), v.vector)
            return None
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_sum(node: Term) -> Term | None:
    """Flatten nested ``Sum`` nodes and drop ``Zero`` summands.

    Collapse to the single summand (or a ``Zero``) when that is all that remains.
    """
    match node:
        case Sum(vectors=vectors):
            flat: list[Term] = []
            changed = False
            for v in vectors:
                match v:
                    case Sum():
                        flat.extend(v.vectors)
                        changed = True
                    case Zero():
                        changed = True
                    case _:  # audit: ok EX-WILDCARD — bail-default (§0): any other summand is kept as-is [approved 2026-07-03]
                        flat.append(v)
            if not flat:
                return Zero(_space(vectors[0]))
            if len(flat) == 1:
                return flat[0]
            return Sum(tuple(flat)) if changed else None
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def _gather_scalar_product(t: Term, num: float, syms: list[Term]) -> float:
    """Flatten a product of scalars into a numeric coefficient and symbolic factors.

    A product of scalars is a nested ``Scale`` whose operands are all ``Scalar``-valued; it folds
    into a numeric coefficient ``num`` and a list of symbolic scalar-atom factors ``syms``.
    """
    match t:
        case Lit():
            return num * t.value
        case Scale() if isinstance(_space(t.vector), Scalar):
            if _is_num(t.scalar):
                num *= t.scalar
            else:
                num = _gather_scalar_product(t.scalar, num, syms)  # type: ignore[arg-type]
            return _gather_scalar_product(t.vector, num, syms)
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): any other Term is an atomic scalar factor [approved 2026-07-03]
            syms.append(t)  # an atomic scalar factor (a dot-product Eval, External scalar, scalar Var…)
            return num


def rule_scalar_product(node: Term) -> Term | None:
    """Canonicalize a **product of scalars** (the commutative multiplicative monoid).

    Flatten nested ``Scale`` nodes whose operands are all ``Scalar``-valued, fold the numeric factor, and sort the
    symbolic factors by ``term_key``. So ``⟨c,x⟩·⟨d,y⟩`` and ``⟨d,y⟩·⟨c,x⟩`` reach the same form. Opt-in via
    ``canon_scalars``.

    This is the multiplicative half of a scalar normal form. Sums of scalar products and
    ``div``/``reciprocal`` with cancellation — the full rational ``P/Q`` — are not
    normalized, so a folded scalar can print un-simplified while remaining value-correct.
    """
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
    """Collect like monomials in a ``Sum``.

    Group summands by atom (after peeling a numeric scalar) and add their coefficients. ``v + v → 2·v``; ``2·v
    + (−2)·v → 0``.
    """
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
    """Drop an identity map applied to a single vector: ``Identity(s)·v → v``."""
    match node:
        case Eval(f=Identity(), positions=(0,), args=(arg,)):
            return arg
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def _mentions_vars(node: Term, uids: set[int]) -> bool:
    """Report whether ``node`` references any ``Var`` whose ``uid`` is in ``uids``."""
    if isinstance(node, Var):
        return node.uid in uids
    return any(_mentions_vars(c, uids) for c in _child_terms(node))


def rule_eta(node: Term) -> Term | None:
    """η-reduce ``λ⊸(x₀,…,xₙ₋₁). f(x₀,…,xₙ₋₁) → f`` — the inverse of β.

    Fires when ``f`` does not mention the bound vars and applies them at slots ``0..n-1`` in
    order. **Sound** because an ``LLam`` lowers identically to the stored map it expands (the
    FFS re-orientation). Collapses an η-expanded map back to the map itself.
    """
    match node:
        case LLam(vars=vars_, body=Eval() as body):
            n = len(vars_)
            if not (body.positions == tuple(range(n)) and tuple(body.args) == vars_):
                return None
            if _mentions_vars(body.f, {v.uid for v in vars_}):
                return None  # f must be closed w.r.t. the bound vars
            return body.f
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def _mentions(t: object, uid: int) -> bool:
    """``t`` (a Term, or a numeric scalar) mentions the variable ``uid``.

    A non-Term scalar (a ``Scale`` coefficient may be a bare ``float``) never mentions a variable.
    """
    return is_term(t) and _mentions_vars(t, {uid})


def _peel_inner(arm: Term, r: Term, dvar: Var) -> Term | None:
    """Peel the covector core of advanced-η out of a δ-bearing inner-product arm.

    ``arm`` (the inner-product slot that contains the seed ``dvar``) and ``r`` (the δ-closed *Riesz arm*).
    Return the lambda-free covector ``M`` with ``⟨δ, evalₘ⟩``-style application equal to ``λδ. ⟨arm, r⟩`` —
    peeling each linear map off the δ-slot moves its adjoint onto ``r`` (``⟨A(δ),r⟩ = ⟨δ, A†(r)⟩``). Metric
    symmetry makes the bare ``⟨δ, r⟩ → r♭`` arm Dagger-free.
    """
    uid = dvar.uid
    match arm:
        case Var(uid=u) if u == uid:
            return Eval(Flat(dvar.space), (0,), (r,))  # ⟨δ, r⟩ → r♭
        case Scale() if not _mentions(arm.scalar, uid):
            return _peel_inner(arm.vector, Scale(arm.scalar, r), dvar)  # ⟨c·v, r⟩ = ⟨v, c·r⟩
        case Eval(positions=(0,), args=(arg0,)) if not _mentions(arm.f, uid):
            # ⟨A(δ), r⟩ = ⟨δ, A†(r)⟩ — peel the map, accumulate the adjoint onto the Riesz arm.
            return _peel_inner(arg0, Eval(Dagger(arm.f), (0,), (r,)), dvar)
        case External() if _is_hadamard(arm):
            # ⟨c ⊙ δ, r⟩ = ⟨δ, c ⊙ r⟩ — the elementwise (diagonal) map is self-adjoint, so the
            # δ-free factor moves onto the Riesz arm unchanged (matrix-calc elementwise chain rule).
            c, v = _hadamard_split(arm, uid)
            if v is not None:
                return _peel_inner(v, External(arm.space, arm.data, (c, r)), dvar)
            return None
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def _is_hadamard(t: object) -> bool:
    return (
        isinstance(t, External)
        and getattr(t.data, "name", None) == "hadamard"
        and len(t.args) == 2
    )


def _hadamard_split(t: External, uid: int) -> tuple[Term, Term | None]:
    """Split a ``hadamard(a, b)`` into ``(δ-free factor, δ-bearing factor)``.

    Exactly one arm may mention δ (else ``(_, None)``: the elementwise product is nonlinear / not our shape).
    """
    a, b = t.args
    if _mentions(a, uid) and not _mentions(b, uid):
        return b, a
    if _mentions(b, uid) and not _mentions(a, uid):
        return a, b
    return a, None


def _flatten_compose(t: Term) -> list[Term]:
    """Flatten a binary ``Compose(f, (0,), (g,))`` chain into its factor list.

    Descends BOTH sides (composition is associative, so left- and right-nested spellings both
    flatten): ``A∘B∘C`` → ``[A, B, C]``. A non-Compose, or a multi-slot ``Compose``, is a
    single factor.
    """
    if isinstance(t, Compose) and t.positions == (0,) and len(t.gs) == 1:
        return [*_flatten_compose(t.f), *_flatten_compose(t.gs[0])]
    return [t]


def _compose_chain(factors: list[Term], fallback_space: Space) -> Term:
    """Rebuild a left-nested binary ``Compose`` chain from a factor list (``[]`` → ``Identity``)."""
    if not factors:
        return Identity(fallback_space)
    m = factors[0]
    for g in factors[1:]:
        m = Compose(m, (0,), (g,))
    return m


def _factor_trace(node: Trace, dvar: Var) -> Term | None:
    """Fold a matrix-trace monomial ``tr(L ∘ δ ∘ R)`` into its Frobenius covector.

    δ is a square map, appearing once as a bare ``δ`` or ``δ†`` factor. Cyclic invariance
    rotates δ to the end: ``tr(L∘δ∘R) = tr((R∘L)∘δ) = ⟨δ, (R∘L)†⟩``  (and
    ``tr((R∘L)∘δ†) = ⟨δ, R∘L⟩``). This turns the matrix-calculus trace derivatives into the
    inner-product form the covector peel already handles. Bails (``None``) unless exactly one
    factor is a *bare* δ / δ† occurrence.
    """
    uid = dvar.uid
    factors = _flatten_compose(node.f)
    idxs = [i for i, fac in enumerate(factors) if _mentions(fac, uid)]
    if len(idxs) != 1:
        return None
    k = idxs[0]
    fac = factors[k]
    match fac:
        case Var(uid=u) if u == uid:
            daggered = False
        case Dagger(f=Var(uid=u)) if u == uid:
            daggered = True
        case _:  # audit: ok EX-WILDCARD — rule probe: δ buried inside a non-bare factor — not a clean trace pairing [approved 2026-07-03]
            return None
    rest = factors[k + 1 :] + factors[:k]  # cyclic rotation: the factor after δ moves to front
    rmap = _compose_chain(rest, dvar.space)
    riesz = rmap if daggered else Dagger(rmap)  # ⟨δ, R⟩ for δ†, else ⟨δ, R†⟩
    return _peel_inner(dvar, riesz, dvar)


def _factor_inner(s: Space, a: Term, b: Term, dvar: Var) -> Term | None:
    """Fold ``λδ. ⟨a, b⟩``, a *scalar* inner-product monomial, into its covector.

    Exactly one arm may mention δ (both ⇒ the nonlinear ``⟨δ,δ⟩``, rejected by the linearity
    gate before we get here; neither ⇒ not a δ-functional). The metric is symmetric, so the
    choice of which arm is the input vs the Riesz arm is value-free.
    """
    uid = dvar.uid
    a_has, b_has = _mentions(a, uid), _mentions(b, uid)
    if a_has and not b_has:
        return _peel_inner(a, b, dvar)
    if b_has and not a_has:
        return _peel_inner(b, a, dvar)
    return None


def _factor_llam_body(node: Term, dvar: Var) -> Term | None:
    """Find a lambda-free ``M`` with ``Eval(M, (0,), (dvar,)) ≡ node``.

    I.e. the linear map / covector that the seed-linear body ``node`` denotes. Returns ``None`` for any shape
    it cannot express lambda-free (strictly conservative — the ``LLam`` is then left intact).
    """
    uid = dvar.uid
    match node:
        # λδ. δ  →  Id  (the identity Jacobian; the vector-regime base case, generalizing rule_eta).
        case Var(uid=u) if u == uid:
            return Identity(dvar.space)
        # λδ. c·v  →  c · M_v   (the coefficient c must be δ-closed).
        case Scale() if not _mentions(node.scalar, uid):
            m = _factor_llam_body(node.vector, dvar)
            return Scale(node.scalar, m) if m is not None else None
        # λδ. Σ vᵢ  →  Σ Mᵢ   (covectors / maps add).
        case Sum():
            ms = [_factor_llam_body(v, dvar) for v in node.vectors]
            if any(m is None for m in ms):
                return None
            return Sum(tuple(ms))
        # Scalar regime: an inner product ⟨a, b⟩ = Eval(Inner(s), (0,1), (a,b)).
        case Eval(f=Inner(space=s), positions=(0, 1), args=(a, b)):
            return _factor_inner(s, a, b, dvar)
        # Scalar regime: a matrix-trace pairing tr(L ∘ δ ∘ R) — the Frobenius covector.
        case Trace():
            return _factor_trace(node, dvar)
        # Vector regime: a δ-closed map applied to δ-stuff  →  A ∘ (δ ↦ arg).
        case Eval(positions=(0,), args=(arg0,)) if not _mentions(node.f, uid):
            m = _factor_llam_body(arg0, dvar)
            return Compose(node.f, (0,), (m,)) if m is not None else None
        case _:  # audit: ok EX-WILDCARD — rule probe: None = shape not expressible lambda-free [approved 2026-07-03]
            return None


def rule_eta_advanced(node: Term) -> Term | None:
    """**Advanced η-reduction** (covector / linear-map folding) — make a derivative look like the map it *is*.

    Collapse a single-seed linear lambda ``λδ. body``, whose body is a covector or
    linear-map expression in δ, into the lambda-free term it denotes:

        ``λδ. ⟨δ, x⟩ + ⟨x, δ⟩ → 2·x♭``,   ``λδ. ⟨A(δ), w⟩ → (A†w)♭``,   ``λδ. A(δ) → A``.

    The body must be **exactly-once linear** in δ (the resource gate rejects the nonlinear
    ``⟨δ,δ⟩`` and the δ-unused body). Sound on this branch because an ``LLam`` lowers identically
    to the stored map it denotes (the FFS re-orientation): the rewrite is an equality
    of the coordinate-free algebra, verified value-identical by the suite. Opt-in
    (``SoPConfig.eta_advanced``); off in ``simplify``/``DEFAULT_SOP``.
    """
    match node:
        case LLam(vars=(dvar,), body=body):
            # Defer the bare η shape λδ.f(δ) to rule_eta (it produces f directly; avoids f∘Id churn).
            if (
                isinstance(body, Eval)
                and body.positions == (0,)
                and tuple(body.args) == node.vars
                and not _mentions_vars(body.f, {dvar.uid})
            ):
                return None
            from grassmann.infer.linearity import LinearityError, check_llam_resources

            try:
                check_llam_resources(node)  # exactly-once gate: rejects ⟨δ,δ⟩ and δ-unused bodies
            except LinearityError:
                return None
            m = _factor_llam_body(body, dvar)
            if m is None or _mentions_vars(m, {dvar.uid}):
                return None  # not expressible lambda-free, or δ leaked — leave the LLam intact
            return m
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_eval_compose_fold(node: Term) -> Term | None:
    """Fold a nested application into a composition — application associativity.

        ``Eval(f, (0,), (Eval(h,(0,),(x,)),)) → Eval(Compose(f,(0,),(h,)), (0,), (x,))``.

    Unambiguous under the normalized convention. Grouping the map algebra lets the map-level
    identities fire on the ``Compose`` — e.g. ``Eval(Solve(g), Eval(g, x))`` folds to
    ``Eval(Solve(g)∘g, x) → Eval(Id, x) → x``. Direction: compose maps before applying
    (value-exact).
    """
    if not (isinstance(node, Eval) and node.positions == (0,) and len(node.args) == 1):
        return None
    inner = node.args[0]
    if isinstance(inner, Eval) and inner.positions == (0,) and len(inner.args) == 1:
        return Eval(Compose(node.f, (0,), (inner.f,)), (0,), inner.args)
    return None


def rule_eval_zero(node: Term) -> Term | None:
    """Reduce ``0·… → 0`` and ``f(…,0,…) → 0``.

    A linear application with a zero map or a zero argument vanishes.
    """
    match node:
        case Eval() if isinstance(node.f, Zero) or any(isinstance(a, Zero) for a in node.args):
            return Zero(_space(node))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_compose(node: Term) -> Term | None:
    """Identity / zero laws for composition.

    Drop ``Identity`` operands, ``Identity ∘ g → g``, and ``0 ∘ … → 0`` / ``… ∘ 0 → 0``.
    """
    match node:
        case Compose(f=Identity(), positions=(0,), gs=(g,)):
            return g
        case Compose(f=Zero()):
            return Zero(_space(node))
        case Compose() if any(isinstance(g, Zero) for g in node.gs):
            return Zero(_space(node))
        case Compose(f=f, positions=positions, gs=gs):
            keep_pos: list[int] = []
            keep_gs: list[Term] = []
            dropped = False
            for p, g in zip(positions, gs, strict=True):
                match g:
                    case Identity():
                        dropped = True
                    case _:  # audit: ok EX-WILDCARD — bail-default (§0): a non-Identity operand is kept [approved 2026-07-03]
                        keep_pos.append(p)
                        keep_gs.append(g)
            if not dropped:
                return None
            if not keep_gs:
                return f
            return Compose(f, tuple(keep_pos), tuple(keep_gs))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_involution(node: Term) -> Term | None:
    """Cancel a double adjoint or double inverse — ``Dagger`` and ``Solve`` are involutive: ``(fᵀ)ᵀ → f`` and ``(f⁻¹)⁻¹ → f``."""
    match node:
        case Dagger(f=Dagger(f=inner)):
            return inner
        case Solve(f=Solve(f=inner)):
            return inner
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_dagger_push(node: Term) -> Term | None:
    """Push the metric-transpose adjoint ``Dagger`` toward the leaves.

    It then only ever wraps a primitive map (the grammar's ``atom``): ``Id† → Id``,
    ``(f⊗g)† → f†⊗g†`` (``Kron``), ``(f∘g)† → g†∘f†`` (``Compose``, single slot). Value-exact
    (verified). ``(f†)† → f`` is handled by :func:`rule_involution`; ``Dagger`` over
    ``+``/``Scale`` by :func:`dist_unary`.

    Not done for ``Solve``/``Pinv``: ``Solve``'s argument must be a ``bij``-asserted *var* (it
    can never be a ``Compose``/``Kron``), and ``Pinv`` does **not** distribute over composition
    (the pseudo-inverse of a product is not the product of pseudo-inverses).
    """
    match node:
        case Dagger(f=Identity() as f):
            return f
        case Dagger(f=Kron(f=a, g=b)):
            return Kron(Dagger(a), Dagger(b))
        case Dagger(f=Compose(f=h, positions=(0,), gs=(g,))):
            return Compose(Dagger(g), (0,), (Dagger(h),))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_musical(node: Term) -> Term | None:
    """Canonicalize the musical isomorphisms to a single metric primitive ``g = Inner(s)``.

    ``♭ = Flat(s) → Inner(s)`` (same space ``s*⊗s*`` and same data), ``♯ = Sharp(s) → Pinv(Inner(s))`` (the
    inverse metric), and cancel an inverse-metric pair in a composition — ``g⁻¹∘g → Id(s)``, ``g∘g⁻¹ →
    Id(s*)`` (a metric is non-degenerate, so ``Pinv(Inner)`` is a true inverse). Direction: collapse musical
    maps toward ``Inner``/``Identity`` (value-exact).
    """
    match node:
        case Flat():
            return Inner(node.space)
        case Sharp():
            return Pinv(Inner(node.space))
        case Compose(f=Pinv(f=Inner(space=si)), positions=(0,), gs=(Inner(space=sg),)) if si == sg:
            return Identity(sg)
        case Compose(f=Inner(space=sf), positions=(0,), gs=(Pinv(f=Inner(space=sg)),)) if sf == sg:
            return Identity(dual(sf))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_inverse_compose(node: Term) -> Term | None:
    """Cancel an inverse/forward map pair in a composition toward ``Identity``.

    The map-algebra analogue of the musical ``g⁻¹∘g→Id``:

    - ``Solve(g)∘g → Id(dom)`` and ``g∘Solve(g) → Id(cod)``    (``g`` asserted ``bij``);
    - ``Pinv(f)∘f → Id(dom)``  (``f`` asserted ``inj`` — left inverse) and
      ``f∘Pinv(f) → Id(cod)``  (``f`` asserted ``surj`` — right inverse).

    The property guards read the operand ``Var``'s asserted ``MapProps``; a pseudo-inverse only
    cancels in the direction its rank condition guarantees (verified). The ``Eval``-applies-
    transpose convention does **not** intrude here — these are stored-matrix map identities.
    Direction: toward ``Identity``.
    """
    from grassmann.infer.properties import is_bijective, is_injective, is_surjective
    from grassmann.infer.synth import map_domain_codomain

    match node:
        case Compose(f=Solve(f=inner), positions=(0,), gs=(g,)) if inner == g and is_bijective(g):
            return Identity(map_domain_codomain(g)[0])
        case Compose(f=f, positions=(0,), gs=(Solve(f=inner),)) if inner == f and is_bijective(f):
            return Identity(map_domain_codomain(f)[1])
        case Compose(f=Pinv(f=inner), positions=(0,), gs=(g,)) if inner == g and is_injective(g):
            return Identity(map_domain_codomain(g)[0])
        case Compose(f=f, positions=(0,), gs=(Pinv(f=inner),)) if inner == f and is_surjective(f):
            return Identity(map_domain_codomain(f)[1])
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_project_embed(node: Term) -> Term | None:
    """Cancel an embed-then-project round-trip onto a value's own subspace: ``Project(Embed(v, S), s) → v``.

    Embedding into a larger space and projecting straight back onto ``v``'s own subspace is
    the identity on that subspace. Space normalization decides whether the projection target
    really is ``v``'s space up to isomorphism (``spaces_iso``); the round-trip is cancelled
    only then. Value-exact, and it moves toward the bare value.
    """
    match node:
        case Project(vector=Embed(vector=v)):
            sv = _space(v)
            if sv is None:
                return None
            target = node.space if node.space is not None else sv  # omitted target ⇒ inferred = sv
            if spaces_iso(target, sv):
                return v
            return None
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_kron(node: Term) -> Term | None:
    """``Kron`` (map ⊗) algebra.

    The identity is multiplicative — ``Id(a)⊗Id(b) → Id(a⊗b)`` — and ``Kron`` is a **bifunctor** —
    ``(f⊗g)∘(h⊗k) → (f∘h)⊗(g∘k)``. Value-exact (verified); collapses toward fewer ``Kron``/``Compose`` nodes
    (and toward ``Identity``).
    """
    from grassmann.core.space import oprod

    match node:
        case Kron(f=Identity(space=sa), g=Identity(space=sb)):
            return Identity(oprod(sa, sb))
        case Compose(f=Kron(f=a, g=b), positions=(0,), gs=(Kron(f=c, g=d),)):
            return Kron(Compose(a, (0,), (c,)), Compose(b, (0,), (d,)))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_trace_identity(node: Term) -> Term | None:
    """Reduce the full trace of an identity map to its dimension — ``tr(Id(s)) → |s|`` — when ``s`` is statically sized.

    The full trace of the 2-factor identity map.
    """
    match node:
        case Trace(f=Identity(space=s), i=i, j=j) if {i, j} == {0, 1}:
            d = _static_space_dim(s)
            return Lit(float(d)) if d is not None else None
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def _map_arity(t: Term) -> int | None:
    """Give the number of top tensor factors of ``t``'s space (1 for a non-⊗), or ``None`` if untypable."""
    from grassmann.core.space import OProd

    sp = _space(t)
    if sp is None:
        return None
    return len(sp.spaces) if isinstance(sp, OProd) else 1


def rule_trace_kron(node: Term) -> Term | None:
    """Factor the full trace of a Kronecker product into a product of traces: ``tr(f⊗g) → tr(f)·tr(g)``.

    The full trace of a Kronecker product factors into the product of the full traces. Fires only on the full
    trace (``{i,j} == {0,1}``) of a ``Kron`` of two simple (2-factor) maps; value-exact.
    """
    match node:
        case Trace(f=Kron(f=f, g=g), i=i, j=j) if {i, j} == {0, 1}:
            if _map_arity(f) != 2 or _map_arity(g) != 2:
                return None
            return Scale(Trace(f, 1, 0), Trace(g, 1, 0))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_invlift_noop(node: Term) -> Term | None:
    """Drop a single-factor in-place fusion — a value-exact no-op: ``InvLift(v, (i,), k) → v``.

    (``Lift`` of a single factor is *not* a no-op — it re-wraps the dual — so it is left alone.)
    """
    if isinstance(node, InvLift) and len(node.fused) == 1 and node.k in (None, node.fused[0]):
        return node.vector
    return None


def rule_permute(node: Term) -> Term | None:
    """Collapse reshape ``Permute`` corrections.

    An identity permutation vanishes, and two nested permutations compose. Value-exact (a permutation of
    tensor axes). Keeps the up-to-iso ``𝒞`` corrections from stacking up into non-canonical
    ``Permute(Permute(…))`` towers.
    """
    if not isinstance(node, Permute):
        return None
    if list(node.perm) == list(range(len(node.perm))):
        return node.vector
    if isinstance(node.vector, Permute):
        p, q = node.vector.perm, node.perm  # Permute(Permute(v, p), q)_i = v[p[q[i]]]
        return Permute(node.vector.vector, tuple(p[i] for i in q))
    return None


def rule_tuplepermute(node: Term) -> Term | None:
    """Collapse ``⊕``-permutation corrections.

    An identity permutation vanishes and two nested ``TuplePermute`` nodes compose (the ``⊕`` analogue of
    :func:`rule_permute`).
    """
    if not isinstance(node, TuplePermute):
        return None
    if list(node.perm) == list(range(len(node.perm))):
        return node.vector
    if isinstance(node.vector, TuplePermute):
        p, q = node.vector.perm, node.perm
        return TuplePermute(node.vector.vector, tuple(p[i] for i in q))
    return None


def rule_tuple_roundtrip(node: Term) -> Term | None:
    """Fold the ``⊕`` injection/projection round-trip ``π_i ∘ ι_j``.

    An **Iverson bracket** over the block indices times the identity, ``[i=j]·I`` (the bracket is purely
    index/dim data): ``Compose(TupleProject(s,i), TupleEmbed(s,j))`` is ``Identity(block_i)`` when ``i=j`` and
    the ``Zero`` map ``block_j ⊸ block_i`` otherwise. (Indices are concrete here, so the bracket folds
    immediately; symbolic-index spaces would keep it as ``δ[i=j]``.)
    """
    match node:
        case Compose(
            f=TupleProject(space=sf, i=i), positions=(0,), gs=(TupleEmbed(space=sg, i=j),)
        ) if sf == sg:
            comps = sf.spaces
            if i == j:
                return Identity(comps[i])
            return Zero(linear_map_space(comps[j], comps[i]))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


# ===========================================================================
# Linearity / multilinearity distribution (toward sum-of-products).
# ===========================================================================


def _push_sum_then_scale(node: Term, slots: list[tuple[Term, Callable[[Term], Term]]]) -> Term | None:
    """Distribute over a node that is **multilinear** in the given ``slots``.

    Each ``slot`` is a child plus a rebuild-with-new-child closure.
    Pushes a ``Sum`` outward (preferred — moves ``+`` toward the root) before pulling a
    scalar outward (``Scale``). Doing one slot per call lets the engine iterate to a
    fixpoint; trying sums across all slots first keeps the result a clean SoP.
    """
    for child, rebuild in slots:
        if isinstance(child, Sum):  # audit: ok EX-ISINSTANCE-LADDER — rule probe / two-phase slot scan (all Sums before any Scale) resists a single match [approved 2026-07-03]
            return Sum(tuple(rebuild(c) for c in child.vectors))
    for child, rebuild in slots:
        if isinstance(child, Scale):  # audit: ok EX-ISINSTANCE-LADDER — rule probe / two-phase slot scan (all Sums before any Scale) resists a single match [approved 2026-07-03]
            return Scale(child.scalar, rebuild(child.vector))
    return None


def dist_scale(node: Term) -> Term | None:
    """Distribute a scaling over a sum in either slot — ``Scale`` is bilinear: ``r·(a+b) → r·a + r·b`` and ``(p+q)·v → p·v + q·v``."""
    match node:
        case Scale(scalar=r, vector=Sum() as v):
            return Sum(tuple(Scale(r, x) for x in v.vectors))
        case Scale(scalar=Sum() as r, vector=v):
            return Sum(tuple(Scale(p, v) for p in r.vectors))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
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
    """``⊗`` is multilinear.

    Distribute over a ``+`` in any factor, pull a scalar from any factor, and vanish if any factor is ``0``.
    """
    match node:
        case OProdV(vectors=vectors):
            if any(isinstance(v, Zero) for v in vectors):
                return Zero(_space(node))
            slots: list[tuple[Term, Callable[[Term], Term]]] = []
            for k in range(len(vectors)):
                def rb(x: Term, k: int = k) -> Term:
                    new = list(vectors)
                    new[k] = x
                    return OProdV(tuple(new))

                slots.append((vectors[k], rb))
            return _push_sum_then_scale(node, slots)
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def dist_oplus(node: Term) -> Term | None:
    """Pull a ``+`` out of one ``⊕`` (``OPlusV``) slot by **zero-padding**.

    A ``⊕`` is a stack rather than an additive combination of its components, so the other
    slots must be zeroed in the extra copies rather than duplicated:
    ``(a+b)⊕c → a⊕c + b⊕0``. Copying the siblings instead, as ``a⊕c + b⊕c``, would double
    ``c``.

    The first addend keeps the original siblings; each further addend gets a copy with that
    slot's value and ``Zero`` everywhere else. Keeps ``⊕`` as a layer whose components are
    ``+``-free (the ``+ > ⊕ > ⊗`` grammar).
    """
    match node:
        case OPlusV(vectors=vectors):
            for k, v in enumerate(vectors):
                if isinstance(v, Sum):
                    parts: list[Term] = []
                    first = list(vectors)
                    first[k] = v.vectors[0]
                    parts.append(OPlusV(tuple(first)))
                    for addend in v.vectors[1:]:
                        slot = [Zero(_space(s)) for s in vectors]
                        slot[k] = addend
                        parts.append(OPlusV(tuple(slot)))
                    return Sum(tuple(parts))
            return None
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def dist_kron(node: Term) -> Term | None:
    """``Kron`` (map ⊗) is bilinear in its two maps; vanishes if either is ``0``."""
    match node:
        case Kron(f=Zero()) | Kron(g=Zero()):
            return Zero(_space(node))
        case Kron():
            slots = [
                (node.f, lambda f: Kron(f, node.g)),
                (node.g, lambda g: Kron(node.f, g)),
            ]
            return _push_sum_then_scale(node, slots)
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


# ---------------------------------------------------------------------------
# Contraction-push: localize Eval/Compose into the ⊗ factor it acts on.
#
# An `Eval`/`Compose` on `OProdV(g₀,…,gₘ)` indexes ONE slot per value-factor (slot i ↔ gᵢ
# as a whole). For the term to type-check, a contracted slot must be a single Dual factor —
# so an Eval'd factor is always fully consumed to a SCALAR (the `x(c)·y` rule), and a
# Compose'd factor stays a covector. No flattened-arity bookkeeping is needed.
# ---------------------------------------------------------------------------


def dist_eval_oprod(node: Term) -> Term | None:
    """Push a contraction into a tensor product.

    ``Eval(⊗(g₀,…,gₘ), pos, args)`` contracts each selected factor fully (it is a single dual ⇒ a scalar
    ``gᵢ(argᵢ)``) and pulls it out as a coefficient — the ``Eval(x⊗y, c) → x(c)·y`` rule. Bails if a
    contracted factor does not reduce to a scalar (an atypical / untypable slot).
    """
    match node:
        case Eval(f=OProdV(vectors=gs)):
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
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def dist_trace_oprod(node: Term) -> Term | None:
    """Push a ``Trace`` into a tensor product.

    In ``⊗(g₀,…,gₘ)`` each ``Trace`` index is a whole factor, so contracting primal slot ``i`` with dual slot
    ``j`` is a contraction *between* two atoms — represented as ``Eval(gⱼ, gᵢ)`` (the user's choice) —
    yielding a scalar coefficient times the remaining product: ``Trace(x⊗c⊗z, 0, 1) → ⟨c,x⟩·z``.
    """
    match node:
        case Trace(f=OProdV(vectors=gs), i=i, j=j):
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
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def dist_compose_oprod(node: Term) -> Term | None:
    """Push a composition into a tensor product.

    ``Compose(⊗(g₀,…,gₘ), pos, hs)`` precomposes each selected factor (a single dual input slot) with its map
    ``hₖ``, leaving the other factors untouched.
    """
    match node:
        case Compose(f=OProdV(vectors=gs), positions=positions, gs=hs):
            composed = dict(zip(positions, hs, strict=True))
            if any(p >= len(gs) for p in composed):
                return None
            factors = [
                Compose(g, (0,), (composed[i],)) if i in composed else g for i, g in enumerate(gs)
            ]
            return OProdV(tuple(factors)) if len(factors) > 1 else factors[0]
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def _unary_linear(node: Term, child: Term, rebuild: Callable[[Term], Term]) -> Term | None:
    """Distribution for a node **linear in one child**.

    ``L(a+b) → L(a)+L(b)``, ``L(r·v) → r·L(v)``, ``L(0) → 0``.
    """
    match child:
        case Zero():
            return Zero(_space(node))
        case Sum():
            return Sum(tuple(rebuild(c) for c in child.vectors))
        case Scale():
            return Scale(child.scalar, rebuild(child.vector))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def dist_unary(node: Term) -> Term | None:
    """Single-child linear operators.

    ``Dagger`` / ``Trace`` / the movement & reshape nodes
    (``Project``/``Embed``/``Assert``/``Lift``/``InvLift``/``Permute``/``Remetric``).
    """
    match node:
        case Dagger():
            return _unary_linear(node, node.f, Dagger)
        case Trace():
            return _unary_linear(node, node.f, lambda g: Trace(g, node.i, node.j))
        case Project():
            return _unary_linear(node, node.vector, lambda v: Project(v, node.space))
        case Embed():
            return _unary_linear(node, node.vector, lambda v: Embed(v, node.space))
        case Assert() if node.props is not None:
            # A property-carrying Assert claims something about the map as a whole, and such a
            # property does not distribute: ``Assert(a+b, bij)`` is not
            # ``Assert(a,bij) + Assert(b,bij)``, since the summands need not be bijective.
            # Keeping it atomic is what lets a downstream ``Solve`` keep its license; the
            # operand is still normalized by the traversal into ``node.vector``.
            return None
        case Assert():
            return _unary_linear(node, node.vector, lambda v: Assert(v, node.space))
        case Lift():
            return _unary_linear(node, node.vector, lambda v: Lift(v, node.fused, node.k))
        case InvLift():
            return _unary_linear(node, node.vector, lambda v: InvLift(v, node.fused, node.k))
        case Permute():
            return _unary_linear(node, node.vector, lambda v: Permute(v, node.perm))
        case TuplePermute():
            return _unary_linear(node, node.vector, lambda v: TuplePermute(v, node.perm))
        case Remetric():
            return _unary_linear(node, node.vector, lambda v: Remetric(v, node.metric))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = not a single-child linear operator [approved 2026-07-03]
            return None


# ===========================================================================
# Rule sets and the normalization engine.
# ===========================================================================

_BETA_RULES: tuple[Rule, ...] = (rule_beta_eval, rule_beta_neval)

# The always-valid algebraic cleanups (scalar/zero/identity laws, sum flattening,
# involutions) — strictly reducing, run in every preset.
_CLEANUP_RULES: tuple[Rule, ...] = (
    rule_eta,
    rule_eval_identity,
    rule_eval_compose_fold,
    rule_eval_zero,
    rule_compose,
    rule_involution,
    rule_dagger_push,
    rule_musical,
    rule_inverse_compose,
    rule_kron,
    rule_trace_identity,
    rule_trace_kron,
    rule_invlift_noop,
    rule_project_embed,
    rule_permute,
    rule_tuplepermute,
    rule_tuple_roundtrip,
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
#                                                --   distinct and sorted by `term_key`
#   mono   ::= atom                              -- coefficient 1 (implicit)
#           |  Scale(coeff, atom)                -- coeff ∉ {0, 1}
#   coeff  ::= r ∈ ℝ∖{0,1}                        -- a numeric scalar, or
#           |  s                                 -- a Scalar-valued atom (no top-level +)
#   atom   ::= a Term whose head is not Sum / Scale / Zero, is not a β-redex
#              (Eval(LLam …) / NEval(Lam …)) nor Eval(Identity …), and all of whose
#              subterms are themselves fully reduced — no distributable `+` and no
#              pullable scalar sits in any linear position.
#
# `term_key` is the total order that makes the form canonical: α-equal terms give equal
# keys, hence identical sorted output. It is `repr`-based, which is deterministic and total,
# and it is the single place the ordering is decided.
# ===========================================================================


def term_key(t: Term) -> str:
    """Give a total, deterministic ordering key on Terms, invariant under α-renaming.

    This is the order that makes canonical forms canonical: sorting ``Sum`` summands by it
    puts α-equivalent terms in the same place regardless of binder uids. ``repr`` is
    injective over these frozen dataclasses, and a term containing a binder is keyed on its
    α-canonical form first; the common binder-free case takes the direct path.
    """
    return repr(alpha_canon(t) if _has_binder(t) else t)


def _atom_of(mono: Term) -> Term:
    """Give the atom of a monomial.

    Strip a leading *numeric* ``Scale`` coefficient (matching the grouping key :func:`rule_collect` uses via
    :func:`_split_num_scale`). A *symbolic* leading ``Scale`` is a genuine scalar-product factor and stays
    part of the atom, so ``a·a`` keys on ``a·a`` (not ``a``) and a sum like ``a + a·a`` has two distinct
    atoms.
    """
    return _split_num_scale(mono)[1]


def rule_sort_sum(node: Term) -> Term | None:
    """Order ``Sum`` summands into canonical order by ``term_key`` of their atoms.

    Sorting by the *atom* (not the whole monomial) is well-defined once :func:`rule_collect` has made atoms
    distinct. Reordering ``+`` is value-safe — addition is commutative and same-space.
    """
    if not isinstance(node, Sum):
        return None
    monos = node.vectors
    order = sorted(range(len(monos)), key=lambda i: term_key(_atom_of(monos[i])))
    if order == list(range(len(monos))):
        return None
    return Sum(tuple(monos[i] for i in order))


def _top_arity(g: Term) -> int | None:
    """Give the number of top-level tensor factors of ``g``'s space (1 for a non-⊗), or ``None`` if untypable."""
    sp = _space(g)
    if sp is None:
        return None
    return len(sp.spaces) if isinstance(sp, OProd) else 1


def _has_additive_structure(t: Term) -> bool:
    """Report whether ``t`` still carries additive / scalar / zero structure.

    ``Sum``/``Scale``/``Zero`` structure that distribution and scalar-pull must dissolve
    before an ordering rule may fire.
    """
    match t:
        case Sum() | Scale() | Zero():
            return True
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): membership predicate, any other head is clean [approved 2026-07-03]
            return False


def rule_order_otimes(node: Term) -> Term | None:
    """Canonicalize the factor order of a tensor product **up to isomorphism**.

    The original layout is restored with a ``Permute`` correction (the ``𝒞``):

        ``OProdV(factors)  ≡  Permute(OProdV(sorted), σ⁻¹)``

    where ``sorted[i] = factors[σ[i]]`` (``σ`` = argsort by ``term_key``). Reordering ``⊗``
    factors changes the space (``s_a⊗s_b ≠ s_b⊗s_a``) and the array axes, so it is *not*
    value-safe on its own — the ``Permute`` swaps the axes back, making the whole rewrite
    layout-exact (verified by run-equality).

    Conservative: only fires for an unsorted product of **single-axis** factors that are
    already clean atoms (no ``Sum``/``Scale``/``Zero`` to distribute first).
    """
    match node:
        case OProdV(vectors=facs):
            if any(_has_additive_structure(f) for f in facs):
                return None  # let distribution/scalar-pull run first
            keys = [term_key(f) for f in facs]
            sigma = sorted(range(len(facs)), key=lambda i: keys[i])
            if sigma == list(range(len(facs))):
                return None  # already canonical
            if any((_top_arity(f) or 2) != 1 for f in facs):
                return None  # multi-axis factor — Permute-on-blocks not handled here
            sorted_facs = tuple(facs[i] for i in sigma)
            inv = [0] * len(sigma)  # perm = σ⁻¹ : Permute(OProdV(sorted), inv) reconstructs the original
            for position, i in enumerate(sigma):
                inv[i] = position
            return Permute(OProdV(sorted_facs), tuple(inv))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_order_oplus(node: Term) -> Term | None:
    """Canonicalize the component order of a ``⊕`` stack **up to isomorphism**.

    The original block layout is restored with a single ``TuplePermute`` correction (the ``⊕``
    analogue of the ``Permute`` used for ``⊗``):

        ``OPlusV(comps)  ≡  TuplePermute(OPlusV(sorted), σ⁻¹)``

    where ``sorted[k] = comps[σ[k]]`` (``σ`` = argsort by ``term_key``). Opt-in via
    ``order_oplus``; conservative — clean components only (no ``Sum``/``Scale``/``Zero`` to
    distribute first).
    """
    match node:
        case OPlusV(vectors=comps):
            if any(_has_additive_structure(c) for c in comps):
                return None
            if any(_space(c) is None for c in comps):
                return None
            sigma = sorted(range(len(comps)), key=lambda i: term_key(comps[i]))
            if sigma == list(range(len(comps))):
                return None  # already canonical
            inv = [0] * len(sigma)
            for k, i in enumerate(sigma):
                inv[i] = k
            return TuplePermute(OPlusV(tuple(comps[i] for i in sigma)), tuple(inv))
        case _:  # audit: ok EX-WILDCARD — rule probe: None = rule not applicable [approved 2026-07-03]
            return None


def rule_dissolve_oplus(node: Term) -> Term | None:
    """Dissolve a ``⊕`` stack into a sum of single-slot injections — the *most aggressive* ``⊕`` treatment.

    ``OPlusV(c₀,…,cₘ) → Σᵢ ιᵢ(cᵢ)`` where ``ιᵢ = Eval(TupleEmbed(⊕space, i))`` zero-pads component ``i`` into
    the full space. ``⊕`` then disappears as a former and ``+`` is strictly outermost (value-preserving).
    Opt-in via ``dissolve_oplus`` (mutually exclusive in spirit with ``order_oplus``).
    """
    if not isinstance(node, OPlusV):
        return None
    comps = node.vectors
    spaces = [_space(c) for c in comps]
    if any(s is None for s in spaces):
        return None
    full = oplus(*spaces)  # type: ignore[arg-type]
    return Sum(tuple(Eval(TupleEmbed(full, i), (0,), (comps[i],)) for i in range(len(comps))))


def _sop_rules(cfg: SoPConfig) -> tuple[Rule, ...]:
    """Assemble the rule tuple selected by ``cfg``.

    The always-valid algebraic cleanups run unconditionally; everything else is gated.
    """
    rules: list[Rule] = []
    if cfg.beta:
        rules += _BETA_RULES
    rules += _CLEANUP_RULES
    if cfg.distribute:
        rules += _DISTRIBUTE_RULES
    if cfg.push_contractions:
        rules += _CONTRACTION_RULES
    if cfg.eta_advanced:
        # After distribution (body is a flat sum of monomials) and before collect, so the
        # emitted Sum of covectors collapses (x♭ + x♭ → 2·x♭).
        rules.append(rule_eta_advanced)
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


def _is_scalar_atom(a: Term) -> bool:
    """Report whether ``a`` is a **scalar atom** of the scalar-polynomial normal form (when ``canon_scalars`` is on).

    A scalar atom is a ``Scalar``-valued Term that is never decomposed further: a scalar
    ``Var``, a scalar ``Eval`` contraction such as a dot product, or a non-linear
    ``External`` application. A numeric ``Lit`` is not one, since it folds into the
    coefficient, and neither is a scalar ``Sum``/``Scale``/``Zero``, whose additive and
    multiplicative structure belongs to the surrounding polynomial.
    """
    if not isinstance(_space(a), Scalar):
        return False
    match a:
        case Lit() | Sum() | Scale() | Zero():
            return False
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): any other scalar head defers to _is_atom_head [approved 2026-07-03]
            return _is_atom_head(a)


def _is_canonical_scalar_product(a: Term) -> bool:
    """Whether ``a`` is a **canonical scalar product**.

    The body of a scalar monomial after its numeric coefficient is peeled (the form
    :func:`rule_scalar_product` emits): ``Scale(s₀, Scale(s₁, … sₙ₋₁))``, right-nested, with
    each ``sᵢ`` a :func:`_is_scalar_atom` and the factors sorted by :func:`term_key`, so
    commuted spellings collapse together. A single scalar atom is the ``n = 1`` case. No
    numeric factor sits inside — that is the outer coefficient — and no factor hides a ``+``.
    """
    factors: list[Term] = []
    node = a
    while isinstance(node, Scale) and not _is_num(node.scalar) and isinstance(_space(node), Scalar):
        if not is_term(node.scalar) or not _is_scalar_atom(node.scalar):
            return False
        factors.append(node.scalar)
        node = node.vector
    if not _is_scalar_atom(node):  # the innermost (right-most) factor
        return False
    factors.append(node)
    if len(factors) < 2:
        return False  # a lone atom is handled by _is_atom_head directly
    keys = [term_key(f) for f in factors]
    return keys == sorted(keys)


def _is_atom_head(a: Term, canon_scalars: bool = False) -> bool:
    """Report whether ``a``'s *head* is a legal atom head.

    Not additive/scalable/zero, and not a redex the normal form forbids (β or ``Id·v``). When
    ``canon_scalars`` is on, a **canonical scalar product** (a sorted right-nested ``Scale`` of scalar atoms)
    is also a legal atom head — the multiplicative monoid of opaque scalar atoms is the polynomial's variable
    set.
    """
    match a:
        case Scale():
            return canon_scalars and _is_canonical_scalar_product(a)
        case Sum() | Zero():
            return False
        case Eval(f=LLam() | Identity()):
            return False
        case NEval(f=Lam()):
            return False
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): any other head is a legal atom head [approved 2026-07-03]
            return True


def _is_monomial(mono: Term, canon_scalars: bool = False) -> bool:
    """Report whether ``mono`` is ``atom`` or ``Scale(coeff, atom)``.

    ``coeff`` is ``∉ {0, 1}`` (numeric) or a sum-free scalar term (symbolic), and ``atom`` a
    legal atom head. With ``canon_scalars`` the atom may be a canonical scalar product, so
    ``Scale(2.0, Scale(a, b))`` (the scalar monomial ``2·a·b``) is a monomial.
    """
    match mono:
        case Sum() | Zero():
            return False
        case Scale(scalar=c):
            if _is_num(c):
                if c == 0 or c == 1:
                    return False
                return _is_atom_head(mono.vector, canon_scalars)
            if isinstance(c, Sum):
                return False  # a distributable scalar sum must have been pushed out
            # A symbolic scalar coefficient: the whole node must itself be a canonical scalar
            # product (so its factors are sorted), else it is just a single scalar atom × atom.
            if canon_scalars and isinstance(_space(mono), Scalar):
                return _is_canonical_scalar_product(mono)
            return _is_atom_head(mono.vector, canon_scalars)
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): a bare atom is the coefficient-1 monomial [approved 2026-07-03]
            return _is_atom_head(mono, canon_scalars)


def _fully_reduced(t: Term, rules: tuple[Rule, ...]) -> bool:
    """Report whether no rule in ``rules`` fires anywhere in ``t``.

    The operational "no redex remains" half of the normal-form check; the additive *shape* is
    checked separately by :func:`is_normalized_sop`.
    """
    for rule in rules:
        out = rule(t)
        if out is not None and out != t:
            return False
    return all(_fully_reduced(c, rules) for c in _child_terms(t))


def is_normalized_sop(t: Term, config: SoPConfig = DEFAULT_SOP) -> bool:
    """Certify that ``t`` is in sum-of-products normal form for ``config``.

    Two independent checks have to pass. The additive *shape* must be a ``Zero``, a lone
    monomial, or a ``Sum`` of two or more monomials whose atoms are distinct when
    ``collect`` is on and sorted by :func:`term_key` when ``order_sum`` is. And the term
    must be *fully reduced*: no rule the config enables fires anywhere in it.

    Because the rule set is not assumed confluent, this predicate rather than the rule set
    is what defines the normal form, and ``is_normalized_sop(sum_of_products(t, c), c)`` is
    the normalizer's acceptance test.
    """
    if not is_term(t):
        return False
    cs = config.canon_scalars
    match t:
        case Zero():
            return True
        case Sum(vectors=monos):
            if len(monos) < 2 or not all(_is_monomial(m, cs) for m in monos):
                return False
            keys = [term_key(_atom_of(m)) for m in monos]
            if config.collect and len(set(keys)) != len(keys):  # atoms not collected
                return False
            if config.order_sum and keys != sorted(keys):  # not in canonical order
                return False
        case _ if not _is_monomial(t, cs):  # audit: ok EX-WILDCARD — bail-default (§0): a non-Sum must be a lone monomial [approved 2026-07-03]
            return False
    if config.canon_binders and _has_binder(t) and alpha_canon(t) != t:
        return False  # bound variables are not in α-canonical form
    return _fully_reduced(t, _sop_rules(config))


def _normalize(term: Term, rules: tuple[Rule, ...]) -> Term:
    """Normalize innermost: children to a fixpoint first, then the root until nothing fires.

    Raises :class:`RewriteError` if a single node fires more than :data:`_MAX_STEPS` times,
    which means the rules are cycling rather than reducing.
    """
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
    """Normalize ``term`` to a fixpoint of a custom ``rules`` tuple.

    The open-ended entry point; :func:`simplify` / :func:`sum_of_products` are the curated
    presets.

    Raises
    ------
    RewriteError
        If ``term`` is not a Term, or the rules do not reach a fixpoint within
        :data:`_MAX_STEPS` local firings at some node.
    """
    if not is_term(term):
        raise RewriteError(f"rewrite: expected a Term, got {term!r}")
    return _normalize(term, rules)


def beta(term: Term) -> Term:
    """β-reduce every applied binder to a fixpoint, changing nothing else.

    The redexes are ``Eval(LLam …)`` for the linear binder and ``NEval(Lam …)`` for the
    general one, including partial applications, which leave a residual binder over the
    arguments not yet supplied.
    """
    return rewrite(term, _BETA_RULES)


def simplify(term: Term) -> Term:
    """β-reduce and apply the cleanups that are valid unconditionally.

    Those are the scalar, zero and identity laws, ``Sum`` flattening, the involutions,
    η-reduction, inverse cancellation and the musical collapse. Every one strictly shrinks
    the term, and none distributes a ``+``, so the result is smaller than the input and
    still in whatever shape the input had.
    """
    return rewrite(term, _SIMPLIFY_RULES)


def sum_of_products(term: Term, config: SoPConfig = DEFAULT_SOP) -> Term:
    """Rewrite ``term`` into sum-of-products normal form under ``config``.

    On top of β and the cleanups, this distributes every linear and multilinear operator
    through ``+`` — including the zero-padded ``⊕`` split, which duplicates a slot without
    duplicating its siblings — pulls scalars to the front, adds like monomials together, and
    sorts the summands canonically. The result is a ``Zero``, a single monomial, or a flat
    ``Sum`` of monomials ``c · atom`` with no distributable ``+`` left anywhere inside.
    :func:`is_normalized_sop` certifies the result against the same config.
    """
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
    "RIESZ_SOP",
    "subst",
    "rewrite",
    "beta",
    "simplify",
    "sum_of_products",
    "normal_form",
    "term_key",
    "is_normalized_sop",
    "alpha_canon",
    "normalize_space",
    "space_key",
    "spaces_iso",
]
