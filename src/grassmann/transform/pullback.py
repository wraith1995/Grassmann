"""Auto pullback — transport a value between spaces (`§40`/`§41 §7`).

``pullback(v, T)`` rewrites a value term ``v : S`` into a new term living in the
target space ``T``, inserting projections/embeddings and recursing structurally:

- ``S == T``                       → unchanged
- ``T ⊑ S``                        → project down (``TupleProject`` slice / ``Project``)
- ``S ⊑ T``                        → embed up   (``TupleEmbed`` pad / ``Embed``)
- ``⊗``                            → componentwise via a ``Kron`` of factor maps
- ``⊕``                            → slice each block, recurse, re-pad, and sum
- ``Dual``                         → **contravariant** (precompose, direction reversed)
- FFS (``Ker``/``Img``/…)          → via the ambient: ``proj ∘ conv ∘ embed``
- otherwise                        → a dim-equal pass-through, else :class:`PullbackError`.

Adapted from FEMOracle ``base/pullback.py::toFromMap`` to the n-ary, positions-based
contraction view. A key lowering constraint shapes the construction: only the *primitive*
map nodes (``TupleProject``/``TupleEmbed``/``Identity``/``Kron``/``Compose``/``Sum``) lower
in the canonical ``(dom, cod)`` layout that ``Eval``/``Compose`` read — an ``LLam``-built
map lowers transposed (its inputs ride as trailing axes). So the transport is assembled
purely from primitive maps and value-level moves, never an ``LLam``-and-apply.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from grassmann.core.space import (
    CoImg,
    CoKer,
    DimLit,
    Dual,
    Img,
    Ker,
    OPlus,
    OPlusRep,
    OProd,
    Space,
    SpaceMeta,
    SVar,
    WithInner,
    is_space,
    oplus,
)
from grassmann.core.term import (
    Compose,
    Embed,
    Eval,
    Identity,
    Kron,
    Project,
    Remetric,
    Sum,
    Term,
    TupleEmbed,
    TupleProject,
    assert_,
    is_term,
)


class PullbackError(Exception):
    """Raised when no transport between two spaces can be built."""


# A caller-supplied leaf transport: given the (source, target) pair the structural
# recursion could not handle, return a map term, or ``None`` to decline.
LeafPrimitives = Callable[[Space, Space], Term | None]

# A caller-supplied leaf SELF-map, keyed on one space (see :func:`endomorphism_map`).
SelfPrimitives = Callable[[Space], Term | None]


def pullback(v: Term, target: Space, *, primitives: LeafPrimitives | None = None) -> Term:
    """Transport the value ``v`` into ``target`` (the public entry point).

    Synthesizes ``v``'s source space and builds the transported value. An identity
    transport returns ``v`` unchanged. If ``target`` carries a metric (``WithInner``), the
    result is re-metricized so it lands in the metricized target.

    ``primitives`` (opt-in; ``None`` ⇒ classic behaviour, byte-identical) is a callable
    ``(source_space, target_space) -> Term | None`` consulted at an FFS leaf that the structural
    recursion cannot build itself (e.g. one ``Img`` mapped into another by a caller-supplied
    inclusion). It lets a caller supply the primitive maps grassmann can't infer, so the whole
    functorial recursion (``Dual``/``⊗``/``⊕``/exterior powers) transports across them generally.
    """
    if not is_term(v):
        raise PullbackError(f"pullback: expected a Term value, got {v!r}")
    if not is_space(target):
        raise PullbackError(f"pullback: expected a target Space, got {target!r}")
    from grassmann.infer.synth import synth_space

    source = synth_space(v)
    out = _pb(v, source, target, primitives)
    if isinstance(target, WithInner) and not isinstance(out, Remetric):
        out = Remetric(out, target.metric)
    return out


def _pb(v: Term, source: Space, target: Space, primitives: LeafPrimitives | None = None) -> Term:
    """Build the transported value ``v : source ⤳ target`` (value-level recursion)."""
    S = _normalize(source)
    T = _normalize(target)

    # WithInner only re-metricizes; transport on the underlying space, the caller restores
    # the target metric via Remetric.
    if isinstance(S, WithInner) or isinstance(T, WithInner):
        return _pb(v, _strip(S), _strip(T), primitives)

    if S == T:
        return v

    from grassmann.infer.subspace import is_subspace

    if is_subspace(T, S):  # T ⊑ S  ⇒  project down
        i = _summand_index(T, S)
        if i is not None:
            return Eval(TupleProject(S, i), (0,), (v,))
        return Project(v, T)
    if is_subspace(S, T):  # S ⊑ T  ⇒  embed up
        i = _summand_index(S, T)
        if i is not None:
            return Eval(TupleEmbed(T, i), (0,), (v,))
        return Embed(v, T)

    # --- structural recursion ---
    if isinstance(S, OProd) and isinstance(T, OProd) and len(S.spaces) == len(T.spaces):
        m = _tensor_map(S, T, primitives)
        return Eval(m, (0,), (v,))
    if isinstance(S, OPlus) and isinstance(T, OPlus) and len(S.spaces) == len(T.spaces):
        return _pb_oplus(v, S, T, primitives)
    if isinstance(S, Dual) and isinstance(T, Dual):  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): exploit structure else _pb_fallback [approved 2026-07-03]
        # a covector transports CONTRAVARIANTLY: precompose with the reversed PRIMAL map (`Compose` = the
        # metric-free adjoint action). The primal FFS leaf's map comes from `primitives` (the chart inclusion).
        inner = transport_map(T.space, S.space, primitives=primitives)
        return Compose(v, (0,), (inner,))
    if _is_ffs(S) and _is_ffs(T):
        if primitives is not None:                        # a caller-supplied leaf map wins over the ambient route
            m = primitives(S, T)
            if m is not None:
                return Eval(m, (0,), (v,))
        return _pb_ffs(v, S, T, primitives)

    return _pb_fallback(v, S, T)


def _pb_oplus(
    v: Term, S: OPlus, T: OPlus, primitives: LeafPrimitives | None = None
) -> Term:
    """``⊕`` at the value level.

    Slice block ``k`` out of ``v`` (``TupleProject``), transport it, re-pad into ``T`` (``TupleEmbed``), and
    sum the blocks. Recursing on the *value* lets each block use any transport (incl. generic
    ``Project``/``Embed``).
    """
    blocks: list[Term] = []
    for k, (s_k, t_k) in enumerate(zip(S.spaces, T.spaces, strict=True)):
        sliced = Eval(TupleProject(S, k), (0,), (v,))
        moved = _pb(sliced, s_k, t_k, primitives)
        blocks.append(Eval(TupleEmbed(T, k), (0,), (moved,)))
    return Sum(tuple(blocks))


def _pb_ffs(
    v: Term, S: Space, T: Space, primitives: LeafPrimitives | None = None
) -> Term:
    """Transport an FFS ``S`` to ``T`` via their ambients: ``project(convert(embed(v)))``."""
    sup_S = _ffs_superspace(S)
    sup_T = _ffs_superspace(T)
    if sup_S is None or sup_T is None:
        raise PullbackError(
            f"pullback: cannot recover the ambient space of an FFS ({S!r} → {T!r})"
        )
    embedded = Embed(v, sup_S)
    converted = _pb(embedded, sup_S, sup_T, primitives)
    return Project(converted, T)


def _pb_fallback(v: Term, S: Space, T: Space) -> Term:
    """No relation/shared constructor.

    Accept only a statically dim-equal pass-through (an honest change-of-coordinates ``Assert``); otherwise
    error.
    """
    from grassmann.lower._dims import DynamicDim, dim_of

    try:
        ds, dt = dim_of(S), dim_of(T)
    except DynamicDim as e:
        raise PullbackError(
            f"pullback: no subspace relation, shared constructor, or static dimension to "
            f"transport {S!r} → {T!r} ({e})"
        ) from e
    if ds != dt:
        raise PullbackError(
            f"pullback: cannot transport {S!r} (dim {ds}) → {T!r} (dim {dt}); spaces are "
            f"unrelated and of unequal dimension"
        )
    return assert_(v, T)


# ---------------------------------------------------------------------------
# Primitive (dom, cod) transport maps — for the ⊗ (Kron) and Dual (precompose) cases,
# and exposed for reuse. Built ONLY from Eval-compatible primitive map nodes.
# ---------------------------------------------------------------------------


def transport_map(
    source: Space, target: Space, *, primitives: LeafPrimitives | None = None
) -> Term:
    """Build a reusable transport map ``source ⊸ target`` as a *primitive* map term.

    Primitive means ``Eval``-applicable.

    Supports the structural cases expressible without an ``LLam``; a leaf that the structure cannot build is
    taken from the caller-supplied ``primitives(source, target)`` (opt-in), else raises :class:`PullbackError`
    — apply :func:`pullback` to a *value* instead.
    """
    S = _normalize(source)
    T = _normalize(target)
    from grassmann.infer.subspace import is_subspace

    match (S, T):
        case (WithInner(), _) | (_, WithInner()):
            return transport_map(_strip(S), _strip(T), primitives=primitives)
        case _ if S == T:
            return Identity(S)
        case (Dual(sd), Dual(td)):
            # A covector-factor map (a `⊗`/`Λᵏ` factor) is the metric-free TRANSPOSE of the
            # reversed primal map f = td⊸sd:  f* : Dual(sd) ⊸ Dual(td),  (f*ω)(x) = ω(f x).
            # NOT `dagger` — the metric adjoint types PRIMAL (`B⊸A`), so `Eval(dagger(f), ω)`
            # pairs a covector against a primal slot (the F1 UnifyError), and a
            # metric may not enter a metric-free trace at all. The transpose is DEFINED in
            # the calculus: λω. ω∘f — an `LLam`/`Compose`, no new node; `Eval(LLam,·)`
            # β-reduces to the same `Compose` the value-level Dual branch above builds.
            f = transport_map(td, sd, primitives=primitives)
            return _transpose_map(f, sd, td)
        case _ if is_subspace(T, S):
            i = _summand_index(T, S)
            if i is not None:
                return TupleProject(S, i)
            m = _ffs_immersion(S, T)
            if m is not None:
                return m
            return _leaf_primitive(S, T, primitives, "Project")
        case _ if is_subspace(S, T):
            i = _summand_index(S, T)
            if i is not None:
                return TupleEmbed(T, i)
            m = _ffs_immersion(S, T)
            if m is not None:
                return m
            return _leaf_primitive(S, T, primitives, "Embed")
        case _ if _ffs_chart_to_chart(S, T) is not None:
            return _ffs_chart_to_chart(S, T)
        case (OProd(ss), OProd(ts)) if len(ss) == len(ts):
            return _tensor_map(S, T, primitives)
        case (OPlus(ss), OPlus(ts)) if len(ss) == len(ts):
            # Block-diagonal map Σₖ embedₖ ∘ mₖ ∘ projectₖ — pure (dom,cod) map algebra.
            from grassmann.core.term import chain

            ms = [transport_map(s_k, t_k, primitives=primitives)
                  for s_k, t_k in zip(ss, ts, strict=True)]
            return Sum(tuple(
                chain(TupleEmbed(T, k), m_k, TupleProject(S, k)) for k, m_k in enumerate(ms)
            ))
        case _:  # audit: ok EX-WILDCARD — pair-dispatch negative answer: no structural rule => caller primitives or loud PullbackError (_leaf_primitive) [approved 2026-07-03]
            return _leaf_primitive(S, T, primitives, "no primitive map")


def endomorphism_map(space: Space, *, primitives: SelfPrimitives) -> Term:
    """Build the functorial SELF-map of ``space`` induced by caller-supplied leaf self-maps.

    ``transport_map``'s structural recursion cannot carry an AUTOMORPHISM: its first arm
    short-circuits ``S == T`` to ``Identity``, and an automorphism (e.g. a reference-cell
    orientation acting on a DOF weight's value space) is exactly a nontrivial self-map that
    structural equality cannot see. This is the same functor with the shortcut removed:
    ``primitives(S)`` is consulted FIRST at every node (a supplied leaf self-map wins, keyed
    on the space as it appears in the tree, then on its normalized form), and the structure
    recurses otherwise —

    - ``Dual``  → the metric-free transpose of the primal self-map (:func:`_transpose_map`),
      so with the leaf supplied as the PUSHFORWARD action every covector factor automatically
      carries the PULLBACK ``ω ↦ ω ∘ f``;
    - ``⊗``     → the ``Kron`` of the factor self-maps;
    - ``⊕``     → block-diagonal (embedₖ ∘ mₖ ∘ projectₖ, summed);
    - ``Img(f)``→ conjugation through the ambient (``project ∘ endo(ambient) ∘ embed`` via the
      corestriction maps ``Img`` itself determines — :func:`_ffs_immersion`). Sound exactly
      when the ambient action maps the image into itself, which holds for the functorial
      actions this serves (an antisymmetrizer commutes with ``f⊗ᵏ``, so ``Λᵏ`` images are
      invariant); a caller whose ambient action does NOT preserve the image must supply the
      ``Img`` leaf directly.
    - ``Scalar``→ ``Identity``; ``WithInner`` strips (represent consults no metric here).

    A ``Ker``/``CoKer``/``CoImg``/``App`` node with no supplied primitive raises
    :class:`PullbackError` — no structural rule determines an induced self-map there.

    ``primitives`` is REQUIRED and is a callable ``space -> Term | None`` (single-space —
    unlike ``transport_map``'s pair form — since source and target coincide). With no leaf
    action everything reduces to ``Identity``; call nothing instead.
    """
    from typing import assert_never

    from grassmann.core.space import Scalar as _ScalarSpace
    from grassmann.core.term import chain

    # Consult `primitives` on the space AS STORED (before any expansion) — the caller keys its
    # leaves off the same tree this recursion walks, and eagerly normalizing first would rewrite
    # an `OPlusRep` ground (`S.base`) into `OPlus` before the callback could ever match it.
    p = primitives(space)
    if p is not None:
        return p
    S = space
    match S:
        case SVar() | SpaceMeta():
            raise PullbackError(
                f"endomorphism_map: space {S!r} is not fully determined; realize it first"
            )
        case OPlusRep(count=DimLit() as n, space=summand):
            # Expand ONE level, keeping the summand as stored so nested primitives still match.
            return endomorphism_map(oplus(*([summand] * n.value)), primitives=primitives)
        case OPlusRep():
            raise PullbackError(
                f"endomorphism_map: symbolic-count direct sum {S!r} is not supported"
            )
        case WithInner():
            return endomorphism_map(_strip(S), primitives=primitives)
        case _ScalarSpace():
            return Identity(S)
        case Dual(sd):
            f = endomorphism_map(sd, primitives=primitives)
            return _transpose_map(f, sd, sd)
        case OProd(ss):
            ms = [endomorphism_map(s_i, primitives=primitives) for s_i in ss]
            out = ms[-1]
            for m in reversed(ms[:-1]):
                out = Kron(m, out)
            return out
        case OPlus(ss):
            ms = [endomorphism_map(s_k, primitives=primitives) for s_k in ss]
            return Sum(tuple(
                chain(TupleEmbed(S, k), m_k, TupleProject(S, k)) for k, m_k in enumerate(ms)
            ))
        case Img():
            ambient = _img_ambient(S)                     # as stored — primitives may key on it
            inner = endomorphism_map(ambient, primitives=primitives)
            project = _ffs_immersion(ambient, S)
            embed = _ffs_immersion(S, ambient)
            if project is not None and embed is not None:
                return chain(project, inner, embed)       # inj defining map: exact corestriction
            # A PROJECTOR image (e.g. an exterior power's `Img(Alt)`): no corestriction, so
            # conjugate with the VALUE-level Embed/Project (the `_pb_ffs` mechanism) inside an
            # `LLam` — Eval-able like the Dual arm's transpose.
            from grassmann.core.term import LLam, var as _var

            v = _var(f"endo_v{next(_TRANSPOSE_COUNTER)}", S)
            return LLam((v,), Project(Eval(inner, (0,), (Embed(v, ambient),)), S))
        case Ker() | CoKer() | CoImg() | App():
            raise PullbackError(
                f"endomorphism_map: no structural rule induces a self-map on {S!r}; supply it "
                "via `primitives`."
            )
        case _ as unreachable:
            assert_never(unreachable)


def _ffs_immersion(S: Space, T: Space) -> Term | None:
    """Build the `Img(f) ↔ ambient` transport from the map that DEFINES the image.

    The canonical BAIL for a subspace pair where neither side is a direct ⊕-summand.

    `Img(f)` is a subspace of `cod(f)` but not a *summand* of it, so `_summand_index` returns `None`
    and both subspace arms used to fall straight to `_leaf_primitive` — i.e. they demanded a
    CALLER-SUPPLIED map for something the space already determines. But `Img` CARRIES `f`
    (`Img.f: Term`), so the maps are ours to build, out of `f`, its pseudo-inverse, and the
    CORESTRICTION `f~ : dom(f) ⊸ Img(f)` (:class:`~grassmann.core.term.Corestrict` — the
    first-isomorphism-theorem iso, injective-only):

        project  `cod(f) ⊸ Img(f)` :  `f~ ∘ f⁺`   — land in the image's OWN coordinates
        embed    `Img(f) ⊸ cod(f)` :  `f  ∘ f~⁺`  — read out of them, back into the ambient

    HISTORY (load-bearing — this is what previous attempts got wrong). The arm used to return the
    BARE `Pinv(T.f)` / `S.f`, whose *values* are right but whose *types* are `cod(f) ⊸ dom(f)` and
    `dom(f) ⊸ cod(f)`: the `Img` endpoint was silently replaced by `dom(f)`. `Project`/`Embed` are
    VALUE-level ops (they take a value `v`), so they never check a map endpoint and the substitution
    went unnoticed — until a consumer that checks the endpoint STRICTLY (`Compose`, hence the
    contravariant `Dual` arm of `_pb`) rejected it with a `UnifyError` `Img(f) ~ dom(f)`. There was
    no way to type a map INTO an `Img` at all; `Corestrict` is exactly that missing primitive, and
    with it the two directions are ordinary well-typed composites — no `LLam`, no coercion.

    `f⁺` is the Moore–Penrose `Pinv`; on an INJECTIVE `f` it is the exact corestriction inverse on
    the image and is metric-independent there (the same fact `pointwise`'s `restrict` relies on when
    it uses `Pinv(J_face)`). `pullback` then recurses `Dual`/`⊗`/`⊕`/`Λᵏ` over these for free — a
    covector transports by the metric-free transpose of this very map.

    Why this belongs HERE and not in the caller: it is a **bail** in the canonical-arm sense (docs/
    auditing.md §0) — the arm handles the case itself, always available and always correct, needing
    nothing from outside. `primitives` (like `supplied_frames`) is a caller-supplied ESCAPE HATCH and
    an OPTIMIZATION; correctness must never depend on one. A consumer supplying this leaf by hand was
    re-deriving what `Img` already knows, and had to wrap it in an `LLam` whose free map then failed
    to realize in a vmap sub-body (`41 §7`).

    Returns `None` when neither side is an `Img` with a usable ambient, or when the defining map is
    not asserted `inj` (`Corestrict`'s scope) — leaving the existing `primitives`/`PullbackError`
    path exactly as it was.
    """
    from grassmann.core.space import Img
    from grassmann.core.term import Corestrict, Pinv, chain

    # DIRECTION MATTERS — `transport_map(S, T)` promises a map whose CODOMAIN IS `T`. `f ∘ f⁺` has
    # codomain `cod(f)` (the ambient) in BOTH directions; used as the map into an `Img(f)` it
    # typechecks loosely, since `Img(f) ⊆ cod(f)`, but the transported value then carries AMBIENT
    # coordinates instead of the image's own. A consumer that reads coordinates OFF the result — the
    # metric-free k-form integrate lane sums output COORDINATES against reference weights — silently
    # gets `edim` of them where the image has `rank(f)`. The corestriction fixes the type AND the
    # coordinates in one move.
    if isinstance(T, Img) and _normalize(_img_ambient(T)) == _normalize(S):
        if not _corestrictable(T.f):
            return None
        return chain(Corestrict(T.f), Pinv(T.f))    # PROJECT `cod(f) ⊸ Img(f)`, image's own coords
    if isinstance(S, Img) and _normalize(_img_ambient(S)) == _normalize(T):
        if not _corestrictable(S.f):
            return None
        return chain(S.f, Pinv(Corestrict(S.f)))    # EMBED  `Img(f) ⊸ cod(f)`
    return None


def _corestrictable(f: Term) -> bool:
    """Report whether `Corestrict(f)` may be formed — `f` must be asserted `inj`."""
    from grassmann.infer.properties import is_injective

    return is_injective(f)


def _ffs_chart_to_chart(S: Space, T: Space) -> Term | None:
    """Build the `Img(f_A) ⊸ Img(f_B)` MAP for two immersions of the SAME ambient.

    The chart-to-chart convert, out of A's own coordinates, through the ambient, into B's:

        ``f~_B ∘ f_B⁺ ∘ f_A ∘ f~_A⁺``   :   `Img(f_A) → dom(f_A) → ambient → dom(f_B) → Img(f_B)`

    The VALUE level already had this (``_pb_ffs``: embed to the ambient, convert, project). The MAP level
    did not: ``transport_map`` is reached from the ``Dual`` branch — a covector transports by the
    *reversed primal map*, so a `Dual(Img(f_A)) → Dual(Img(f_B))` needs `transport_map(Img(f_B), Img(f_A))`
    as a TERM — and no arm matched, so it fell to `_leaf_primitive` and raised for a relation the spaces
    fully determine. That asymmetry is why a form-valued DOF could be transported but a covector-valued
    one could not.

    HISTORY (load-bearing). The first fix here answered with the bare middle of that chain,
    ``chain(Pinv(f_B), f_A)``, which synthesizes as `dom(f_A) ⊸ dom(f_B)` — NOT `Img(f_A) ⊸ Img(f_B)`.
    It survived only because the consumers of the day were value-level (`Project`/`Embed` never check a
    map endpoint); `Compose` — the very node the `Dual` arm needs — rejects it. Wrapping the whole thing
    in an `LLam` to force the endpoints was the other attempt, and that is worse: an `LLam`-built map
    lowers TRANSPOSED (its inputs ride as trailing axes), which is exactly what the module docstring
    warns against, and its free map failed to realize inside a vmap sub-body. The real gap was the
    missing primitive — a map whose endpoint IS an `Img`. That is
    :class:`~grassmann.core.term.Corestrict`, and with the two corestrictions in place the chain above is
    an ordinary well-typed primitive composite: no `LLam`, no coercion, and `Pinv(f~_A)`/`f~_B` are exact
    (each `f~` is an isomorphism).

    The corestrictions are not decoration: `f~_A⁺` divides out A's image-frame `R` factor and `f~_B`
    multiplies in B's, so the composite is a genuine change of coordinates between the two realized
    image frames rather than between the two parameter domains (which agree only when |R_A| = |R_B|).

    This is also the map that carries a σ ORIENTATION. Two parameterizations of one edge (canonical and
    permuted) are two `Img` spaces over one ambient, and the convert between them is exactly `−1` on a
    reversed edge — the sign living in the composite, so it survives the frames' sign-fixed QR
    (`diag R > 0`), which cancels the sign out of either realized basis on its own.

    BOTH sides must be STRICT sub-ambient immersions. A full-dimensional `Img` (a cell's own `J`, where
    `Img(J) == Rᵉ`) is not a chart-to-chart pair: the face-into-cell inclusion `Img(J_face) ⊸ Img(J_cell)`
    has its own honest spelling in the consumer (built inside the calculus, with the constant reference
    `ι_std` this cannot know), and answering it here mistypes that path. Consumers supply it through
    `primitives`, which is consulted first.
    """
    from grassmann.core.space import Img
    from grassmann.core.term import Corestrict, Pinv, chain
    from grassmann.lower._dims import DynamicDim, dim_of

    if not (isinstance(S, Img) and isinstance(T, Img)) or S.f is T.f:
        return None
    if _normalize(_img_ambient(S)) != _normalize(_img_ambient(T)):
        return None
    if not (_corestrictable(S.f) and _corestrictable(T.f)):
        return None
    try:
        for space in (S, T):
            if dim_of(space) >= dim_of(_img_ambient(space)):
                return None
    except DynamicDim:
        return None
    # applied right-to-left: out of A's image frame, into A's domain, into the ambient, into B's
    # domain, into B's image frame.
    return chain(Corestrict(T.f), Pinv(T.f), S.f, Pinv(Corestrict(S.f)))


def _img_ambient(img: Space) -> Space:
    """Give the ambient space an `Img(f)` sits inside — its declared `target`, else `cod(f)`."""
    from grassmann.infer.synth import map_domain_codomain

    if img.target is not None:
        return img.target
    return map_domain_codomain(img.f)[1]


def _leaf_primitive(
    S: Space, T: Space, primitives: LeafPrimitives | None, why: str
) -> Term:
    """Take the caller-supplied primitive map for a leaf the structure cannot build.

    Uses ``primitives(S, T)`` if any, else raises :class:`PullbackError` with the plain
    "no transport" message.
    """
    if primitives is not None:
        m = primitives(S, T)
        if m is not None:
            return m
    raise PullbackError(
        f"transport_map: {why} for {S!r} → {T!r}; supply a `primitives` map or apply pullback to a value"
    )


_TRANSPOSE_COUNTER = itertools.count()


def _transpose_map(f: Term, sd: Space, td: Space) -> Term:
    """Build the metric-free transpose ``f* : Dual(sd) ⊸ Dual(td)`` of ``f : td ⊸ sd``.

    As the DERIVED form ``λω. ω ∘ f`` (LLam + Compose — definable in the calculus, no new
    node). In any bases the stored matrix is exactly ``fᵀ``; no Gram is consulted anywhere.
    """
    from grassmann.core.term import Compose, LLam, var

    w = var(f"transp_w{next(_TRANSPOSE_COUNTER)}", Dual(sd))
    return LLam((w,), Compose(w, (0,), (f,)))


def _tensor_map(
    S: OProd, T: OProd, primitives: LeafPrimitives | None = None
) -> Term:
    """Build the ``⊗`` transport — the factor maps as a right-nested ``Kron``.

    The result is a primitive ``(dom, cod)`` map.
    """
    ms = [transport_map(s_i, t_i, primitives=primitives)
          for s_i, t_i in zip(S.spaces, T.spaces, strict=True)]
    out = ms[-1]
    for m in reversed(ms[:-1]):
        out = Kron(m, out)
    return out


# ---------------------------------------------------------------------------
# Space helpers.
# ---------------------------------------------------------------------------


def _normalize(s: Space) -> Space:
    """Reject unresolved spaces and expand *static-count* ``OPlusRep`` to ``OPlus``.

    Recurses into ``⊕`` children so a nested ``⊕ⁿ Scalar`` (the new ``ℝⁿ`` ground) expands the
    same way whether it sits at the top level or as a summand — so structural summand matching
    sees one canonical form.
    """
    if isinstance(s, (SVar, SpaceMeta)):
        raise PullbackError(
            f"pullback: space {s!r} is not fully determined; annotate/realize it first"
        )
    if isinstance(s, OPlusRep):
        if isinstance(s.count, DimLit):
            return oplus(*([_normalize(s.space)] * s.count.value))
        raise PullbackError(
            f"pullback: symbolic-count direct sum {s!r} is not supported; give a concrete "
            f"count"
        )
    if isinstance(s, OPlus):  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): reject/expand specific else passthrough [approved 2026-07-03]
        return oplus(*(_normalize(x) for x in s.spaces))
    return s


def _strip(s: Space) -> Space:
    return s.space if isinstance(s, WithInner) else s


def _summand_index(component: Space, whole: Space) -> int | None:
    """Give the index ``i`` with ``whole.spaces[i] == component``, else ``None``.

    Only a *direct* ⊕-summand matches.
    """
    if isinstance(whole, OPlus):
        for i, s in enumerate(whole.spaces):
            if s == component:
                return i
    return None


# Canonical FFS probes — the hand-rolled copies formerly here now live in
# grassmann.infer.subspace.
from grassmann.infer.subspace import ffs_superspace as _ffs_superspace
from grassmann.infer.subspace import is_ffs as _is_ffs


def euclidean_superspace(s: Space) -> Space:
    """Give a tensor-product + direct-sum SUPERSPACE of ``s`` with every FFS resolved.

    Every FFS (``Ker``/``Img``/``CoImg``/``CoKer``) is replaced by its ambient, RECURSIVELY.
    So an exterior power ``Λᵏ(V) = Img(alt(V,k))`` resolves to the tensor power ``V⊗ᵏ`` (an
    ``OProd`` of ``k`` factors); a direct sum stays an ``OPlus``; a ``Dual`` distributes; a
    ``WithInner`` drops its (non-structural) metric. Reading a form's DEGREE is then just
    counting the tensor factors of this superspace (a ``k``-form ⊑ ``V⊗ᵏ``). Idempotent on
    FFS-free (already-euclidean) spaces. Exhaustive over the Space sum type
    (``assert_never`` backstop).
    """
    from typing import assert_never

    from grassmann.core import space as S
    match s:
        case S.Scalar() | S.SVar() | S.App() | S.SpaceMeta():
            return s                                              # euclidean leaves / opaque atoms
        case S.Dual(space=inner):
            return S.Dual(euclidean_superspace(inner))
        case S.OProd(spaces=ss):
            return S.OProd(tuple(euclidean_superspace(f) for f in ss))
        case S.OPlus(spaces=ss):
            return S.OPlus(tuple(euclidean_superspace(f) for f in ss))
        case S.OPlusRep(count=c, space=b):
            return S.OPlusRep(c, euclidean_superspace(b))
        case S.WithInner(space=b):
            return euclidean_superspace(b)                        # the metric is not structural — drop it
        case S.Ker() | S.Img() | S.CoImg() | S.CoKer():
            amb = _ffs_superspace(s)
            return euclidean_superspace(amb) if amb is not None else s
        case _:
            assert_never(s)


__all__ = ["pullback", "transport_map", "PullbackError"]
