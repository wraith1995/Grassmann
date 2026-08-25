"""Moving a value into another space using only the relation the two spaces already have.

``pullback(v, T)`` rewrites a value ``v : S`` into a term living in ``T``, provided ``S``
and ``T`` are related structurally. No map is applied: the whole construction is inclusions,
projections and the functorial action of the space formers, chosen by how ``S`` and ``T``
are built::

    S == T                     the value, unchanged
    T ⊑ S                      project down    (a TupleProject slice, or Project)
    S ⊑ T                      embed up        (a TupleEmbed pad, or Embed)
    S = ⊗(sᵢ), T = ⊗(tᵢ)       factorwise, as a Kron of the factor maps
    S = ⊕(sᵢ), T = ⊕(tᵢ)       slice each block, recurse, re-pad, and sum
    Dual(s) → Dual(t)          contravariant: precompose, direction reversed
    Img(f) and friends         through the ambient: project ∘ convert ∘ embed
    nothing above applies      a dimension-equal pass-through, else PullbackError

One lowering constraint shapes all of it. Only the primitive map nodes —
``TupleProject``, ``TupleEmbed``, ``Identity``, ``Kron``, ``Compose``, ``Sum`` — lower in
the canonical ``(dom, cod)`` layout that ``Eval`` and ``Compose`` read back; a map built
from an ``LLam`` lowers transposed, with its inputs riding as trailing axes. So a transport
is assembled from primitive maps and value-level moves, and never from an ``LLam`` applied
to its argument.

Carrying a value *along a map* is a different operation and lives in
:mod:`grassmann.transform.transport`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

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
    """Raised when the two spaces share no relation the structural recursion can use.

    Neither containment nor a common former applies, no caller-supplied primitive covers
    the leaf, and the dimensions do not even match, so there is nothing honest to build.
    """


# A caller-supplied leaf transport: given the (source, target) pair the structural
# recursion could not handle, return a map term, or ``None`` to decline.
LeafPrimitives = Callable[[Space, Space], Term | None]

# A caller-supplied leaf self-map, keyed on one space (see :func:`endomorphism_map`).
SelfPrimitives = Callable[[Space], Term | None]


def pullback(v: Term, target: Space, *, primitives: LeafPrimitives | None = None) -> Term:
    """Transport the value ``v`` into ``target`` (the public entry point).

    Synthesizes ``v``'s source space and builds the transported value. An identity
    transport returns ``v`` unchanged. If ``target`` carries a metric (``WithInner``), the
    result is re-metricized so it lands in the metricized target.

    ``primitives`` is an optional callable ``(source_space, target_space) -> Term | None``,
    consulted at a fundamental-subspace leaf the structural recursion cannot build on its own
    — one ``Img`` mapped into another by an inclusion only the caller knows, say. Supplying
    those leaves lets the whole functorial recursion over ``Dual``, ``⊗``, ``⊕`` and the
    exterior powers carry values across them. Omitting it means the recursion uses only what
    the spaces themselves determine.
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
    """Build the transported value, recursing on the value rather than on a map.

    The arms are tried in the order the module docstring lists: identity, containment either
    way, then the shared formers, then the fundamental subspaces, then the fallback.
    """
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
        # a covector transports contravariantly: precompose with the reversed primal map (`Compose` = the
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
    """Transport a ``⊕`` block by block, at the value level.

    Each block is sliced out of ``v`` with a ``TupleProject``, transported, re-padded into
    ``T`` with a ``TupleEmbed``, and the padded blocks are summed. Recursing on the value
    rather than building a block-diagonal map lets each block use whatever transport it
    needs, including a generic ``Project``/``Embed`` pair.
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
    """Transport a value between fundamental subspaces by going through their ambients.

    The value is embedded into ``S``'s ambient, converted there, and projected into ``T`` —
    ``project(convert(embed(v)))``.
    """
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
    """Handle a pair with no containment and no shared former.

    The one thing still defensible is a pass-through when the two spaces have the same
    static dimension, recorded as an ``Assert`` — an honest change of coordinates. Anything
    else raises :class:`PullbackError`.
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
# and exposed for reuse. Built only from Eval-compatible primitive map nodes.
# ---------------------------------------------------------------------------


def transport_map(
    source: Space, target: Space, *, primitives: LeafPrimitives | None = None
) -> Term:
    """Build the transport ``source ⊸ target`` once, as a reusable primitive map term.

    Primitive means the result lowers in the canonical ``(dom, cod)`` layout, so it can be
    applied with ``Eval`` and composed. That restricts it to the structural cases expressible
    without an ``LLam``. A leaf the structure cannot build is taken from the optional
    ``primitives(source, target)``, and failing that raises :class:`PullbackError`; applying
    :func:`pullback` to a value handles cases this cannot.
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
            # A covector-factor map (a `⊗`/`Λᵏ` factor) is the metric-free transpose of the
            # reversed primal map f = td⊸sd:  f* : Dual(sd) ⊸ Dual(td),  (f*ω)(x) = ω(f x).
            # This is not `dagger`: the metric adjoint types primal (`B⊸A`), so `Eval(dagger(f), ω)`
            # pairs a covector against a primal slot (a UnifyError), and a
            # metric may not enter a metric-free trace at all. The transpose is defined in
            # the calculus as λω. ω∘f — an `LLam`/`Compose`, no new node; `Eval(LLam,·)`
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
    """Build the self-map of ``space`` that caller-supplied leaf self-maps induce functorially.

    :func:`transport_map` cannot carry an automorphism, because its first arm short-circuits
    ``S == T`` to ``Identity``, and an automorphism — a reference-cell orientation acting on
    a weight's value space, say — is exactly a nontrivial self-map that structural equality
    cannot distinguish from the identity. This is the same functor with that shortcut
    removed: ``primitives(S)`` is consulted first at every node, keyed on the space as it
    appears in the tree and then on its normalized form, and the structure recurses when no
    leaf map is supplied —

    - ``Dual``  → the metric-free transpose of the primal self-map (:func:`_transpose_map`),
      so a leaf supplied as the pushforward action makes every covector factor carry the
      pullback ``ω ↦ ω ∘ f`` automatically;
    - ``⊗``     → the ``Kron`` of the factor self-maps;
    - ``⊕``     → block-diagonal (embedₖ ∘ mₖ ∘ projectₖ, summed);
    - ``Img(f)`` → conjugation through the ambient (``project ∘ endo(ambient) ∘ embed`` via the
      corestriction maps ``Img`` itself determines — :func:`_ffs_immersion`). Sound exactly
      when the ambient action maps the image into itself, which holds for the functorial
      actions this serves (an antisymmetrizer commutes with ``f⊗ᵏ``, so ``Λᵏ`` images are
      invariant); a caller whose ambient action does not preserve the image must supply the
      ``Img`` leaf directly.
    - ``Scalar`` → ``Identity``; ``WithInner`` strips (represent consults no metric here).

    A ``Ker``/``CoKer``/``CoImg``/``App`` node with no supplied primitive raises
    :class:`PullbackError`, since no structural rule determines an induced self-map there.

    ``primitives`` is required, and takes a single space rather than the pair
    :func:`transport_map` takes, because source and target coincide. With no leaf action at
    all everything reduces to ``Identity``, so there would be nothing to call it for.
    """
    from typing import assert_never

    from grassmann.core.space import Scalar as _ScalarSpace
    from grassmann.core.term import chain

    # Consult `primitives` on the space as stored (before any expansion) — the caller keys its
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
            # Expand one level, keeping the summand as stored so nested primitives still match.
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
            # A projector image (e.g. an exterior power's `Img(Alt)`): no corestriction, so
            # conjugate with the value-level Embed/Project (the `_pb_ffs` mechanism) inside an
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
    """Build the ``Img(f) ↔ ambient`` transport out of the map that defines the image.

    An ``Img(f)`` is a subspace of ``cod(f)`` but not a summand of it, so no ⊕-slice reaches
    it. The map that does reach it is already in hand: the node carries ``f``, and from
    ``f``, its pseudo-inverse, and the corestriction ``f~ : dom(f) ⊸ Img(f)``
    (:class:`~grassmann.core.term.Corestrict`, the first-isomorphism-theorem isomorphism,
    defined for an injective map) both directions compose::

        project  cod(f) ⊸ Img(f) :  f~ ∘ f⁺    land in the image's own coordinates
        embed    Img(f) ⊸ cod(f) :  f  ∘ f~⁺   read back out into the ambient

    The corestrictions are what make these well typed, and the endpoints are the whole
    point. A bare ``Pinv(f)`` has the right *values* but the type ``cod(f) ⊸ dom(f)``, which
    silently replaces the ``Img`` endpoint with ``dom(f)``. ``Project`` and ``Embed`` take a
    value rather than a map, so they never inspect a map endpoint and such a substitution
    passes unnoticed — until a consumer that does check it, ``Compose`` and hence the
    contravariant ``Dual`` arm of :func:`_pb`, refuses to unify ``Img(f)`` with ``dom(f)``.
    ``Corestrict`` is the primitive that types a map *into* an ``Img``, and with it the two
    directions above are ordinary composites: no ``LLam``, no coercion.

    ``f⁺`` is the Moore–Penrose pseudo-inverse. On an injective ``f`` it is the exact inverse
    of the corestriction on the image, and metric-independent there. :func:`pullback` then
    recurses over ``Dual``, ``⊗``, ``⊕`` and ``Λᵏ`` on top of these for free — a covector
    transports by the metric-free transpose of this very map.

    This belongs in the arm rather than in the caller because it is a bail in the
    canonical-arm sense (``docs/auditing.md``): it is always available and always correct,
    and needs nothing from outside. ``primitives`` is an escape hatch and an optimization,
    and correctness must never depend on one being supplied.

    Returns ``None`` when neither side is an ``Img`` with a usable ambient, or when the
    defining map is not asserted ``inj`` and so lies outside ``Corestrict``'s scope, leaving
    the ``primitives``/:class:`PullbackError` path to handle it.
    """
    from grassmann.core.space import Img
    from grassmann.core.term import Corestrict, Pinv, chain

    # Direction matters: `transport_map(S, T)` promises a map whose codomain is `T`. `f ∘ f⁺` has
    # codomain `cod(f)` (the ambient) in both directions; used as the map into an `Img(f)` it
    # typechecks loosely, since `Img(f) ⊆ cod(f)`, but the transported value then carries ambient
    # coordinates instead of the image's own. A consumer that reads coordinates off the result — the
    # metric-free k-form integrate lane sums output coordinates against reference weights — silently
    # gets `edim` of them where the image has `rank(f)`. The corestriction fixes the type and the
    # coordinates in one move.
    if isinstance(T, Img) and _normalize(_img_ambient(T)) == _normalize(S):
        if not _corestrictable(T.f):
            return None
        return chain(Corestrict(T.f), Pinv(T.f))    # project `cod(f) ⊸ Img(f)`, image's own coords
    if isinstance(S, Img) and _normalize(_img_ambient(S)) == _normalize(T):
        if not _corestrictable(S.f):
            return None
        return chain(S.f, Pinv(Corestrict(S.f)))    # embed  `Img(f) ⊸ cod(f)`
    return None


def _corestrictable(f: Term) -> bool:
    """Report whether ``Corestrict(f)`` may be formed, which requires ``f`` asserted ``inj``."""
    from grassmann.infer.properties import is_injective

    return is_injective(f)


def _ffs_chart_to_chart(S: Space, T: Space) -> Term | None:
    """Build the ``Img(f_A) ⊸ Img(f_B)`` map between two immersions of one ambient space.

    The chart-to-chart convert leaves A's own coordinates, passes through the ambient, and
    arrives in B's::

        f~_B ∘ f_B⁺ ∘ f_A ∘ f~_A⁺

        Img(f_A) ─▶ dom(f_A) ─▶ ambient ─▶ dom(f_B) ─▶ Img(f_B)

    The value level reaches the same place by embedding into the ambient, converting, and
    projecting (:func:`_pb_ffs`). The map level needs this arm because a covector transports
    by the *reversed primal map*: a ``Dual(Img(f_A)) → Dual(Img(f_B))`` asks
    :func:`transport_map` for ``Img(f_B) ⊸ Img(f_A)`` as a term. Without the arm that request
    falls through to a caller-supplied primitive and raises, for a relation the two spaces
    fully determine — which is why a form-valued degree of freedom transports and a
    covector-valued one would not.

    The endpoints are what the two corestrictions are for. The bare middle of the chain,
    ``Pinv(f_B) ∘ f_A``, synthesizes as ``dom(f_A) ⊸ dom(f_B)`` rather than
    ``Img(f_A) ⊸ Img(f_B)`` — a difference no value-level consumer notices, since
    ``Project`` and ``Embed`` never inspect a map endpoint, and one that ``Compose`` — the
    very node the ``Dual`` arm needs — refuses. Forcing the endpoints with an ``LLam``
    wrapper is worse: an ``LLam``-built map lowers transposed, its inputs riding as trailing
    axes, which is exactly what the module docstring warns against.
    :class:`~grassmann.core.term.Corestrict` is the primitive that types a map into an
    ``Img``, and with both corestrictions in place the chain is an ordinary well-typed
    composite. Each ``f~`` is an isomorphism, so ``Pinv(f~_A)`` and ``f~_B`` are exact.

    They are not decoration either: ``f~_A⁺`` divides out A's image-basis ``R`` factor and
    ``f~_B`` multiplies in B's, so the composite converts between the two *realized image
    bases* rather than between the two parameter domains, which agree only when the two
    ``R`` factors have equal determinant.

    This is also the map that carries an orientation. Two parameterizations of one edge are
    two ``Img`` spaces over a single ambient, and the convert between them is exactly ``−1``
    on a reversed edge. The sign lives in the composite, so it survives the sign-fixed QR
    (``diag R > 0``) that would cancel it out of either realized basis on its own.

    Both sides must be strict sub-ambient immersions. A full-dimensional ``Img`` — a cell's
    own ``J``, where ``Img(J)`` is the whole ambient — is not a chart-to-chart pair: the
    face-into-cell inclusion ``Img(J_face) ⊸ Img(J_cell)`` is spelled in the consumer, with a
    constant reference inclusion this arm cannot know, so answering it here would mistype
    that path. A consumer supplies it through ``primitives``, which is consulted first.
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
    # applied right-to-left: out of A's image basis, into A's domain, into the ambient, into B's
    # domain, into B's image basis.
    return chain(Corestrict(T.f), Pinv(T.f), S.f, Pinv(Corestrict(S.f)))


def _img_ambient(img: Space) -> Space:
    """Give the ambient an ``Img(f)`` sits inside — its declared ``target``, else ``cod(f)``."""
    from grassmann.infer.synth import map_domain_codomain

    if img.target is not None:
        return img.target
    return map_domain_codomain(img.f)[1]


def _leaf_primitive(
    S: Space, T: Space, primitives: LeafPrimitives | None, why: str
) -> Term:
    """Take the caller's primitive map for a leaf the structural recursion cannot build.

    Consults ``primitives(S, T)`` when one was supplied, and otherwise raises
    :class:`PullbackError`: there is no relation left to exploit.
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

    It is the derived form ``λω. ω ∘ f``, an ``LLam`` over a ``Compose``, definable in the
    calculus without a new node. In any bases the stored matrix is exactly ``fᵀ``: no Gram
    matrix is consulted, which is what makes the transpose metric-free rather than the
    adjoint.
    """
    from grassmann.core.term import Compose, LLam, var

    w = var(f"transp_w{next(_TRANSPOSE_COUNTER)}", Dual(sd))
    return LLam((w,), Compose(w, (0,), (f,)))


def _tensor_map(
    S: OProd, T: OProd, primitives: LeafPrimitives | None = None
) -> Term:
    """Build the ``⊗`` transport by combining the factor maps into a right-nested ``Kron``.

    The result is a primitive map in the canonical ``(dom, cod)`` layout, so it can be
    applied or composed directly.
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
    """Reject an unresolved space, and expand a static-count ``⊕ⁿ s`` to the explicit ``⊕``.

    The expansion recurses into ``⊕`` children, so a nested ``⊕ⁿ Scalar`` — the ``ℝⁿ`` ground
    — reads the same whether it sits at the top level or as a summand, and structural
    summand matching sees a single canonical form.
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
    """Give the index of ``component`` among ``whole``'s summands, or ``None``.

    Only a direct summand matches; a space buried inside a nested ``⊕`` does not.
    """
    if isinstance(whole, OPlus):
        for i, s in enumerate(whole.spaces):
            if s == component:
                return i
    return None


# Canonical FFS probes live in grassmann.infer.subspace.
from grassmann.infer.subspace import ffs_superspace as _ffs_superspace
from grassmann.infer.subspace import is_ffs as _is_ffs


def euclidean_superspace(s: Space) -> Space:
    """Give the superspace of ``s`` built from ⊗ and ⊕ alone, with every subspace resolved.

    Each fundamental subspace — ``Ker``, ``Img``, ``CoImg``, ``CoKer`` — is replaced by the
    ambient it sits in, recursively, so an exterior power ``Λᵏ(V) = Img(alt(V,k))`` resolves
    to the tensor power ``V⊗ᵏ``. A direct sum stays a direct sum, a ``Dual`` distributes over
    what it wraps, and a ``WithInner`` drops its metric, which carries no structure here.

    Reading a form's degree then reduces to counting the tensor factors of this superspace,
    since a ``k``-form is a subspace of ``V⊗ᵏ``. The function is idempotent on a space that
    has no fundamental subspace in it, and exhaustive over the Space sum.
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
