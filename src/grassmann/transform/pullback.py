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


def pullback(v: Term, target: Space, *, primitives=None) -> Term:
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


def _pb(v: Term, source: Space, target: Space, primitives=None) -> Term:
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
    if isinstance(S, Dual) and isinstance(T, Dual):  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): exploit structure else _pb_fallback [approved Teo 2026-07-03]
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


def _pb_oplus(v: Term, S: OPlus, T: OPlus, primitives=None) -> Term:
    """``⊕`` at the value level: slice block ``k`` out of ``v`` (``TupleProject``), transport
    it, re-pad into ``T`` (``TupleEmbed``), and sum the blocks. Recursing on the *value*
    lets each block use any transport (incl. generic ``Project``/``Embed``)."""
    blocks: list[Term] = []
    for k, (s_k, t_k) in enumerate(zip(S.spaces, T.spaces, strict=True)):
        sliced = Eval(TupleProject(S, k), (0,), (v,))
        moved = _pb(sliced, s_k, t_k, primitives)
        blocks.append(Eval(TupleEmbed(T, k), (0,), (moved,)))
    return Sum(tuple(blocks))


def _pb_ffs(v: Term, S: Space, T: Space, primitives=None) -> Term:
    """FFS ``S`` to ``T`` via their ambients: ``project(convert(embed(v)))``."""
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
    """No relation/shared constructor: accept only a statically dim-equal pass-through (an
    honest change-of-coordinates ``Assert``); otherwise error."""
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


def transport_map(source: Space, target: Space, *, primitives=None) -> Term:
    """A reusable transport map ``source ⊸ target`` as a *primitive* (``Eval``-applicable)
    map term. Supports the structural cases expressible without an ``LLam``; a leaf that the
    structure cannot build is taken from the caller-supplied ``primitives(source, target)``
    (opt-in), else raises :class:`PullbackError` — apply :func:`pullback` to a *value* instead."""
    S = _normalize(source)
    T = _normalize(target)
    from grassmann.infer.subspace import is_subspace

    match (S, T):
        case (WithInner(), _) | (_, WithInner()):
            return transport_map(_strip(S), _strip(T), primitives=primitives)
        case _ if S == T:
            return Identity(S)
        case (Dual(sd), Dual(td)):
            # a covector-factor map (e.g. a `⊗`/`Λᵏ` factor) is the ADJOINT of the reversed primal map.
            from grassmann.core.term import dagger

            return dagger(transport_map(td, sd, primitives=primitives))
        case _ if is_subspace(T, S):
            i = _summand_index(T, S)
            if i is not None:
                return TupleProject(S, i)
            return _leaf_primitive(S, T, primitives, "Project")
        case _ if is_subspace(S, T):
            i = _summand_index(S, T)
            if i is not None:
                return TupleEmbed(T, i)
            return _leaf_primitive(S, T, primitives, "Embed")
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
        case _:  # audit: ok EX-WILDCARD — pair-dispatch negative answer: no structural rule => caller primitives or loud PullbackError (_leaf_primitive) [approved Teo 2026-07-03]
            return _leaf_primitive(S, T, primitives, "no primitive map")


def _leaf_primitive(S: Space, T: Space, primitives, why: str) -> Term:
    """A leaf ``transport_map`` cannot build structurally: take the caller-supplied primitive map
    (``primitives(S, T)``) if any, else raise. Keeps the classic error when no ``primitives`` given."""
    if primitives is not None:
        m = primitives(S, T)
        if m is not None:
            return m
    raise PullbackError(
        f"transport_map: {why} for {S!r} → {T!r}; supply a `primitives` map or apply pullback to a value"
    )


def _tensor_map(S: OProd, T: OProd, primitives=None) -> Term:
    """``⊗`` — the tensor product of the factor maps, as a right-nested ``Kron`` (a
    primitive ``(dom,cod)`` map)."""
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
    if isinstance(s, OPlus):  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): reject/expand specific else passthrough [approved Teo 2026-07-03]
        return oplus(*(_normalize(x) for x in s.spaces))
    return s


def _strip(s: Space) -> Space:
    return s.space if isinstance(s, WithInner) else s


def _summand_index(component: Space, whole: Space) -> int | None:
    """Index ``i`` with ``whole.spaces[i] == component`` (a *direct* ⊕-summand); else
    ``None``."""
    if isinstance(whole, OPlus):
        for i, s in enumerate(whole.spaces):
            if s == component:
                return i
    return None


# Canonical FFS probes (fem audit #18) — the hand-rolled copies formerly here now live in
# grassmann.infer.subspace.
from grassmann.infer.subspace import ffs_superspace as _ffs_superspace
from grassmann.infer.subspace import is_ffs as _is_ffs


def euclidean_superspace(s: Space) -> Space:
    """A tensor-product + direct-sum space that is a SUPERSPACE of ``s``, with every FFS (``Ker``/``Img``/
    ``CoImg``/``CoKer``) replaced by its ambient, RECURSIVELY. So an exterior power ``Λᵏ(V) = Img(alt(V,k))``
    resolves to the tensor power ``V⊗ᵏ`` (an ``OProd`` of ``k`` factors); a direct sum stays an ``OPlus``;
    a ``Dual`` distributes; a ``WithInner`` drops its (non-structural) metric. Reading a form's DEGREE is
    then just counting the tensor factors of this superspace (a ``k``-form ⊑ ``V⊗ᵏ``). Idempotent on
    FFS-free (already-euclidean) spaces. Exhaustive over ``space._SPACE_TYPES`` (``assert_never`` backstop)."""
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
