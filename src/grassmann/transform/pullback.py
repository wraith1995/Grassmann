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


def pullback(v: Term, target: Space) -> Term:
    """Transport the value ``v`` into ``target`` (the public entry point).

    Synthesizes ``v``'s source space and builds the transported value. An identity
    transport returns ``v`` unchanged. If ``target`` carries a metric (``WithInner``), the
    result is re-metricized so it lands in the metricized target.
    """
    if not is_term(v):
        raise PullbackError(f"pullback: expected a Term value, got {v!r}")
    if not is_space(target):
        raise PullbackError(f"pullback: expected a target Space, got {target!r}")
    from grassmann.infer.synth import synth_space

    source = synth_space(v)
    out = _pb(v, source, target)
    if isinstance(target, WithInner) and not isinstance(out, Remetric):
        out = Remetric(out, target.metric)
    return out


def _pb(v: Term, source: Space, target: Space) -> Term:
    """Build the transported value ``v : source ⤳ target`` (value-level recursion)."""
    S = _normalize(source)
    T = _normalize(target)

    # WithInner only re-metricizes; transport on the underlying space, the caller restores
    # the target metric via Remetric.
    if isinstance(S, WithInner) or isinstance(T, WithInner):
        return _pb(v, _strip(S), _strip(T))

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
        m = _tensor_map(S, T)
        return Eval(m, (0,), (v,))
    if isinstance(S, OPlus) and isinstance(T, OPlus) and len(S.spaces) == len(T.spaces):
        return _pb_oplus(v, S, T)
    if isinstance(S, Dual) and isinstance(T, Dual):
        inner = transport_map(T.space, S.space)  # contravariant: reversed primal map
        return Compose(v, (0,), (inner,))
    if _is_ffs(S) and _is_ffs(T):
        return _pb_ffs(v, S, T)

    return _pb_fallback(v, S, T)


def _pb_oplus(v: Term, S: OPlus, T: OPlus) -> Term:
    """``⊕`` at the value level: slice block ``k`` out of ``v`` (``TupleProject``), transport
    it, re-pad into ``T`` (``TupleEmbed``), and sum the blocks. Recursing on the *value*
    lets each block use any transport (incl. generic ``Project``/``Embed``)."""
    blocks: list[Term] = []
    for k, (s_k, t_k) in enumerate(zip(S.spaces, T.spaces, strict=True)):
        sliced = Eval(TupleProject(S, k), (0,), (v,))
        moved = _pb(sliced, s_k, t_k)
        blocks.append(Eval(TupleEmbed(T, k), (0,), (moved,)))
    return Sum(tuple(blocks))


def _pb_ffs(v: Term, S: Space, T: Space) -> Term:
    """FFS ``S`` to ``T`` via their ambients: ``project(convert(embed(v)))``."""
    sup_S = _ffs_superspace(S)
    sup_T = _ffs_superspace(T)
    if sup_S is None or sup_T is None:
        raise PullbackError(
            f"pullback: cannot recover the ambient space of an FFS ({S!r} → {T!r})"
        )
    embedded = Embed(v, sup_S)
    converted = _pb(embedded, sup_S, sup_T)
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


def transport_map(source: Space, target: Space) -> Term:
    """A reusable transport map ``source ⊸ target`` as a *primitive* (``Eval``-applicable)
    map term. Supports the structural cases expressible without an ``LLam``; a case that
    needs a generic ``Project``/``Embed`` (FFS, an explicit-basis subspace) raises
    :class:`PullbackError` — apply :func:`pullback` to a *value* there instead."""
    S = _normalize(source)
    T = _normalize(target)
    if isinstance(S, WithInner) or isinstance(T, WithInner):
        return transport_map(_strip(S), _strip(T))
    if S == T:
        return Identity(S)

    from grassmann.infer.subspace import is_subspace

    if is_subspace(T, S):
        i = _summand_index(T, S)
        if i is not None:
            return TupleProject(S, i)
        raise PullbackError(
            f"transport_map: {T!r} ⊑ {S!r} is not a direct ⊕-summand; no primitive map "
            f"form (apply pullback to a value, which can use a generic Project)"
        )
    if is_subspace(S, T):
        i = _summand_index(S, T)
        if i is not None:
            return TupleEmbed(T, i)
        raise PullbackError(
            f"transport_map: {S!r} ⊑ {T!r} is not a direct ⊕-summand; no primitive map "
            f"form (apply pullback to a value, which can use a generic Embed)"
        )
    if isinstance(S, OProd) and isinstance(T, OProd) and len(S.spaces) == len(T.spaces):
        return _tensor_map(S, T)
    if isinstance(S, OPlus) and isinstance(T, OPlus) and len(S.spaces) == len(T.spaces):
        # Block-diagonal map Σₖ embedₖ ∘ mₖ ∘ projectₖ — pure (dom,cod) map algebra.
        from grassmann.core.term import chain

        ms = [transport_map(s_k, t_k) for s_k, t_k in zip(S.spaces, T.spaces, strict=True)]
        return Sum(tuple(
            chain(TupleEmbed(T, k), m_k, TupleProject(S, k)) for k, m_k in enumerate(ms)
        ))
    raise PullbackError(
        f"transport_map: no primitive map for {S!r} → {T!r}; apply pullback to a value"
    )


def _tensor_map(S: OProd, T: OProd) -> Term:
    """``⊗`` — the tensor product of the factor maps, as a right-nested ``Kron`` (a
    primitive ``(dom,cod)`` map)."""
    ms = [transport_map(s_i, t_i) for s_i, t_i in zip(S.spaces, T.spaces, strict=True)]
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
    if isinstance(s, OPlus):
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


def _is_ffs(s: Space) -> bool:
    return isinstance(s, (Ker, Img, CoImg, CoKer))


def _ffs_superspace(s: Space) -> Space | None:
    """The ambient a four-fundamental-subspace sits inside (``Ker``/``CoImg`` ⊑ dom;
    ``Img``/``CoKer`` ⊑ cod), from the explicit endpoint or decoded from ``f``."""
    from grassmann.infer.synth import map_domain_codomain

    if isinstance(s, (Ker, CoImg)):
        if s.source is not None:
            return s.source
        try:
            dom, _ = map_domain_codomain(s.f)
            return dom
        except Exception:
            return None
    if isinstance(s, (Img, CoKer)):
        if s.target is not None:
            return s.target
        try:
            _, cod = map_domain_codomain(s.f)
            return cod
        except Exception:
            return None
    return None


__all__ = ["pullback", "transport_map", "PullbackError"]
