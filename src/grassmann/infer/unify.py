"""First-order unification on SPACE terms (`25 §2`).

This is **not** a higher-order unifier. It unifies *space* metavariables (``?α``,
modeled by :class:`SpaceMeta`) and decomposes structurally over the Space ADT. Per the
locked decision (`25 §F5`):

- **Dimensions are compared (``≡``), never unified** — ``Base(n,d) ~ Base(n,d')`` checks
  ``d ≡ d'`` via :mod:`grassmann.infer.dims`.
- **Nats/ints stay out of unification** — int-indexed ops are primitives; an
  ``OPlusRep`` count must match by dim-equality, a ground ``?n`` only by a ground nat
  (`25 §2`, last line). We do not introduce nat metavars.

A :class:`SpaceMeta` is a fresh placeholder space (it is itself a ``Space`` subtype so it
can sit inside core nodes during inference); the :class:`MetaStore` records solutions and
performs the occurs-check.
"""

from __future__ import annotations

from typing import assert_never

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
    Space,
    SpaceMeta,
    SVar,
    WithInner,
    is_space,
)

from .dims import dim_eq


def _expand_rep(s: Space) -> Space:
    """Expand a static-count ``OPlusRep(n, s)`` to the explicit ``OPlus(s, …, s)`` (else
    pass through), so ``⊕ⁿ s`` and ``⊕(s,…,s)`` are unified as the same space."""
    if isinstance(s, OPlusRep) and isinstance(s.count, DimLit):
        return OPlus(tuple([s.space] * s.count.value))
    return s


class UnifyError(Exception):
    """Raised when two spaces cannot be unified (`25 §2` FAIL)."""


def fresh_meta(hint: str = "α") -> SpaceMeta:
    """Allocate a fresh space metavariable ``?α`` (`SpaceMeta` is a core Space node)."""
    return SpaceMeta(hint=hint)


class MetaStore:
    """The metavar store ``Θ`` (`25 §0`): solutions for :class:`SpaceMeta`."""

    def __init__(self) -> None:
        self._solved: dict[int, Space] = {}

    def __repr__(self) -> str:
        return f"MetaStore({self._solved!r})"

    def is_solved(self, m: SpaceMeta) -> bool:
        return m.uid in self._solved

    def get(self, m: SpaceMeta) -> Space | None:
        return self._solved.get(m.uid)

    def solve(self, m: SpaceMeta, s: Space) -> None:
        self._solved[m.uid] = s

    def resolve(self, s: Space) -> Space:
        """Apply the current substitution all the way down (a *zonk*)."""
        s = self.walk(s)
        match s:
            case SpaceMeta():
                return s
            case Dual():
                return Dual(self.resolve(s.space))
            case WithInner():
                return WithInner(self.resolve(s.space), s.metric)
            case OPlus():
                return OPlus(tuple(self.resolve(c) for c in s.spaces))
            case OProd():
                return OProd(tuple(self.resolve(c) for c in s.spaces))
            case OPlusRep():
                return OPlusRep(s.count, self.resolve(s.space))
            case _:  # audit: ok EX-WILDCARD — bail-default (§0): zonk recursion; wildcard returns node unchanged [approved 2026-07-03]
                return s

    def walk(self, s: Space) -> Space:
        """Follow solved metavars one chain step (shallow)."""
        while isinstance(s, SpaceMeta):
            nxt = self._solved.get(s.uid)
            if nxt is None:
                return s
            s = nxt
        return s

    def has_unsolved(self, s: Space) -> bool:
        """True iff ``resolve(s)`` still mentions an unsolved metavar (`25 §5`).

        Descends into FFS defining maps / endpoints and ``App`` like
        :func:`grassmann.infer.subst.subst_space` — a metavar hiding inside
        ``Img(f, source=…)`` counts. ``WithInner.metric``
        is skipped for parity with ``subst_space``; ``Dim``s carry no metavars."""
        s = self.walk(s)
        match s:
            case SpaceMeta():
                return True
            case Dual() | WithInner():
                return self.has_unsolved(s.space)
            case OPlus() | OProd():
                return any(self.has_unsolved(c) for c in s.spaces)
            case OPlusRep():
                return self.has_unsolved(s.space)
            case Ker() | Img() | CoKer() | CoImg():
                return (
                    _space_pred_in_term(s.f, self.has_unsolved)
                    or (s.source is not None and self.has_unsolved(s.source))
                    or (s.target is not None and self.has_unsolved(s.target))
                )
            case App():
                return _space_pred_in_term(s.g, self.has_unsolved) or any(
                    self.has_unsolved(a) if is_space(a) else _space_pred_in_term(a, self.has_unsolved)
                    for a in s.args
                )
            case Scalar() | SVar():
                return False
            case _ as unreachable:
                assert_never(unreachable)


def _occurs(store: MetaStore, m: SpaceMeta, s: Space) -> bool:
    """Occurs-check with the same FFS/``App`` descent as :meth:`MetaStore.has_unsolved`
    (a cyclic ``m:= Img(f, source=…m…)`` solution must be rejected —)."""
    s = store.walk(s)

    def pred(sp: Space) -> bool:
        return _occurs(store, m, sp)

    match s:
        case SpaceMeta():
            return s.uid == m.uid
        case Dual() | WithInner():
            return _occurs(store, m, s.space)
        case OPlus() | OProd():
            return any(_occurs(store, m, c) for c in s.spaces)
        case OPlusRep():
            return _occurs(store, m, s.space)
        case Ker() | Img() | CoKer() | CoImg():
            return (
                _space_pred_in_term(s.f, pred)
                or (s.source is not None and _occurs(store, m, s.source))
                or (s.target is not None and _occurs(store, m, s.target))
            )
        case App():
            return _space_pred_in_term(s.g, pred) or any(
                _occurs(store, m, a) if is_space(a) else _space_pred_in_term(a, pred)
                for a in s.args
            )
        case Scalar() | SVar():
            return False
        case _ as unreachable:
            assert_never(unreachable)


def _space_pred_in_term(term: object, pred) -> bool:
    """True iff ``pred(space)`` holds for some Space embedded in ``term`` — the same
    dataclass-field descent as :func:`grassmann.infer.subst._subst_space_in_term`
    (binder var spaces, asserted spaces, nested FFS maps, ``App`` args)."""
    from dataclasses import fields, is_dataclass

    from grassmann.core.term import is_term

    def in_node(node: object) -> bool:
        for fld in fields(node):
            val = getattr(node, fld.name)
            for c in val if isinstance(val, tuple) else (val,):
                if is_space(c) and pred(c):
                    return True
                if is_term(c) and in_node(c):
                    return True
        return False

    return is_dataclass(term) and in_node(term)


def unify(a: Space, b: Space, store: MetaStore) -> None:
    """Unify ``a ~ b`` (`25 §2`), mutating ``store``. Raises :class:`UnifyError`.

    Dimensions are compared by ``≡`` (never unified); nats are not unified.
    """
    if not is_space(a) or not is_space(b):
        raise UnifyError(f"unify: both sides must be spaces, got {a!r} ~ {b!r}")
    a = store.walk(a)
    b = store.walk(b)

    match (a, b):
        # Metavariable cases (?α ~ s).
        case (SpaceMeta(), SpaceMeta()) if a.uid == b.uid:
            return
        case (SpaceMeta(), _):
            _solve_meta(a, b, store)
            return
        case (_, SpaceMeta()):
            _solve_meta(b, a, store)
            return

        # Structural cases.
        case (Scalar(), Scalar()):
            return
        case (SVar(), SVar()):
            if a.name != b.name:
                raise UnifyError(f"space variable mismatch: {a.name} ~ {b.name}")
            return
        case (Dual(), Dual()):
            unify(a.space, b.space, store)
            return
        case (OPlus(), OPlus()):
            _unify_seq(a.spaces, b.spaces, store, "⊕")
            return
        case (OProd(), OProd()):
            na, nb = _drop_scalar_factors(a), _drop_scalar_factors(b)
            if (isinstance(na, OProd) and isinstance(nb, OProd)
                    and len(na.spaces) == len(nb.spaces)):
                _unify_seq(na.spaces, nb.spaces, store, "⊗")
            elif na is not a or nb is not b:
                # dropping the tensor-unit (Scalar) changed a side ⇒ re-dispatch. Guarded on ACTUAL
                # progress: `_drop_scalar_factors` returns its argument UNCHANGED when there is no
                # unit to drop, so re-dispatching unconditionally re-entered `unify` on the IDENTICAL
                # pair and recursed until the stack blew (measured: still `RecursionError` at a
                # 20000-frame limit, the same pair observed at depth 100 and 300).
                unify(na, nb, store)
            else:
                # Two unit-free products of DIFFERENT arity. Mathematically these can still denote one
                # space, because ⊗ is associative and this model encodes a map as a tensor
                # (`A ⊸ (B ⊸ C) ≅ A*⊗B*⊗C`): `tensor_power(V, k)` builds the FLAT product while
                # `linear_map_space(cod, dom) = oprod(dual(dom), cod)` RIGHT-NESTS as soon as the
                # codomain is compound, so `V*⊗V*⊗V*` meets `V*⊗(V*⊗V*)`. Λ³ is the first degree at
                # which that is observable (for k = 2 both sides are arity 2), and it is what blocks
                # any Λ³-valued lane.
                #
                # Deciding it HERE would be wrong: `shape_of_space` makes ⊗ factors SEPARATE AXES, so
                # the two spellings realize as `(3,3,3)` and `(3,9)`; unifying them would let
                # mismatched layouts meet silently. Splicing at construction (in `oprod`) is likewise
                # not a drop-in — it flattens a map space's `Dual(dom) ⊗ cod` into its component axes,
                # so `Alt_3 : V⊗³ ⊸ V⊗³` stops being the `(27,27)` MATRIX the SVD seam expects
                # (measured). Resolving it needs the value-tensor / map-space distinction settled in
                # the representation. Until then, FAIL LOUDLY rather than hang.
                raise UnifyError(
                    f"⊗ arity mismatch: {len(a.spaces)} ≢ {len(b.spaces)} factors, and neither side "
                    "has a Scalar (⊗-unit) factor to drop, so there is no rewrite that makes "
                    "progress.\nThis is usually ONE space spelled two ways — a flat "
                    "`tensor_power(V, k)` product meeting a right-nested map space "
                    "`linear_map_space` builds by currying. They are isomorphic under the tensor "
                    "encoding of maps but have different AXIS structure, so they must be made to "
                    "agree at construction, not here.\n"
                    f"  a = {a!r}\n  b = {b!r}")
            return
        # Scalar is the ⊗-unit (``X ⊗ ℝ ≅ X``) and a single-factor product is its factor
        # (``OProd((X,)) ≅ X`` — a k=1 exterior/tensor power ``V⊗¹``). So an ``A ⊸ ℝ`` (≡ ``A* ⊗
        # Scalar``) unifies with the bare dual ``A*`` from ``flat``, and ``Λ¹(V)`` (in ``V⊗¹``)
        # with a bare ``V`` value.
        case (OProd(), _) if len(a.spaces) == 1 or any(isinstance(x, Scalar) for x in a.spaces):
            unify(_drop_scalar_factors(a), b, store)
            return
        case (_, OProd()) if len(b.spaces) == 1 or any(isinstance(x, Scalar) for x in b.spaces):
            unify(a, _drop_scalar_factors(b), store)
            return
        case (OPlusRep(), OPlusRep()):
            if not dim_eq(a.count, b.count):
                raise UnifyError(f"⊕ⁿ count mismatch: {a.count!r} ≢ {b.count!r}")
            unify(a.space, b.space, store)
            return
        # ``⊕ⁿ s`` (OPlusRep) and the explicit ``⊕(s,…,s)`` (OPlus) are the SAME space (the new
        # ground ``ℝⁿ = ⊕ⁿ Scalar`` can appear either way); a static-count OPlusRep expands to
        # OPlus and unifies component-wise.
        case (OPlusRep(), OPlus()) | (OPlus(), OPlusRep()):
            unify(_expand_rep(a), _expand_rep(b), store)
            return
        case (WithInner(), WithInner()):
            unify(a.space, b.space, store)
            if not _metric_eq(a.metric, b.metric):
                raise UnifyError(f"WithInner metric mismatch: {a.metric!r} ≠ {b.metric!r}")
            return
        # A metricized space ``WithInner(s, g)`` is the underlying ``s`` carrying extra
        # metric *metadata*: dimension- and contraction-identical. So it unifies with a
        # bare ``s`` by stripping the metric (the metric only matters for musical/metric
        # ops, which re-read it from the annotated side). This lets a tangent-space value
        # contract against its base space (e.g. the covariant-derivative connection).
        case (WithInner(), _):
            unify(a.space, b, store)
            return
        case (_, WithInner()):
            unify(a, b.space, store)
            return
        case ((Ker() | Img() | CoKer() | CoImg()), _) if type(a) is type(b):
            # FFS unify by α-equivalence of the keying map (`25 §2`): two independently-built
            # FFS spaces — e.g. `Img(alt(V,k))` from two `exterior_power` sites — differ only in
            # binder uids, yet denote the SAME subspace. Fast path is raw equality; on a raw
            # mismatch fall back to α-canonical comparison (binders renamed to a canonical form).
            if a.f != b.f and not _ffs_map_alpha_eq(a.f, b.f):  # type: ignore[attr-defined]
                raise UnifyError(f"FFS map mismatch: {a!r} ~ {b!r}")
            return
        case (App(), App()):
            if a.g != b.g or a.positions != b.positions or a.args != b.args:
                raise UnifyError(f"SpaceFunction application mismatch: {a!r} ~ {b!r}")
            return
        # Involution ``s** ≡ s`` (`20 §5`).  Tried only AFTER the structural cases (so
        # ``Dual(Dual(V)) ~ Dual(?α)`` still solves ``?α := Dual(V)`` via the ``(Dual,
        # Dual)`` rule above, never collapsing prematurely).  When ONE side is a bare
        # double dual and the other is NOT a ``Dual`` (e.g. a contraction against an
        # already-dual factor — what differentiating through a ``Dagger`` produces:
        # ``⊕ⁿ ℝ ~ Dual(Dual(⊕ⁿ ℝ))``), peel the outer pair and retry.  For these
        # finite-dim Euclidean spaces the double dual is an identity, not just an iso.
        case (Dual(), _) if isinstance(store.walk(a.space), Dual):
            unify(store.walk(a.space).space, b, store)
            return
        case (_, Dual()) if isinstance(store.walk(b.space), Dual):  # audit: ok EX-ISINSTANCE-LADDER — narrowing: substructure guards inside exhaustive match [approved 2026-07-03]
            unify(a, store.walk(b.space).space, store)
            return
        case _:  # audit: ok EX-WILDCARD — unification-failure arm: no rule <=> not unifiable; the raise IS the algorithm's negative answer (pairs, not node dispatch) [approved 2026-07-03]
            raise UnifyError(f"cannot unify {a!r} ~ {b!r}")


def _solve_meta(m: SpaceMeta, s: Space, store: MetaStore) -> None:
    s = store.walk(s)
    if isinstance(s, SpaceMeta) and s.uid == m.uid:
        return
    if _occurs(store, m, s):
        raise UnifyError(f"occurs-check: {m!r} occurs in {s!r}")
    store.solve(m, s)


def _drop_scalar_factors(s: OProd) -> Space:
    """An ``OProd`` with its ``Scalar`` (⊗-unit) factors removed and a single-factor product
    collapsed to its factor (``OProd((X,)) ≅ X``); an all-scalar product becomes ``Scalar``."""
    kept = tuple(x for x in s.spaces if not isinstance(x, Scalar))
    if not kept:
        return Scalar()
    if len(kept) == 1:
        return kept[0]
    return s if len(kept) == len(s.spaces) else OProd(kept)


def _ffs_map_alpha_eq(f: object, g: object) -> bool:
    """Are two FFS keying maps α-equivalent (equal up to renaming of bound variables)?  Used so
    independently-constructed FFS spaces (``Img(alt(V,k))`` etc.) unify.  Lazy import of
    ``alpha_canon`` avoids an ``infer ← transform`` import cycle; only hit on a raw mismatch."""
    try:
        from grassmann.transform.rewrite import alpha_canon
    except Exception:  # noqa: BLE001 - transform unavailable ⇒ fall back to raw inequality
        return False
    return alpha_canon(f) == alpha_canon(g)


def _metric_eq(a: object, b: object) -> bool:
    """Structural equality of two ``WithInner`` metric **Terms**.

    The metric is always a Term (enforced in ``space.WithInner``), so plain ``==`` (the frozen
    dataclass's structural equality) is well-defined — no numpy-array fallback needed.
    """
    return a is b or bool(a == b)


def _unify_seq(
    xs: tuple[Space, ...], ys: tuple[Space, ...], store: MetaStore, op: str
) -> None:
    if len(xs) != len(ys):
        raise UnifyError(f"{op} arity mismatch: {len(xs)} vs {len(ys)}")
    for x, y in zip(xs, ys, strict=True):
        unify(x, y, store)


__all__ = [
    "UnifyError",
    "SpaceMeta",
    "MetaStore",
    "fresh_meta",
    "unify",
]
