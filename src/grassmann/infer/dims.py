"""Symbolic dimension arithmetic, and the line between a static size and a runtime rank.

A ``Dim`` (from :mod:`grassmann.core.space`) is a Nat-expression, and this module answers
three questions about one: what is ``|s|``, the dimension of a space; what is ``rank(f)``
for a map; and do two dimensions denote the same number.

The interesting part is the second. Grassmann never guesses a rank. A rank is either
something the user asserted, in which case it is a static expression that arithmetic can
work with, or it is genuinely unknown until numbers exist, in which case it becomes an
opaque **runtime rank atom** ``δ_f`` — a :class:`~grassmann.core.space.DimAtomRef` keyed
by the map ``f``, which lowering materializes as the numeric rank an SVD reports::

    rank(f)
      ├── f asserted bij/inj/surj, or given an explicit rank
      │       ⇒ static:  |dom f|, |cod f|, or the asserted r
      └── nothing asserted
              ⇒ δ_f  — opaque here, an SVD-sized dynamic axis at lowering

Both flow through the same arithmetic. ``|Ker f| = |dom f| − rank(f)`` is written the same
way whichever branch supplied the rank; only when a value must be laid out does the
difference bite, because a ``δ_f`` sizes a dynamic array axis while a static dimension
sizes a fixed one. :func:`is_runtime_dim` is how a caller asks which it has.

Equality is decided by normalizing both sides to a canonical polynomial over the opaque
generators — ``DimVar``, ``DimOf`` and ``DimAtomRef`` are the indeterminates, with
``δ_f ≡ δ_g`` exactly when ``f`` and ``g`` are structurally the same term. Dimensions are
compared this way and never unified: unification solves space metavariables, and a
dimension mismatch is a mismatch rather than something to solve for.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import assert_never

from grassmann.core.space import (
    App,
    CoImg,
    CoKer,
    Dim,
    DimAtomRef,
    DimLit,
    DimMod,
    DimNeg,
    DimOf,
    DimProd,
    DimSum,
    DimVar,
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
    dim_prod,
    dim_sum,
    is_dim,
    is_space,
)


class DimError(Exception):
    """Raised when a dimension cannot be computed — an unresolved metavariable, say."""


# ===========================================================================
# |s| — the dimension of a space.
# ===========================================================================


def dim_of(s: Space) -> Dim:
    """Compute ``|s|``, the dimension of a space, as a symbolic ``Dim``.

    Structural spaces reduce by the obvious arithmetic: a direct sum adds its components,
    a tensor product multiplies them, a dual and a metricized space keep the dimension of
    what they wrap. The four fundamental subspaces
    (:class:`~grassmann.core.space.Ker` / :class:`~grassmann.core.space.Img` /
    :class:`~grassmann.core.space.CoKer` / :class:`~grassmann.core.space.CoImg`) instead
    need the defining map's rank and its endpoints: :func:`rank_of` supplies the rank —
    static or the runtime atom ``δ_f`` — and the endpoints come from the node when it
    carries them, otherwise from synthesizing the map's space.
    """
    match s:
        case Scalar():
            return DimLit(1)
        case SVar():
            # |SVar α| is an opaque but static Nat keyed by the space var — a dimension the
            # caller has not named, not one that waits for the numbers.
            return DimOf(s)
        case Dual():
            return dim_of(s.space)
        case WithInner():
            return dim_of(s.space)
        case OPlus():
            return dim_sum(*(dim_of(c) for c in s.spaces))
        case OProd():
            return dim_prod(*(dim_of(c) for c in s.spaces))
        case OPlusRep():
            return dim_prod(s.count, dim_of(s.space))
        case Img():
            return rank_of(s.f)
        case CoImg():
            return rank_of(s.f)
        case Ker():
            # |Ker f| = |dom f| − rank(f)
            dom = _ffs_source(s)
            return dim_diff(dim_of(dom), rank_of(s.f))
        case CoKer():
            # |CoKer f| = |cod f| − rank(f)
            cod = _ffs_target(s)
            return dim_diff(dim_of(cod), rank_of(s.f))
        case App() | SpaceMeta():
            raise DimError(f"dim_of: unsupported space {s!r}")
        case _ as unreachable:
            assert_never(unreachable)


def _ffs_source(node: Ker | Img | CoKer | CoImg) -> Space:
    if node.source is not None:
        return node.source
    # Recover dom(f) from f's synthesized map space. Lazy import to avoid a cycle.
    from .synth import map_domain_codomain

    dom, _cod = map_domain_codomain(node.f)
    return dom


def _ffs_target(node: Ker | Img | CoKer | CoImg) -> Space:
    if node.target is not None:
        return node.target
    from .synth import map_domain_codomain

    _dom, cod = map_domain_codomain(node.f)
    return cod


# ===========================================================================
# rank(f) — a static dimension when asserted, else the runtime atom δ_f.
# ===========================================================================


def rank_of(f: object) -> Dim:
    """Resolve ``rank(f)`` — a static dimension when asserted, otherwise the atom ``δ_f``.

    An explicit ``rank`` or ``dim_img`` on the map's ``Var`` wins outright. Failing that,
    ``bij`` and ``inj`` both give ``|dom f|``, ``surj`` gives ``|cod f|``, and an asserted
    kernel dimension gives ``|dom f| − dim_ker``. With nothing asserted the rank is not
    knowable before the numbers are, so the result is the runtime atom
    :class:`~grassmann.core.space.DimAtomRef` keyed by ``f`` — opaque to every caller here,
    and materialized by an SVD at lowering.
    """
    from grassmann.core.props import MapProps
    from grassmann.core.term import Var

    props: MapProps | None = getattr(f, "props", None) if isinstance(f, Var) else None
    if props is not None and not props.is_empty():
        # Explicit rank / dim_img wins.
        if props.rank is not None:
            return props.rank
        if props.dim_img is not None:
            return props.dim_img
        # bij ⇒ |dom| = |cod| = rank.
        if props.bij:
            dom, _cod = _map_dom_cod(f)
            return dim_of(dom)
        # inj ⇒ rank = |dom|.
        if props.inj:
            dom, _cod = _map_dom_cod(f)
            return dim_of(dom)
        # surj ⇒ rank = |cod|.
        if props.surj:
            _dom, cod = _map_dom_cod(f)
            return dim_of(cod)
        # dim_ker given ⇒ rank = |dom| − dim_ker.
        if props.dim_ker is not None:
            dom, _cod = _map_dom_cod(f)
            return dim_diff(dim_of(dom), props.dim_ker)
    # No usable assertion ⇒ runtime SVD atom δ_f.
    return DimAtomRef(f)


def _map_dom_cod(f: object) -> tuple[Space, Space]:
    from .synth import map_domain_codomain

    return map_domain_codomain(f)


# ===========================================================================
# Dim arithmetic + normalized equality (d ≡ d').
# ===========================================================================


def dim_diff(a: Dim, b: Dim) -> Dim:
    """Compute the symbolic difference ``a − b``.

    The core ``Dim`` algebra has no subtraction node, so the difference is taken on the
    normalized polynomial and a ``Dim`` is reconstructed from the result. Monomials with a
    negative coefficient come back wrapped in ``DimNeg``, which is legal as an
    intermediate: ``|Ker f| = |dom f| − rank(f)`` is written this way and is non-negative
    only once the rank is known. The surface form is not load-bearing, since
    :func:`dim_eq` re-normalizes both sides before comparing.
    """
    poly = _poly_sub(_to_poly(a), _to_poly(b))
    return _from_poly(poly)


def dim_eq(a: object, b: object) -> bool:
    """Decide ``a ≡ b`` by normalizing both to the same canonical Nat polynomial.

    ``DimVar``, ``DimOf`` and ``DimAtomRef`` are the indeterminates of that polynomial, so
    ``2·n`` and ``n + n`` are equal while ``n`` and ``m`` are not. Two runtime rank atoms
    are the same indeterminate exactly when their keying terms are structurally equal —
    frozen-dataclass ``==`` on the term — so ``δ_f ≡ δ_g`` iff ``f`` and ``g`` are the same
    map.
    """
    if not _is_dimlike(a) or not _is_dimlike(b):
        raise DimError(f"dim_eq: expected two Dims, got {a!r} and {b!r}")
    return _to_poly(a) == _to_poly(b)  # type: ignore[arg-type]


def _is_dimlike(d: object) -> bool:
    return isinstance(
        d, (DimLit, DimVar, DimOf, DimAtomRef, DimSum, DimProd, DimNeg, DimMod)
    )


# --- Polynomial normal form -------------------------------------------------
#
# A polynomial is a dict: monomial -> int coefficient, where a monomial is a
# frozen sorted tuple of (generator, exponent) pairs. The empty monomial () is the
# constant term. Generators are the opaque atoms, made hashable below.

_Monomial = tuple[tuple[Hashable, int], ...]
_Poly = dict[_Monomial, int]


def _gen_key(d: DimVar | DimOf | DimAtomRef) -> Hashable:
    """Give the hashable key identifying one indeterminate of the polynomial form.

    Two generators share a key exactly when they are the same indeterminate, which for a
    runtime rank atom means their keying maps are structurally equal.
    """
    match d:
        case DimVar():
            return ("var", d.name)
        case DimOf():
            # Spaces are frozen dataclasses ⇒ hashable & structurally compared.
            return ("dimof", d.space)
        case DimAtomRef():
            return ("atom", d.key)
        case _ as unreachable:
            assert_never(unreachable)


def _to_poly(d: Dim) -> _Poly:
    match d:
        case DimLit():
            return {} if d.value == 0 else {(): d.value}
        case DimVar() | DimOf() | DimAtomRef():
            return {((_gen_key(d), 1),): 1}
        case DimSum():
            out: _Poly = {}
            for t in d.terms:
                _poly_iadd(out, _to_poly(t))
            return out
        case DimProd():
            acc: _Poly = {(): 1}
            for t in d.terms:
                acc = _poly_mul(acc, _to_poly(t))
            return acc
        case DimNeg():
            return {mono: -c for mono, c in _to_poly(d.term).items()}  # type: ignore[arg-type]
        case DimMod():
            return _mod_poly(d)
        case _ as unreachable:
            assert_never(unreachable)


def _poly_const(p: _Poly) -> int | None:
    """Return the integer value of ``p`` if it is a pure constant (no generators)."""
    if not p:
        return 0
    if len(p) == 1 and () in p:
        return p[()]
    return None


def _mod_poly(d: DimMod) -> _Poly:
    """Normalize ``a mod b``.

    - Both operands static ⇒ constant-fold to ``a % b`` (Python non-negative remainder);
      a static zero modulus is an error.
    - Otherwise opaque: key a fresh generator by the *normalized* operand polynomials so
      ``(a mod b) ≡ (a' mod b')`` iff ``a ≡ a'`` and ``b ≡ b'``. ``a mod 1 ≡ 0`` is the one
      symbolic identity we fold (a literal-1 modulus).
    """
    tp = _to_poly(d.term)
    mp = _to_poly(d.modulus)
    t_const = _poly_const(tp)
    m_const = _poly_const(mp)
    if m_const == 0:
        raise DimError(f"_to_poly: modulus of {d!r} is statically zero")
    if m_const == 1:
        return {}  # a mod 1 ≡ 0
    if t_const is not None and m_const is not None:
        return {} if t_const % m_const == 0 else {(): t_const % m_const}
    key = ("mod", _poly_canon(tp), _poly_canon(mp))
    return {((key, 1),): 1}


def _poly_canon(p: _Poly) -> Hashable:
    """Give a hashable, order-independent canonical form of a polynomial (for keying)."""
    return tuple(sorted(p.items(), key=lambda kv: repr(kv[0])))


def _poly_iadd(dst: _Poly, src: _Poly) -> None:
    for mono, coef in src.items():
        dst[mono] = dst.get(mono, 0) + coef
        if dst[mono] == 0:
            del dst[mono]


def _poly_sub(a: _Poly, b: _Poly) -> _Poly:
    out = dict(a)
    for mono, coef in b.items():
        out[mono] = out.get(mono, 0) - coef
        if out[mono] == 0:
            del out[mono]
    return out


def _mono_mul(m1: _Monomial, m2: _Monomial) -> _Monomial:
    exps: dict[Hashable, int] = defaultdict(int)
    for g, e in m1:
        exps[g] += e
    for g, e in m2:
        exps[g] += e
    return tuple(sorted(exps.items(), key=lambda kv: repr(kv[0])))


def _poly_mul(a: _Poly, b: _Poly) -> _Poly:
    out: _Poly = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            mono = _mono_mul(m1, m2)
            out[mono] = out.get(mono, 0) + c1 * c2
            if out[mono] == 0:
                del out[mono]
    return out


def _from_poly(poly: _Poly) -> Dim:
    """Reconstruct a canonical :class:`Dim` from a polynomial (for display/storage).

    Generators are mapped back to their ``Dim`` nodes; equality is still decided via
    :func:`dim_eq`, so the exact surface form is not load-bearing.
    """
    if not poly:
        return DimLit(0)
    sum_terms: list[Dim] = []
    for mono in sorted(poly, key=lambda m: (len(m), repr(m))):
        coef = poly[mono]
        neg = coef < 0
        mag = -coef if neg else coef
        factors: list[Dim] = []
        for gen, exp in mono:
            node = _gen_to_dim(gen)
            factors.extend([node] * exp)
        if not factors:
            term: Dim = DimLit(mag)
        elif mag == 1:
            term = dim_prod(*factors)
        else:
            term = dim_prod(DimLit(mag), *factors)
        sum_terms.append(DimNeg(term) if neg else term)
    if len(sum_terms) == 1:
        return sum_terms[0]
    return dim_sum(*sum_terms)


def _gen_to_dim(gen: Hashable) -> Dim:
    kind = gen[0]  # type: ignore[index]
    if kind == "var":
        return DimVar(gen[1])  # type: ignore[index]
    if kind == "dimof":
        return DimOf(gen[1])  # type: ignore[index]
    if kind == "atom":
        return DimAtomRef(gen[1])  # type: ignore[index]
    if kind == "mod":
        _, canon_t, canon_m = gen  # type: ignore[misc]
        return DimMod(_from_poly(dict(canon_t)), _from_poly(dict(canon_m)))
    raise DimError(f"_gen_to_dim: unknown generator {gen!r}")


def is_runtime_dim(d: object) -> bool:
    """Report whether ``d`` mentions a runtime rank atom, so its value is unknown here.

    A dimension that mentions a ``δ_f`` sizes a dynamic array axis at lowering rather than
    a fixed one, so callers that need a concrete extent ask this first.
    """
    if not is_dim(d):
        return False
    match d:
        case DimAtomRef():
            return True
        case DimSum() | DimProd():
            return any(is_runtime_dim(t) for t in d.terms)
        case DimNeg():
            return is_runtime_dim(d.term)
        case DimMod():
            return is_runtime_dim(d.term) or is_runtime_dim(d.modulus)
        case DimLit() | DimVar() | DimOf():
            # No runtime atom: literals and static-opaque generators. A future Dim node
            # that CAN carry one must be classified here explicitly (assert_never below).
            return False
        case _ as unreachable:
            assert_never(unreachable)


def space_mentions_runtime_dim(s: object) -> bool:
    """Report whether ``|s|`` mentions a runtime rank atom.

    A space whose dimension cannot be computed at all answers ``False``: the question is
    whether the size is runtime-determined, not whether it is presently known.
    """
    if not is_space(s):
        return False
    try:
        return is_runtime_dim(dim_of(s))  # type: ignore[arg-type]
    except DimError:
        return False


__all__ = [
    "DimError",
    "dim_of",
    "rank_of",
    "dim_diff",
    "dim_eq",
    "is_runtime_dim",
    "space_mentions_runtime_dim",
]
