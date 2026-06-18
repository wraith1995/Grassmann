"""Symbolic dimension reasoning (`25 §1`).

A ``Dim`` (from :mod:`grassmann.core.space`) is a Nat-expression over

- static params (``DimLit``, ``DimVar``) and ``DimOf`` of opaque spaces, and
- **runtime dimension atoms** ``δ_f`` (``DimAtomRef``) — opaque Nats the SVD deduces
  at lowering, keyed by a term ``f``.

This module computes ``|s|`` for a Space, resolves ``rank(f)`` (asserted ⇒ static;
otherwise a runtime atom ``δ_f``), and decides dim equality ``d ≡ d'`` by normalizing
to a canonical polynomial over the opaque generators (``DimVar``/``DimOf``/``DimAtomRef``
treated as atoms; ``δ_f ≡ δ_g`` iff ``f ≈ g`` structurally).

Dims are **compared, never unified** (`25 §F5`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import assert_never

from grassmann.core.space import (
    Base,
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
    SVar,
    WithInner,
    dim_prod,
    dim_sum,
    is_space,
)


class DimError(Exception):
    """Raised when a dimension cannot be computed (e.g. an unsupported Space)."""


# ===========================================================================
# |s| — the dimension of a space (`25 §1`).
# ===========================================================================


def dim_of(s: Space) -> Dim:
    """Compute ``|s|`` as a symbolic :class:`Dim` (`25 §1`).

    FFS dims (``Ker``/``Img``/``CoKer``/``CoImg``) require knowing the source/target
    spaces and the map's rank; rank is resolved via :func:`rank_of`.
    """
    match s:
        case Scalar():
            return DimLit(1)
        case Base():
            return s.dim
        case SVar():
            # |SVar α| = δα = dim(α) — an opaque static Nat keyed by the space var.
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
        case _:
            raise DimError(f"dim_of: unsupported space {s!r}")


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
# rank(f) — asserted (static) OR a runtime atom δ_f (`25 §1`/§7).
# ===========================================================================


def rank_of(f: object) -> Dim:
    """Resolve ``rank(f)`` per `25 §1`/§7.

    - asserted props on ``f`` (a ``Var``): ``bij``/``inj``/``surj``/explicit ``rank``;
    - otherwise a runtime rank atom ``δ_f`` (:class:`DimAtomRef` keyed by ``f``).
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
# Dim arithmetic + normalized equality (`25 §1`: d ≡ d').
# ===========================================================================


def dim_diff(a: Dim, b: Dim) -> Dim:
    """Symbolic ``a − b``.

    Represented internally via the normalized polynomial; if the result has only
    non-negative integer monomials it stays as a sum, otherwise it carries negative
    coefficients (legal as an intermediate, e.g. ``|dom| − rank``). The core ``Dim``
    algebra has no subtraction node, so we encode ``a − b`` as a normalized poly and
    reconstruct a ``Dim`` from it; this round-trips through :func:`dim_eq`.
    """
    poly = _poly_sub(_to_poly(a), _to_poly(b))
    return _from_poly(poly)


def dim_eq(a: object, b: object) -> bool:
    """``a ≡ b`` — equal normalized Nat polynomials (`25 §1`).

    Opaque generators (``DimVar``, ``DimOf``, ``DimAtomRef``) are treated as
    indeterminates; ``δ_f ≡ δ_g`` iff ``f ≈ g`` (structural term equality, captured by
    frozen-dataclass ``==`` on the keying term).
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
    """A hashable key identifying an opaque generator (so δ_f ≡ δ_g iff f≈g)."""
    match d:
        case DimVar():
            return ("var", d.name)
        case DimOf():
            # Spaces are frozen dataclasses ⇒ hashable & structurally compared.
            return ("dimof", d.space)
        case DimAtomRef():
            return ("atom", d.key)
        case _:
            raise DimError(f"_gen_key: not a generator {d!r}")


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
    """A hashable, order-independent canonical form of a polynomial (for keying)."""
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
    """True iff ``d`` mentions a runtime rank atom ``δ_f`` (`25 §1` F3 seam)."""
    match d:
        case DimAtomRef():
            return True
        case DimSum() | DimProd():
            return any(is_runtime_dim(t) for t in d.terms)
        case DimNeg():
            return is_runtime_dim(d.term)
        case DimMod():
            return is_runtime_dim(d.term) or is_runtime_dim(d.modulus)
        case _:
            return False


def space_mentions_runtime_dim(s: object) -> bool:
    """True iff ``|s|`` would mention a runtime atom (used by tests for the F3 seam)."""
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
