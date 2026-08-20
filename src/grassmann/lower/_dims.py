"""Static-dimension resolution for spaces and bases.

A small helper shared by :mod:`space_basis` and :mod:`cob`: turn a structural
:class:`~grassmann.core.space.Space` (or :class:`~grassmann.core.basis.Basis`) into a
concrete ``int`` dimension *when statically known*, raising :class:`DynamicDim` when the
size is a runtime rank ``δ_f`` (a :class:`~grassmann.core.space.DimAtomRef`) or otherwise
symbolic. The substrate emits polyarray :class:`~polyarray.DimAtom`-sized (dynamic) axes
in those cases, so callers catch :class:`DynamicDim` and route to the bulk/dynamic lane.

This is *structural* dimension arithmetic only (the full dim-inference machinery does the
real inference); it suffices for the substrate's static numeric tests and for sizing the
``OutSpec``s of the FFS realizations.
"""

from __future__ import annotations

from typing import assert_never

from grassmann.core import basis as B
from grassmann.core import space as S


class DynamicDim(Exception):
    """Raised when a space/basis dimension is not a static ``int`` (a runtime ``δ_f``)."""


def dim_of(space: S.Space) -> int:
    """Return the static integer dimension of ``space`` or raise :class:`DynamicDim`."""
    match space:
        case S.Scalar():
            return 1
        case S.Dual(s):
            return dim_of(s)
        case S.OPlus(spaces):
            return sum(dim_of(s) for s in spaces)
        case S.OProd(spaces):
            out = 1
            for s in spaces:
                out *= dim_of(s)
            return out
        case S.OPlusRep(count, s):
            return _dim_int(count) * dim_of(s)
        case S.WithInner(s, _):
            return dim_of(s)
        case S.Img() | S.CoImg() | S.Ker() | S.CoKer():
            # FFS dims are runtime ranks UNLESS the defining map's asserted props
            # (inj/surj/bij, rank/dim_img/dim_ker) pin a static rank; when they
            # don't, the substrate sizes these with a polyarray DimAtom (dynamic axis).
            return _ffs_dim(space)
        case S.SVar() | S.App():
            raise DynamicDim(f"space {space!r} has no static dimension")
        case S.SpaceMeta():
            raise DynamicDim(f"space {space!r} has no static dimension (unsolved metavar)")
        case _ as unreachable:
            assert_never(unreachable)


def _ffs_dim(space: S.Img | S.Ker | S.CoImg | S.CoKer) -> int:
    """Give the static dim of an FFS space when the map's props pin its rank, else raise.

    Dimension laws: ``|Img| = |CoImg| = rank``, ``|Ker| = |dom| − rank``,
    ``|CoKer| = |cod| − rank``.  The rank is static iff the defining map carries an
    asserted ``rank``/``dim_img`` (or ``dim_ker``), or is ``inj`` (``rank = |dom|``),
    ``surj`` (``rank = |cod|``), or ``bij`` (``rank = |dom| = |cod|``).
    """
    dom, cod = _ffs_endpoints(space)
    rank = _static_rank(space, dom, cod)
    match space:
        case S.Img() | S.CoImg():
            return rank
        case S.Ker():
            return dim_of(dom) - rank
        case S.CoKer():
            return dim_of(cod) - rank
        case _ as unreachable:
            assert_never(unreachable)


def _ffs_endpoints(space: S.Space) -> tuple[S.Space, S.Space]:
    """Give the (domain, codomain) spaces of an FFS map, from ``source``/``target`` or ``f``.

    Prefers the node's explicit ``source``/``target``; falls back to reading the map
    var's space ``~dom ⊗ cod`` (the ``OProd(Dual(dom), cod)`` linear-map encoding).
    """
    src = getattr(space, "source", None)
    tgt = getattr(space, "target", None)
    if src is not None and tgt is not None:
        return src, tgt
    f = getattr(space, "f", None)
    fs = getattr(f, "space", None)
    from grassmann.infer.subspace import split_map_space
    split = split_map_space(fs) if S.is_space(fs) else None
    if split is not None:
        dom = src if src is not None else split[0]
        cod = tgt if tgt is not None else split[1]
        return dom, cod
    raise DynamicDim(f"FFS space {space!r} has no resolvable source/target")


def _static_rank(space: S.Space, dom: S.Space, cod: S.Space) -> int:
    """Give the map's static rank from its asserted props, or raise :class:`DynamicDim`."""
    props = getattr(getattr(space, "f", None), "props", None)
    if props is not None:
        if props.rank is not None:
            return _dim_int(props.rank)
        if props.dim_img is not None:
            return _dim_int(props.dim_img)
        if props.dim_ker is not None:
            return dim_of(dom) - _dim_int(props.dim_ker)
        if props.bij:
            return dim_of(dom)  # = dim_of(cod) for a bijection
        if props.inj:
            return dim_of(dom)
        if props.surj:
            return dim_of(cod)
    raise DynamicDim(f"FFS space {space!r} has a runtime (δ_f) rank (no asserted props)")


def _dim_int(d: S.Dim) -> int:
    match d:
        case S.DimLit(v):
            return v
        case S.DimSum(terms):
            return sum(_dim_int(t) for t in terms)
        case S.DimProd(terms):
            out = 1
            for t in terms:
                out *= _dim_int(t)
            return out
        case S.DimMod(term, modulus):
            # Python-style non-negative remainder; only foldable when both static.
            m = _dim_int(modulus)
            if m == 0:
                raise DynamicDim(f"dim {d!r} has a zero modulus")
            return _dim_int(term) % m
        case S.DimOf(s):
            return dim_of(s)
        case S.DimNeg(term):
            # ``-d`` (appears inside a DimSum for the FFS subtraction laws).
            return -_dim_int(term)
        case S.DimVar() | S.DimAtomRef():
            raise DynamicDim(f"dim {d!r} is symbolic / runtime")
        case _ as unreachable:
            assert_never(unreachable)


def _list_basis_ambient(vectors: tuple) -> S.Space | None:
    """Give the (ambient) space of a ``ListBasis``.

    Its vectors' ``synth_space``, or ``None`` when undeterminable (abstract vectors).  A ``ListBasis`` is
    always a full basis of this space.
    """
    if not vectors:
        return None
    try:
        from grassmann.infer.synth import synth_space

        return synth_space(vectors[0])
    except Exception:
        return None


def basis_dim(b: B.Basis) -> int:
    """Return the static integer dimension of a basis or raise :class:`DynamicDim`."""
    match b:
        case B.ScalarBasis():
            return 1
        case (B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _)
              | B.OrthonormalBasis(inner)):
            return basis_dim(inner)
        case B.BasisSum(bases):
            return sum(basis_dim(x) for x in bases)
        case B.BasisProd(bases):
            out = 1
            for x in bases:
                out *= basis_dim(x)
            return out
        case B.BasisRep(count, inner):
            return _dim_int(count) * basis_dim(inner)
        case B.ListBasis(vectors, backup, reduce):
            # A ``ListBasis`` is ALWAYS a full basis of its AMBIENT (the vectors' space): its
            # dim is the ambient dim (static when the ambient is static).  An explicit backup
            # also gives the ambient.  Only when the ambient is genuinely undeterminable
            # (abstract vectors, no backup) do we fall back to a runtime/length size.
            if backup is not None:
                return basis_dim(backup)
            amb = _list_basis_ambient(vectors)
            if amb is not None:
                return dim_of(amb)
            if reduce is B.Reduce.RREF:
                raise DynamicDim(
                    f"ListBasis(reduce=RREF, backup=None) {b!r} is sized by its runtime rank"
                )
            return len(vectors)
        case B.CompleteBasis(sub_basis, complement, reduce):
            # RREF/ORTHONORMALIZE complete to the ambient (the complement's dimension); else
            # the stacked count sub + complement.
            if reduce is B.Reduce.NONE:
                return basis_dim(sub_basis) + basis_dim(complement)
            return basis_dim(complement)
        case B.ImageBasis() | B.ProjectedBasis() | B.BVar() | B.BLam() | B.BApp():
            raise DynamicDim(f"basis {b!r} has a runtime/symbolic dimension")
        case _ as unreachable:
            assert_never(unreachable)
