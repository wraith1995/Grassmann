"""The dimension of a space or basis, when it is a number at build time.

Lowering has to size arrays, and most spaces have a dimension it can compute outright by
recursing over their structure. Some do not: the dimension of ``Img(f)`` is the rank of
``f``, which is generally not known until the program runs, and a space variable has no
dimension at all until it is instantiated. This module draws that line::

    ⊕(V₂, V₃)               ⇒  5
    Dual(V₃) ⊗ V₄           ⇒  12
    Img(f), f asserted inj  ⇒  |dom f|, since injectivity pins the rank
    Img(f), rank unasserted ⇒  DynamicDim — the rank δ_f is a runtime value
    SVar α                  ⇒  DynamicDim — no dimension until instantiated

A caller catches :class:`DynamicDim` and sizes that axis with a polyarray
:class:`~polyarray.DimAtom` instead, which resolves at run time.

The arithmetic here is structural rather than inferential — the dimension inference proper
lives in :mod:`grassmann.infer` — and it is enough to size the outputs of the
subspace realizations and to decide the static cases the lowering substrate tests for.
"""

from __future__ import annotations

from typing import assert_never

from grassmann.core import basis as B
from grassmann.core import space as S


class DynamicDim(Exception):
    """Raised when a dimension is not a build-time integer.

    A caller treats this as an instruction rather than a failure: size the axis with a runtime
    :class:`~polyarray.DimAtom` and let it resolve when the program runs.
    """


def dim_of(space: S.Space) -> int:
    """Give the dimension of ``space`` as an integer, or raise :class:`DynamicDim`.

    The structural constructors compose as one expects: a direct sum adds, a tensor product
    multiplies, and ``Dual`` and ``WithInner`` leave the dimension alone. A fundamental
    subspace defers to :func:`_ffs_dim`, which can answer only when the defining map's
    asserted properties pin its rank. A space variable and an unreduced application have no
    dimension to give, and neither has an unsolved inference metavariable.
    """
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
            # FFS dims are runtime ranks unless the defining map's asserted props
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
    """Give a fundamental subspace's dimension when the map's properties pin its rank.

    All four dimensions follow from the rank by the rank-nullity laws::

        |Img| = |CoImg| = rank
        |Ker|  = |dom| − rank
        |CoKer| = |cod| − rank

    So the question is whether the rank itself is known at build time, and it is exactly when
    the defining map carries an asserted ``rank``, ``dim_img`` or ``dim_ker``, or is asserted
    injective (rank ``|dom|``), surjective (rank ``|cod|``) or bijective (both, and they
    agree). With none of those the rank is the runtime value ``δ_f`` and this raises.
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
    """Give the domain and codomain of a subspace node's defining map.

    A node that declares its endpoints explicitly is believed. Otherwise the endpoints are read
    off the map's own space, since a linear map ``dom ⊸ cod`` is encoded as
    ``OProd(Dual(dom), cod)`` and the two factors are the endpoints. A node offering neither
    raises :class:`DynamicDim`.
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
    """Give the defining map's rank from its asserted properties, or raise :class:`DynamicDim`.

    An explicit ``rank`` or ``dim_img`` gives it directly, ``dim_ker`` gives it by subtraction
    from the domain, and injectivity, surjectivity or bijectivity each determine it from an
    endpoint dimension. An unasserted map has only the runtime rank ``δ_f``.
    """
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
    """Fold a symbolic dimension expression to an integer, or raise :class:`DynamicDim`.

    Sums, products, negations and modulos fold when their operands do; ``DimOf`` defers to
    :func:`dim_of` for the space it names. A dimension variable and a runtime rank atom are
    exactly the two that cannot fold.
    """
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
    """Give the ambient space a ``ListBasis``'s vectors live in, or ``None``.

    It is the space the first vector synthesizes to, since a ``ListBasis`` is always a full
    basis of the space its vectors inhabit. Abstract vectors synthesize to nothing, and the
    answer is then ``None``.
    """
    if not vectors:
        return None
    try:
        from grassmann.infer.synth import synth_space

        return synth_space(vectors[0])
    except Exception:
        return None


def basis_dim(b: B.Basis) -> int:
    """Give the dimension of a basis as an integer, or raise :class:`DynamicDim`.

    The structural constructors mirror the spaces they coordinatize, so a sum adds, a product
    multiplies, and the variance and metric wrappers pass through. Two cases need more care. A
    ``ListBasis`` is a full basis of its ambient space, so its dimension is the ambient one —
    taken from an explicit backup basis if it has one, else from the vectors' own space — and
    only a basis whose ambient cannot be determined falls back to the vector count, or, when it
    is row-reduced, is sized by a runtime rank. A ``CompleteBasis`` that reduces spans the
    ambient, so its dimension is the complement's; one that does not is the two stacked.

    An image or projected basis has the dimension of the subspace it coordinatizes, which is a
    runtime rank here, and an abstract basis has no dimension until it is reduced.
    """
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
            # A ``ListBasis`` is always a full basis of its ambient (the vectors' space): its
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
