"""What basis a space gets, and how a subspace basis becomes actual columns.

Two things live here. The first is the map from a space to its bases: its default basis,
and the alternatives :mod:`grassmann.lower.choose` scores against it. The second is the
realization of a fundamental-subspace basis — the point at which ``Img(f)`` stops being a
space and becomes a concrete matrix of columns, produced by a factorization of ``f``.

The default is the standard basis, orthonormal with respect to the space's own inner
product, so that the metric is the identity in coordinates and ``Inner``, ``Dagger`` and
the solves all lower to numeric operations. It is built by mirroring the space tower onto
the basis tower, one constructor at a time::

    Scalar          ⇒  ScalarBasis
    Dual s          ⇒  DualBasis(default s)
    ⊕(sᵢ)           ⇒  BasisSum(default sᵢ)
    ⊗(sᵢ)           ⇒  BasisProd(default sᵢ)
    ⊕ⁿ s            ⇒  BasisRep(n, default s)
    WithInner s g   ⇒  WithInnerBasis(default s, g), g-orthonormal
    Img/Ker/
    CoImg/CoKer f   ⇒  ProjectedBasis(f, which)
    SVar α          ⇒  BVar(β), the basis supplied at instantiation

The candidates add the alternatives worth scoring: the literal ``ImageBasis(f)`` of an
image, the endpoint bases of a subspace's defining map, and for a metric space the plain
reuse of the underlying space's basis.

Realizing a subspace basis is the seam where grassmann meets a matrix factorization, and
which factorization runs depends on what is known about the map::

    injective, no metric  ⇒  QrOp    Q is the orthonormal range basis; rank is static
    a metric is present   ⇒  GSvdOp  orthonormal in the metrics rather than the standard
                                     inner product
    otherwise             ⇒  SvdOp   the rank δ_f is a runtime value, so the subspace
                                     axis is a dynamic polyarray.DimAtom

:func:`realize_ffs_basis` is that seam, and an operator whose value is a build-time
constant is folded to constant columns before any of it is emitted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_never

from dataclasses import dataclass

import numpy as np
import polyarray as pa

from grassmann.core import basis as B
from grassmann.core import props as P
from grassmann.core import space as S
from grassmann.core import term as T

__all__ = [
    "space_basis",
    "default_basis",
    "candidate_bases",
    "FfsRealization",
    "realize_ffs_basis",
    "realize_image_application",
    "try_constant_ffs",
]


# ---------------------------------------------------------------------------
# Default (inner-product-orthonormal) basis.
# ---------------------------------------------------------------------------


def default_basis(space: S.Space) -> B.Basis:
    """Give the standard basis of ``space``: the one orthonormal in its own inner product.

    The construction mirrors the space tower onto the basis tower, recursing through the
    structural constructors and bottoming out at ``Scalar``, at a fundamental subspace, or at a
    space variable. It is total over ``Space`` except for the two nodes that have no basis to
    give: an unreduced ``App`` and an unsolved inference metavariable both raise
    :class:`~grassmann.core.space.SpaceError`.
    """
    match space:
        case S.Scalar():
            return B.scalar_basis()
        case S.Dual(s):
            return B.dual_basis(default_basis(s))
        case S.OPlus(spaces):
            return B.basis_sum(*(default_basis(s) for s in spaces))
        case S.OProd(spaces):
            return B.basis_prod(*(default_basis(s) for s in spaces))
        case S.OPlusRep(count, s):
            return B.basis_rep(count, default_basis(s))
        case S.WithInner(s, g):
            # default = g-orthonormal (metric becomes identity in these coords).
            return B.with_inner_basis(default_basis(s), g)
        case S.Img(f, src, tgt):
            # The natural basis of the range is the ORTHONORMAL (Q/SVD-U) range basis —
            # ``ProjectedBasis(f, "image")`` (the immersed side).  ``ImageBasis(f)`` is the
            # separate LITERAL map-application basis (raw ``f(eᵢ)`` columns), offered only
            # as a candidate.
            return B.projected_basis(f, "image", _maybe(src), _maybe(tgt))
        case S.Ker(f, src, tgt):
            return B.projected_basis(f, "ker", _maybe(src), _maybe(tgt))
        case S.CoImg(f, src, tgt):
            return B.projected_basis(f, "coimg", _maybe(src), _maybe(tgt))
        case S.CoKer(f, src, tgt):
            return B.projected_basis(f, "coker", _maybe(src), _maybe(tgt))
        case S.SVar(name):
            return B.bvar("β_" + name)
        case S.App():
            raise S.SpaceError(
                f"space_basis: cannot give a default basis for an unreduced App {space!r}"
            )
        case S.SpaceMeta():
            raise S.SpaceError(f"space_basis: no default basis for an unsolved metavar {space!r}")
        case _ as unreachable:
            assert_never(unreachable)


def map_basis(space: S.Space, choice: Callable[[S.Space], B.Basis | None]) -> B.Basis:
    """Build a basis for ``space`` from a caller's leaf policy, mirroring the space tower.

    ``choice`` is consulted at every node on the way down, and returning a basis there overrides
    the whole subtree; returning ``None`` declines and lets the walk continue. An un-overridden
    structural wrapper — ``Dual``, ``⊕``, ``⊗``, ``⊕ⁿ``, ``WithInner`` — mirrors into the
    corresponding basis constructor and recurses, and every other un-overridden node takes
    :func:`default_basis`::

        choice(V ⊗ Dual(Img f)) → None      recurse
          choice(V)             → None      default_basis(V)
          choice(Dual(Img f))   → None      recurse
            choice(Img f)       → a pin     the caller's basis, used as given

    A consumer that wants a particular basis at one kind of leaf — a pinned image basis for an
    immersed tangent space, say — supplies only that rule and gets the standard basis
    everywhere else, rather than writing the tower walk again.
    """
    picked = choice(space)
    if picked is not None:
        return picked
    match space:
        case S.Dual(s):
            return B.dual_basis(map_basis(s, choice))
        case S.OPlus(spaces):
            return B.basis_sum(*(map_basis(s, choice) for s in spaces))
        case S.OProd(spaces):
            return B.basis_prod(*(map_basis(s, choice) for s in spaces))
        case S.OPlusRep(count, s):
            return B.basis_rep(count, map_basis(s, choice))
        case S.WithInner(s, g):
            return B.with_inner_basis(map_basis(s, choice), g)
        case S.Img() | S.Ker() | S.CoImg() | S.CoKer() | S.Scalar() | S.SVar() | S.App() | S.SpaceMeta():
            # FFS nodes are LEAVES for the tower walk (their internals are the map's business);
            # Scalar/SVar take the standard default; App/SpaceMeta raise there, loudly.
            return default_basis(space)
        case _ as unreachable:
            assert_never(unreachable)


def map_space(
    space: S.Space,
    choice: Callable[[S.Space], S.Space | None],
    *,
    map_choice: Callable[[T.Term], T.Term | None] | None = None,
) -> S.Space:
    """Rewrite ``space`` from a caller's leaf policy, mirroring the space tower.

    This is the ``Space -> Space`` sibling of :func:`map_basis` and it works the same way:
    ``choice`` is consulted at every node, a returned space overrides that subtree, and ``None``
    declines. An un-overridden structural wrapper mirrors and recurses; every other
    un-overridden node comes back unchanged.

    ``map_choice`` applies to the *defining map* of a fundamental subspace, so a consumer can
    rewrite ``Img(f)`` by rewriting ``f``. That matters because it keeps the consumer from
    matching over ``Space`` itself, which would be a partial ladder over a closed sum type, and
    because a subspace often hides another one inside its map: ``Λᵏ(Img J)`` is ``Img(alt)``,
    and the ``Img(J)`` it is built over sits in ``alt``'s endpoints rather than anywhere in the
    outer space.

    Swapping one immersion for another inside a value space is what this is for. Reading a
    form-valued degree of freedom against a permuted chart is the same space with ``Img(J)``
    replaced by ``Img(J')`` wherever it occurs, however deep it sits under ``Dual``, ``⊗`` or
    ``⊕``, and inside the maps that define its subspaces.
    """
    picked = choice(space)
    if picked is not None:
        return picked
    mc = map_choice
    match space:
        case S.Dual(s):
            return S.dual(map_space(s, choice, map_choice=mc))
        case S.OPlus(spaces):
            return S.OPlus(tuple(map_space(s, choice, map_choice=mc) for s in spaces))
        case S.OProd(spaces):
            return S.OProd(tuple(map_space(s, choice, map_choice=mc) for s in spaces))
        case S.OPlusRep(count, s):
            return S.OPlusRep(count, map_space(s, choice, map_choice=mc))
        case S.WithInner(s, g):
            return S.WithInner(map_space(s, choice, map_choice=mc), g)
        case S.Img(f, src, tgt):
            return S.Img(_map_f(f, map_choice), _map_opt(src, choice, map_choice),
                         _map_opt(tgt, choice, map_choice))
        case S.Ker(f, src, tgt):
            return S.Ker(_map_f(f, map_choice), _map_opt(src, choice, map_choice),
                         _map_opt(tgt, choice, map_choice))
        case S.CoImg(f, src, tgt):
            return S.CoImg(_map_f(f, map_choice), _map_opt(src, choice, map_choice),
                         _map_opt(tgt, choice, map_choice))
        case S.CoKer(f, src, tgt):
            return S.CoKer(_map_f(f, map_choice), _map_opt(src, choice, map_choice),
                         _map_opt(tgt, choice, map_choice))
        case S.Scalar() | S.SVar() | S.App() | S.SpaceMeta():
            return space                                  # true leaves
        case _ as unreachable:
            assert_never(unreachable)


def _map_f(f: T.Term, map_choice: Callable[[T.Term], T.Term | None] | None) -> T.Term:
    """Give a subspace's defining map after ``map_choice``, or unchanged when it declines.

    An absent ``map_choice``, or one returning ``None``, leaves the map as it stands.
    """
    if map_choice is None:
        return f
    out = map_choice(f)
    return f if out is None else out


def _map_opt(
    s: S.Space | None,
    choice: Callable[[S.Space], S.Space | None],
    map_choice: Callable[[T.Term], T.Term | None] | None = None,
) -> S.Space | None:
    """Run :func:`map_space` on a subspace node's optional endpoint space.

    A fundamental subspace is not a leaf of the walk, because an immersion buried inside one
    reaches the tower only through the ``source`` and ``target`` of the map that defines it.
    ``Λᵏ(Img J)`` is ``Img(alt)``, and its ``Img(J)`` lives in ``alt``'s endpoints; a walk that
    stopped at ``Img`` would find nothing to substitute there and return the space unchanged,
    so a rewrite meant to transport a value between two charts would silently become the
    identity. ``_immersed_basis`` and ``_immersed_endpoints`` recurse the same way on the basis
    side; this is its counterpart over spaces.
    """
    return None if s is None else map_space(s, choice, map_choice=map_choice)


def _maybe(s: S.Space | None) -> B.Basis | None:
    """Give the default basis of an optional endpoint space, or ``None`` when it is absent.

    A subspace node may or may not declare the endpoints of its defining map, and an undeclared
    endpoint contributes no basis to the ``ProjectedBasis`` built over it.
    """
    return None if s is None else default_basis(s)


# ---------------------------------------------------------------------------
# Candidate bases (fed to the basis choice).
# ---------------------------------------------------------------------------


def candidate_bases(space: S.Space) -> tuple[B.Basis, ...]:
    """Give the bases of ``space`` worth scoring against a term, the default among them.

    The default basis is always first, so a caller that scores these can never come away with
    nothing. What is added to it depends on the space: the image of a map offers the literal
    ``ImageBasis(f)`` and, when the map declares one, the codomain's standard basis; a kernel or
    coimage offers the domain's; a cokernel the codomain's; and a metric space offers the plain
    basis of the space underneath, which leaves the metric symbolic but needs no whitening cob.
    """
    out: list[B.Basis] = [default_basis(space)]

    match space:
        case S.WithInner(s, _):
            # the g-orthonormal default is already out[0]; a candidate is to reuse the
            # underlying standard basis (metric stays symbolic, but no whitening cob).
            _add(out, default_basis(s))
        case S.Img(f, src, tgt):
            # default (out[0]) is the orthonormal range ProjectedBasis(f,"image"); the
            # literal map-application ImageBasis(f) and the codomain's standard basis are
            # alternatives the heuristic can score against the term.
            _add(out, B.image_basis(f))
            if tgt is not None:
                _add(out, default_basis(tgt))
        case S.Ker(f, src, tgt) | S.CoImg(f, src, tgt):
            if src is not None:
                _add(out, default_basis(src))
        case S.CoKer(f, src, tgt):
            if tgt is not None:
                _add(out, default_basis(tgt))
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): out[0]=default_basis; extras are optimization [approved 2026-07-03]
            pass
    return tuple(out)


def _add(acc: list[B.Basis], b: B.Basis) -> None:
    """Append ``b`` to the candidate list unless an equal basis is already there.

    Offering the same basis twice would only make the scorer weigh it twice.
    """
    if b not in acc:
        acc.append(b)


def space_basis(space: S.Space) -> tuple[B.Basis, tuple[B.Basis, ...]]:
    """Give ``space``'s default basis together with the candidates worth scoring against it.

    The default is the standard basis, orthonormal in the space's inner product. The candidates
    include it, so :func:`~grassmann.lower.choose.choose` always has something to return.
    """
    return default_basis(space), candidate_bases(space)


# ---------------------------------------------------------------------------
# FFS basis realization — the SVD / QR / GSVD seam.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FfsRealization:
    """One fundamental-subspace basis, realized: its columns and how they were produced.

    ``cols`` holds the basis vectors as the columns of a matrix, written in the ambient
    coordinates — the standard basis of the defining map's source or target, whichever side the
    subspace lives on::

        cols :  ambient × subspace      each column one basis vector of the subspace
        rank :  the numerical rank δ_f, a 0-d output, absent when the rank is static
        rank_dim : the DimAtom sizing the subspace axis, absent when that axis is static

    ``rank`` and ``rank_dim`` are both absent for the injective QR route, where the rank is
    known from the map's asserted properties and the column count is an ordinary integer.
    ``op`` names the operation that produced the columns, which provenance records and which
    :func:`ffs_cols_orthonormal` reads to decide whether the columns are orthonormal.
    """

    cols: pa.SymArray
    rank: pa.SymArray | None
    rank_dim: pa.DimAtom | None
    op: str
    """The operation that produced ``cols``, which is what settles whether they are orthonormal.

    Orthonormality is a property of the factorization, not of the subspace, so a consumer that
    needs the coordinate map *into* the frame branches on :func:`ffs_cols_orthonormal`: the
    transpose ``Pᵀ`` reads coordinates only for an orthonormal frame, and the general case needs
    the matched dual ``pinv(P)``.
    """
    # The upper-triangular ``R`` of the injective QR, ``J = Q·R``, kept because it IS the change
    # of basis between the two bases of the same image: the literal ``ImageBasis`` columns are
    # ``J`` and the orthonormal ``ProjectedBasis`` columns are ``Q``, so converting between them
    # costs one matrix already in hand rather than realizing both and pseudo-inverting.  Absent
    # on the SVD and GSVD routes, which produce no such factor.
    r: pa.SymArray | None = None
    # The same subspace basis written in the declared endpoint basis rather than the ambient
    # canonical one: the covariant realization's ``Q₀``, before the ``cols = T·Q₀`` that exists
    # only to satisfy this dataclass's promise that ``cols`` is in ambient standard coordinates.
    # Only ``represent._realize_image_covariant`` sets it.
    #
    # It is worth carrying because the two factors differ sharply in what they depend on.  For a
    # natural ``f`` — the exterior antisymmetrizer under transported ⊗ᵏ bases, which
    # ``try_constant_ffs`` certifies by folding it to a build-time constant — ``Q₀`` is constant,
    # while ``T`` carries every cell dependence and, at an ``Img(J)`` endpoint, the QR radical
    # ``R`` as well.  A consumer whose operand already lives in ``endpoint_basis`` can use
    # ``endpoint_cols`` directly and skip ``T``, its inverse, and its own operand cob.
    #
    # The two bases span the same subspace and give the same subspace coordinates, because the
    # ambient change of basis is applied consistently to basis and operand alike — but only where
    # the projection is exact.  With a residual, ``pinv`` projects along the ambient orthogonal
    # complement, and that complement differs between the two ambient bases.  A consumer must
    # therefore guard on exactness.  For a Λᵏ trace the residual is identically zero, since a
    # diagonal pullback of an antisymmetric tensor stays antisymmetric under any immersion; for a
    # general operand it is not.
    endpoint_cols: pa.SymArray | None = None
    endpoint_basis: object | None = None      # the `B.Basis` `endpoint_cols` is expressed in


# The realizations whose ``cols`` are orthonormal in the standard inner product, ``PᵀP = I``
# exactly, so that ``Pᵀ`` already is the matched dual ``pinv(P) = (PᵀP)⁻¹Pᵀ`` and a consumer can
# read coordinates with a transpose.  This is a positive list, and anything not on it needs the
# general ``pinv``: ``GSvdOp`` is orthonormal in a metric instead (``UᵀGU = I``), ``MapApply`` and
# ``Gram`` are the literal map columns, and a covariant basis is ``T·Q₀``, orthonormal in the
# target basis's inner product rather than the standard one.
_STD_ORTHONORMAL_FFS_OPS = frozenset({"QrOp", "SvdOp", "const", "const-svd", "supplied"})


def ffs_cols_orthonormal(real: FfsRealization) -> bool:
    """Report whether ``real.cols`` is provably orthonormal in the standard inner product.

    The two answers are not symmetric in their consequences, which is why this is a positive
    list of the operations that provably orthonormalize rather than a test. A ``False`` on an
    orthonormal basis costs a consumer the general ``pinv`` or Gram solve and nothing else; a
    ``True`` on a basis that is not orthonormal makes every coordinate read through it silently
    wrong.
    """
    return real.op in _STD_ORTHONORMAL_FFS_OPS


# Which subspaces are read off the right factor of a decomposition rather than the left.
# An SvdOp yields (U, S, Vh, rank) and a GSvdOp (U, UI, V, VI, S, rank); the image and cokernel
# come from the left factor U, the kernel and coimage from the right factor V.
_SVD_WHICH_TAKES_RIGHT = ("ker", "coimg")


def realize_ffs_basis(
    program: pa.Program,
    which: str,
    f_matrix: pa.SymArray,
    dom_dim: int,
    cod_dim: int,
    *,
    props: P.MapProps | None = None,
    metric_dom: pa.SymArray | None = None,
    metric_cod: pa.SymArray | None = None,
    label: str = "f",
    dim_atom: pa.DimAtom | None = None,
    literal_frame: bool = False,
) -> FfsRealization:
    """Realize one fundamental subspace of a map as concrete basis columns.

    ``which`` names the subspace and is one of ``img``, ``coker``, ``coimg`` or ``ker``, with
    ``image`` — the spelling :class:`~grassmann.core.basis.ProjectedBasis` uses — accepted as
    an alias of ``img``. The statements that compute the columns are emitted into ``program``
    and the result is an :class:`FfsRealization`.

    Which factorization runs depends on what is known about the map::

        injective, image, no metric  ⇒  QrOp    Q is the orthonormal range basis, and the
                                                rank is static at dom_dim
        a metric on either endpoint  ⇒  GSvdOp  orthonormal in those metrics
        otherwise                    ⇒  SvdOp   the rank δ_f is only known at run time and
                                                sizes a dynamic axis

    Before the general SVD route, an operator whose value is a provable build-time constant is
    folded to constant columns outright, so a program never carries a factorization of a matrix
    it already knows.

    ``dim_atom`` lets a caller supply the canonical dimension atom for the subspace axis. A
    map's subspace dimension does not depend on the metric — it is the rank, or the ambient
    dimension less the rank for a kernel or cokernel — so the metric-free SVD, the metric GSVD
    and every value typed by that subspace must size their dynamic axis by the same atom, or
    their einsums will not unify. Left ``None``, the seam mints a fresh atom and reports it in
    :attr:`FfsRealization.rank_dim` for the caller to record and pass back.

    ``literal_frame`` applies to the injective-image route alone, where it realizes the map's
    literal columns instead of the orthonormal ``Q``. That is the same subspace in a different
    basis; :func:`_realize_via_gram` gives the precondition and the price in conditioning.
    """
    if which == "image":
        which = "img"
    if which not in ("img", "coker", "coimg", "ker"):
        raise ValueError(f"realize_ffs_basis: bad which {which!r}")

    injective = props is not None and props.inj
    has_metric = metric_dom is not None or metric_cod is not None

    if which == "img" and injective and not has_metric:
        if literal_frame:
            return _realize_via_gram(program, f_matrix, dom_dim, label)
        return _realize_via_qr(program, f_matrix, dom_dim, label)
    if has_metric:
        return _realize_via_gsvd(
            program, which, f_matrix, dom_dim, cod_dim,
            metric_dom, metric_cod, label, dim_atom=dim_atom,
        )
    # A constant fold argued from dependency rather than sampling: when `f_matrix`'s cone is a
    # closed constant subprogram — no live input ref, only `stmt_out` generators, see
    # `_cone_is_structurally_constant` — its plain-SVD basis is a provable build-time constant,
    # so the columns are emitted directly instead of deferring an SvdOp.  Deferring is what hurts:
    # the downstream `fold_numeric` has to work through a dynamic rank-δ axis, and a high-degree
    # integrand repeats that hundreds of times over a handful of distinct constant operators (a
    # reference tangent-basis identity, a Λᵏ antisymmetrizer) until codegen runs out of memory.
    #
    # Only the plain-SVD route folds.  The metric GSVD and the sign-fixed injective QR
    # orthonormalize differently, so a euclidean-SVD constant would be the wrong basis for them.
    # The folded columns are what `_realize_via_svd` would produce for this same `M`.  The
    # sampling `try_constant_ffs` covers the case this check deliberately calls non-constant: an
    # operator whose cone does read live generators but whose value is invariant because they
    # cancel.
    if _cone_is_structurally_constant(program, f_matrix):
        return _fold_constant_ffs(program, which, f_matrix)
    return _realize_via_svd(
        program, which, f_matrix, dom_dim, cod_dim, label, dim_atom=dim_atom,
    )


def try_constant_ffs(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
    dom_dim: int, cod_dim: int, *, probes: int = 3, atol: float = 1e-7,
) -> FfsRealization | None:
    """Give the ``which`` subspace basis as build-time constant columns, or ``None``.

    This applies where ``f_matrix``'s value does not depend on the program's inputs at all, and
    it establishes that by probing: the dependency cone of ``f_matrix`` is evaluated at
    ``probes`` generic feeds and the results compared. Only the cone runs, so this stays safe on
    a large, partially built, shared program — an unrelated singular operation or a dynamic
    input outside the cone is never touched. When the value holds still, its SVD is taken once
    at build and the range or complement basis read off it as constants.

    An input sized by a runtime dimension is bound with a size-1 placeholder along its dynamic
    axes. Every declared input needs a value for the binder to accept the feed, but one outside
    the cone is never read, so the placeholder only has to exist — and a value typed by a
    subspace, sized by a runtime rank, does sit outside the cone of a purely geometric
    factorization.

    Invariance under probing is weaker evidence than
    :func:`_cone_is_structurally_constant`'s dependency argument, and it reaches a case that one
    cannot: an operator invariant by cancellation, whose cone genuinely reads live generators
    that cancel out.
    """
    import numpy as np

    from polyarray import evaluate_cone
    try:
        mats: list[np.ndarray] = []
        for s in range(probes):
            rng = np.random.default_rng(s)
            binding: dict[str, np.ndarray] = {}
            for inp in program.inputs:
                shape = tuple(d if isinstance(d, int) else 1 for d in inp.shape)  # dynamic axis ⇒ placeholder 1
                binding[inp.name] = rng.standard_normal(shape)
            mats.append(np.asarray(evaluate_cone(program, f_matrix, binding), dtype=float))
        m0 = mats[0]
        if m0.ndim != 2 or not all(np.allclose(m0, m, atol=atol) for m in mats[1:]):
            return None                                          # not feed-invariant
        u, sv, vh = np.linalg.svd(m0, full_matrices=True)
        r = int((sv > 1e-9 * max(1.0, float(sv.max()) if sv.size else 1.0)).sum())
        return _const_ffs_from_svd(u, vh, r, which, op="const-svd")
    except Exception:  # noqa: BLE001 — best-effort; the dynamic SVD is the fallback
        return None


# ---------------------------------------------------------------------------
# Folding a structurally-constant realization to constant columns, by dependency rather than by
# sampling.  Both constant routes — the sampling ``try_constant_ffs`` above and the structural
# fold below — go through one shared core, :func:`_const_ffs_from_svd`, which turns a numeric
# matrix's SVD into the constant basis for a given subspace, so the two agree column for column.
# ---------------------------------------------------------------------------


def _canonical_subspace_basis(q: np.ndarray) -> np.ndarray:
    """Build the canonical orthonormal basis of ``span(q)``.

    The result is a function of the subspace and the ambient index order alone, and not of the
    factorization that produced ``q``. That distinction is the whole point of this function.

    A slice ``U[:, :r]`` of an SVD determines the subspace but not a basis within it: inside a
    degenerate singular subspace the columns are fixed only up to a rotation, which makes them a
    discontinuous function of the operator. The exterior antisymmetrizer is exactly such an
    operator — its nonzero singular values are all equal (``σ = 1, 1, 1, 0…`` for a 3-dimensional
    ``Λ²``) and it is invariant only by cancellation, to about 1e-15 rather than bitwise. So a
    rounding difference of 1e-15 rotates the realized basis by O(1), and two ``Space`` nodes
    naming the same subspace by different maps — the images of a canonical and a permuted
    antisymmetrizer for a reoriented facet — end up with unrelated bases. The change of basis
    composed across them is then a rotation where an oriented degree of freedom owes a signed
    permutation. No memo can tie those two together: their keys are legitimately different, and
    a memo cannot span compilations in any case.

    So the basis is not asked of the factorization. The orthogonal projector ``P = q·qᵀ`` is
    unique to the subspace and continuous in it, where ``q`` is neither, and a basis read off
    ``P`` in a fixed ambient order inherits that stability. The construction is a pivoted
    Cholesky of the projector in index order: orthogonalize the columns ``P·eⱼ`` for
    ``j = 0…n−1`` in turn and keep those with a non-negligible residual.

    It fixes its own signs, which is why no sign convention appears here. After orthogonalizing
    against the accepted columns, ``w = P'·eⱼ`` with ``P'`` still an orthogonal projector, so
    ``w[j] = eⱼᵀP'ᵀP'eⱼ = ‖w‖² ≥ 0``: an accepted column's own coordinate is positive by
    construction, and ``‖w‖² = w[j]`` exactly, which is what the acceptance test uses.

    It carries no orientation, and must not. A 3-cycle and a transposition of a facet's vertices
    span the same plane, so ``Λᵏ`` of it is the same subspace, and any rule keyed on the subspace
    owes both the same vector. The orientation sign belongs instead to the ambient chart-to-chart
    conversion in the middle of ``Project ∘ convert ∘ Embed``, where with the two endpoint bases
    equal to ``p`` the composite is ``pᵀ(det M · p) = det M = ±1``. Making the two endpoints agree
    is precisely what turns that into a clean sign.
    """
    n, r = q.shape
    if r <= 0:
        return np.zeros((n, 0))
    p = q @ q.T
    tol = float(np.sqrt(np.finfo(float).eps))              # ‖w‖² = w[j]; accept well above rounding
    cols: list[np.ndarray] = []
    for j in range(n):
        w = np.array(p[:, j], dtype=float)
        for c in cols:                                     # modified Gram–Schmidt, twice for stability
            w -= float(c @ w) * c
        for c in cols:
            w -= float(c @ w) * c
        nrm = float(np.linalg.norm(w))
        if nrm > tol:
            cols.append(w / nrm)
            if len(cols) == r:
                break
    if len(cols) != r:                                     # cannot happen for a rank-r projector
        raise AssertionError(
            f"_canonical_subspace_basis: recovered {len(cols)} of {r} columns from a rank-{r} "
            "projector — the greedy acceptance tolerance is mis-scaled for this operator.")
    return np.ascontiguousarray(np.stack(cols, axis=1))


def _const_ffs_from_svd(
    u: np.ndarray, vh: np.ndarray, r: int, which: str, *, op: str,
) -> FfsRealization:
    """Read the constant columns of subspace ``which`` off a numeric matrix's full SVD.

    Given ``M = U·diag(S)·Vh`` and the numeric rank ``r``, the four subspaces are the same
    slices of the same factors that :func:`_realize_via_svd` takes from the runtime SVD::

        img    first r columns of U            the orthonormal range basis
        coker  last  cod−r columns of U        its orthogonal complement
        coimg  first r columns of V = Vhᵀ
        ker    last  dom−r columns of V

    The slice fixes the subspace but not a basis inside it, so the columns are re-derived
    canonically by :func:`_canonical_subspace_basis` before they are returned; that is what
    makes one subspace have one basis wherever it is realized.

    The resulting columns are constant, and both ``rank`` and ``rank_dim`` are absent because
    the rank is known at build and the axis is an ordinary integer. Both constant routes — the
    sampling :func:`try_constant_ffs` and the structural fold in :func:`realize_ffs_basis` —
    come through here, so they cannot disagree about the basis for a given ``M``.
    """
    if which in ("img", "coker"):
        cols = u[:, :r] if which == "img" else u[:, r:]
    else:                                                        # coimg / ker: right factor V = Vhᵀ
        v = vh.T
        cols = v[:, :r] if which == "coimg" else v[:, r:]
    # The SLICE fixes the SUBSPACE; the slice's own columns do not fix a basis (they are arbitrary
    # inside a degenerate singular subspace). Re-derive the basis canonically from that subspace so
    # one subspace has ONE basis everywhere — see `_canonical_subspace_basis`.
    cols = _canonical_subspace_basis(np.ascontiguousarray(cols))
    return FfsRealization(cols=pa.SymArray(np.ascontiguousarray(cols)),
                          rank=None, rank_dim=None, op=op)


def _cone_is_structurally_constant(program: pa.Program, f_matrix: pa.SymArray) -> bool:
    """Report whether ``f_matrix``'s value is a provable build-time constant.

    The argument is from dependency rather than from sampling: ``f_matrix``'s cone is a closed
    constant subprogram exactly when no statement in it reads a live feed
    (:class:`~polyarray.ir.InputRef` or :class:`~polyarray.ir.IntAtomRef`) and no cell generator
    anywhere in the cone, or in ``f_matrix`` itself, has a provenance kind other than
    ``"stmt_out"``. This delegates to :func:`polyarray.is_structurally_constant`, which is where
    the check belongs, since it uses only polyarray's own dependency cone, ref types and
    environment provenance.

    It is exact in one direction and conservative in the other: it never mistakes an operator
    that varies with the feed for a constant, and any ref or generator it cannot positively
    classify as constant-safe — or a cone it cannot enumerate — makes it answer ``False``,
    leaving the basis to the SVD, QR or GSVD seam.
    """
    return pa.is_structurally_constant(program, f_matrix)


def _fold_constant_ffs(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
) -> FfsRealization:
    """Fold a provably constant ``f_matrix`` to constant columns for subspace ``which``.

    :func:`_cone_is_structurally_constant` is the precondition. Since the cone reads no live
    input, any binding evaluates it to the same matrix, so it is evaluated once at build under a
    zero binding and the numeric operator ``M`` handed to :func:`_const_ffs_from_svd`.

    The basis this produces is the one :func:`_realize_via_svd` would produce for the same
    ``M``: the same full SVD, the same numeric-rank tolerance as :class:`~polyarray.SvdOp`, the
    same column convention. The difference is that it is a constant, and no SVD or rank
    statement enters the program at all.
    """
    from polyarray import evaluate_cone
    binding: dict[str, np.ndarray] = {}
    for inp in program.inputs:
        shape = tuple(d if isinstance(d, int) else 1 for d in inp.shape)  # dynamic axis ⇒ size-1 placeholder
        binding[inp.name] = np.zeros(shape)
    m0 = np.asarray(evaluate_cone(program, f_matrix, binding), dtype=float)
    u, sv, vh = np.linalg.svd(m0, full_matrices=True)
    # Numeric rank with numpy's ``matrix_rank`` rule — IDENTICAL to ``SvdOp``'s default
    # (``rcond=None``) tolerance, so the folded constant equals what the runtime SVD path
    # (``_realize_via_svd``) would slice for this same ``M``.
    tol = float(sv.max()) * max(m0.shape) * float(np.finfo(float).eps) if sv.size else 0.0
    r = int((sv > tol).sum())
    return _const_ffs_from_svd(u, vh, r, which, op="const")


def realize_image_application(
    program: pa.Program, f_matrix: pa.SymArray, dom_dim: int, cod_dim: int, *,
    label: str = "f",
) -> FfsRealization:
    """Realize an :class:`~grassmann.core.basis.ImageBasis`: the literal pushforward columns.

    The columns are ``f`` applied to each source basis vector, ``f(eᵢ)``, which is to say the
    operator matrix itself in its canonical ``(cod, dom)`` orientation. Nothing is factorized
    and nothing is orthonormalized — the columns *are* ``f_matrix``, and this is the basis a
    consumer transports along ``f`` rather than an orthonormal basis for the same subspace.

    The column axis is the static source dimension, one column per source basis vector, and
    neither a rank nor a dimension atom arises since no decomposition is taken.
    """
    [cols] = program.emit_stmt(
        pa.IdentityOp(), [f_matrix],
        [pa.OutSpec("imgapp_" + label, (cod_dim, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):map-application",
    )
    return FfsRealization(cols=cols, rank=None, rank_dim=None, op="MapApply")


def _sb_signs_zero_to_one(low: object, signs: object) -> object:
    """Build the pyab expression ``where(signs == 0, 1.0, signs)``.

    This is the positive-diagonal sign convention's treatment of a zero, mapping it to ``+1``,
    written as an expression rather than as the numpy assignment ``s[s == 0] = 1.0`` that
    :meth:`_QrSignConventionOp.__call__` uses on concrete arrays.
    """
    c = low.core
    zero = c.ArrayExpr(obj=c.FloatLit(value=0.0), dtype=low._dtype_f64())
    one = c.ArrayExpr(obj=c.FloatLit(value=1.0), dtype=low._dtype_f64())
    return c.WhereExpr(cond=c.BinaryExpr(op=c.BinaryOp.EQ, lhs=signs, rhs=zero), x=one, y=signs)


def _realize_via_qr(
    program: pa.Program, f_matrix: pa.SymArray, dom_dim: int, label: str,
) -> FfsRealization:
    """Realize an injective map's image as the orthonormal ``Q`` of its QR factorization.

    Injectivity is what makes this route available: full column rank means ``Q`` has exactly
    ``dom_dim`` orthonormal columns spanning the image, so the rank is static and no runtime
    dimension is needed. The ``R`` factor is kept on the realization, since it is the change of
    basis between this basis and the literal ``ImageBasis`` of the same map.
    """
    # f : (cod, dom); injective ⇒ Q has dom orthonormal columns (the range basis).
    Q, R = program.emit_stmt(
        pa.QrOp(mode="reduced"),
        [f_matrix],
        [pa.OutSpec("Q_" + label, _qr_q_shape(f_matrix, dom_dim)),
         pa.OutSpec("R_" + label, (dom_dim, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):inj=>QrOp",
    )
    # numpy's qr leaves the column signs free, so fix them: with s = sign(diag R), the pair
    # (Q·diag(s), diag(s)·R) is the unique QR whose R has a positive diagonal, and whose Q
    # therefore keeps the orientation of f's own columns.
    Q, R = program.emit_stmt(
        _QrSignConventionOp(),
        [Q, R],
        [pa.OutSpec("Qs_" + label, _qr_q_shape(f_matrix, dom_dim)),
         pa.OutSpec("Rs_" + label, (dom_dim, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):QR-sign(+diag R)",
    )
    return FfsRealization(cols=Q, rank=None, rank_dim=None, op="QrOp", r=R)


def _realize_via_gram(
    program: pa.Program, f_matrix: pa.SymArray, dom_dim: int, label: str,
) -> FfsRealization:
    """Realize an injective map's image as its own literal columns, read through a Gram solve.

    The columns are ``J`` itself, so this is a second realization of the same subspace the QR
    route realizes, in a different basis. It satisfies the same :class:`FfsRealization`
    contract, carries no ``R`` — ``J = cols·I`` — and is tagged so that
    :func:`ffs_cols_orthonormal` reports ``False``. A consumer needing coordinates *in* the
    basis therefore takes its general route, the matched dual ``pinv(J) = (JᵀJ)⁻¹Jᵀ = G⁻¹Jᵀ``,
    which is the Gram solve this realization is named for, rather than a transpose.

    It exists because the bases differ in exactness, not in the subspace they span. QR
    normalizes by ``√(…)``, so :func:`_realize_via_qr`'s ``Q`` is algebraic-irrational in the
    program inputs and polyarray's exact rational lane cannot carry it; anything downstream that
    wanted an exact answer falls back to probing. ``J`` is rational in those same inputs, so the
    exact lane runs through it. The price is conditioning, since coordinates now come from
    ``G = JᵀJ`` and ``κ(G) = κ(J)²`` — which is why the route is opt-in and never the numeric
    default.

    It requires ``f`` injective, the same precondition :func:`_realize_via_qr` already carries,
    so no new assumption enters. Injectivity means full column rank, hence that the columns are
    a basis of the image, hence that ``G`` is invertible and ``pinv(J)·J = I`` identically.
    :func:`realize_ffs_basis` selects this route only under that guard.
    """
    [cols] = program.emit_stmt(
        pa.IdentityOp(), [f_matrix],
        [pa.OutSpec("Jg_" + label, _qr_q_shape(f_matrix, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):inj=>literal-cols(Gram)",
    )
    return FfsRealization(cols=cols, rank=None, rank_dim=None, op="Gram")


@dataclass(frozen=True)
class _QrSignConventionOp:
    """Fix the free column signs of a ``(Q, R)`` factorization by the sign of ``R``'s diagonal.

    With ``s = sign(diag R)`` and a zero read as ``+1``, the pair ``(Q·diag(s), diag(s)·R)`` is
    the factorization of the same matrix whose ``R`` has a positive diagonal — the unique one,
    which is what makes a QR basis reproducible from one realization to the next.

    The same convention is written three times over, for the three ways a program is executed:
    :meth:`__call__` on numpy arrays, :meth:`__numpy_source__` for emitted numpy source, and
    :meth:`__pyab_lower__` for the pyab lowering.
    """

    def __call__(self, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        s = np.sign(np.diag(R))
        s[s == 0] = 1.0
        return Q * s, s[:, None] * R

    def __numpy_source__(self, args: list[str]) -> str:
        """Render this op as numpy source, for ``to_numpy_source`` to find on the op itself.

        Carrying the hook here means any :class:`polyarray.Program` containing this op emits
        standalone numpy without its caller having to thread ``op_renderers=`` through, since
        ``numpy_source._render_op`` discovers it unaided.

        Parameters
        ----------
        args
            The already-rendered argument expressions ``[Q, R]``, in operand order.

        Returns
        -------
        str
            A numpy expression yielding the multi-output ``(Q·diag(s), diag(s)·R)`` with
            ``s = sign(diag R)`` and zeros mapped to ``+1``.
        """
        return (f"(lambda _Q, _R: (lambda _s: (_Q * _s, _s[:, None] * _R))"
                f"(np.where(np.sign(np.diag(_R)) == 0, 1.0, np.sign(np.diag(_R)))))"
                f"(np.asarray({args[0]}, dtype=float), np.asarray({args[1]}, dtype=float))")

    def __pyab_lower__(self, builder: object, args: list, low: object) -> list:
        """Lower this op into pyab expressions, computing ``(Q·s, s[:, None]·R)``.

        The sign vector is ``s = sign(diag R)`` with a zero read as ``+1``, exactly as in
        :meth:`__call__`. It differs from chartlib's own sign fix only in taking ``(Q, R)`` as
        given rather than factorizing.
        """
        c = low.core
        q, r = args
        raw = low._ns_call("sign", (low._ns_call("diag", (r,)),))
        signs = builder.assign_new(_sb_signs_zero_to_one(low, raw), base="s_qrsc")
        q_out = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=q, rhs=signs)
        s_col = c.ReshapeExpr(a=signs, shape=(c.IntLit(value=-1), c.IntLit(value=1)))
        r_out = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=s_col, rhs=r)
        return [q_out, r_out]


def _qr_q_shape(f_matrix: pa.SymArray, dom_dim: int) -> tuple[int, ...]:
    """Give the shape of a reduced QR's ``Q``: the map's row count by its column count.

    The map is stored ``(cod, dom)``, and a reduced ``Q`` keeps one orthonormal column per
    source dimension.
    """
    shp = _shape_of(f_matrix)
    return (shp[0], dom_dim)


def _realize_via_svd(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
    dom_dim: int, cod_dim: int, label: str,
    *, dim_atom: pa.DimAtom | None = None,
) -> FfsRealization:
    """Realize a subspace from the full SVD of the map, with the rank known only at run time.

    This is the general route, taken when nothing about the map licenses a cheaper one. The
    numerical rank ``δ`` is an output of the decomposition, so the subspace axis is dynamic and
    sized by a :class:`~polyarray.DimAtom` — either the caller's canonical atom, which is what
    makes the metric-free and metric realizations of one subspace agree, or a fresh one minted
    from this decomposition's own rank output.
    """
    U, S, Vh, rank = program.emit_stmt(
        pa.SvdOp(full_matrices=True),
        [f_matrix],
        [pa.OutSpec("U_" + label, (cod_dim, cod_dim)),
         pa.OutSpec("S_" + label, (min(cod_dim, dom_dim),)),
         pa.OutSpec("Vh_" + label, (dom_dim, dom_dim)),
         pa.OutSpec("rank_" + label, ())],
        note=f"ffs-basis:SvdOp({label})",
    )
    svd_idx = len(program.statements) - 1
    # For the image and coimage the subspace dimension IS the rank, so the canonical atom is the
    # rank atom: reuse the caller's if one was given, which is what makes the metric-free SVD and
    # the metric GSVD of this map share one axis, and otherwise mint it from this rank output.
    # For the kernel and cokernel the rank atom is internal and the subspace dimension is the
    # complement, so a supplied atom is passed on as the complement atom instead.
    if which in ("img", "coimg") and dim_atom is not None:
        delta = dim_atom
    else:
        delta = pa.DimAtom(name=f"rank:{label}", source=(svd_idx, 3))
    comp_atom = dim_atom if which in ("ker", "coker") else None
    cols = _take_ffs_block(
        program, which, U, Vh, rank, delta, cod_dim, dom_dim, label,
        comp_atom=comp_atom,
    )
    return FfsRealization(cols=cols, rank=rank, rank_dim=delta, op="SvdOp")


def _realize_via_gsvd(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
    dom_dim: int, cod_dim: int,
    metric_dom: pa.SymArray | None, metric_cod: pa.SymArray | None, label: str,
    *, dim_atom: pa.DimAtom | None = None,
) -> FfsRealization:
    """Realize a subspace whose basis must be orthonormal in the endpoint metrics.

    The generalized decomposition takes the map together with a metric on each endpoint, an
    identity standing in wherever one is absent, and produces factors orthonormal in those
    metrics rather than in the standard inner product. Its basis is therefore not standard-
    orthonormal, and :func:`ffs_cols_orthonormal` reports as much.
    """
    m_v = metric_dom if metric_dom is not None else _eye(program, dom_dim, label + "_Mv")
    m_w = metric_cod if metric_cod is not None else _eye(program, cod_dim, label + "_Mw")
    # The block widths here are runtime values — δ and its complement — and a DimAtom must
    # reference a *prior* Stmt's output, so the decomposition cannot size its own blocks.  Hence
    # two statements: the first emits the full-width factors [U|UI] (cod×cod) and [V|VI]
    # (dom×dom) at static sizes, and the second slices the runtime-δ block out of them, by which
    # point the rank output it needs is prior.
    Ufull, Vfull, _S, rank = program.emit_stmt(
        pa.GSvdFullOp(),
        [f_matrix, m_v, m_w],
        [pa.OutSpec("Ufull_" + label, (cod_dim, cod_dim)),
         pa.OutSpec("Vfull_" + label, (dom_dim, dom_dim)),
         pa.OutSpec("S_" + label, (min(cod_dim, dom_dim),)),
         pa.OutSpec("rank_" + label, ())],
        note=f"ffs-basis:GSvdOp({label})",
    )
    gsvd_idx = len(program.statements) - 1
    # Image and coimage size their axis by δ itself, the same atom the metric-free SVD of this
    # map uses; kernel and cokernel size theirs by the complement.
    if which in ("img", "coimg") and dim_atom is not None:
        delta = dim_atom
    else:
        delta = pa.DimAtom(name=f"rank:{label}", source=(gsvd_idx, 3))
    # The subspace is the leading block of a factor and its complement the trailing one: image =
    # first δ of Ufull, coker = last (cod−δ) of Ufull; coimg = first δ of Vfull, ker = last
    # (dom−δ) of Vfull.
    factor, ambient = (Ufull, cod_dim) if which in ("img", "coker") else (Vfull, dom_dim)
    if which in ("img", "coimg"):
        [block] = program.emit_stmt(
            pa.FirstColsOp(), [factor, rank],
            [pa.OutSpec("ffs_" + which + "_" + label, (ambient, delta))],
            note=f"ffs-basis(metric):{which}=first-δ({label})",
        )
    else:
        comp = dim_atom if dim_atom is not None else _comp_dim(program, rank, ambient, label)
        [block] = program.emit_stmt(
            pa.LastColsOp(), [factor, rank],
            [pa.OutSpec("ffs_" + which + "_" + label, (ambient, comp))],
            note=f"ffs-basis(metric):{which}=last-(n-δ)({label})",
        )
    return FfsRealization(cols=block, rank=rank, rank_dim=delta, op="GSvdOp")


def _take_ffs_block(
    program: pa.Program, which: str, U: pa.SymArray, Vh: pa.SymArray,
    rank: pa.SymArray, delta: pa.DimAtom, cod_dim: int, dom_dim: int, label: str,
    *, comp_atom: pa.DimAtom | None = None,
) -> pa.SymArray:
    """Slice the columns of one subspace out of the full SVD factors.

    The image and cokernel come from the left factor and the kernel and coimage from the right,
    with the subspace itself the leading block and its orthogonal complement the trailing one::

        img    first δ columns of U        coker  last cod−δ columns of U
        coimg  first δ columns of V        ker    last dom−δ columns of V

    ``V`` is materialized as a transpose of ``Vh`` when a right-factor subspace is asked for.
    Both block widths are runtime values, so ``δ`` sizes the leading slice and a complement atom
    — supplied by the caller or computed here — sizes the trailing one.
    """
    if which in ("img", "coker"):
        factor, ambient = U, cod_dim
    # build V = Vhᵀ for the right-factor cases.
    if which in ("coimg", "ker"):
        [V] = program.emit_stmt(
            pa.TransposeOp(),
            [Vh],
            [pa.OutSpec("V_" + label, (dom_dim, dom_dim))],
            note=f"ffs-basis:transpose(Vh)({label})",
        )
        factor, ambient = V, dom_dim

    if which in ("img", "coimg"):
        out_shape: tuple[int | pa.DimAtom, ...] = (factor_rows(factor, ambient), delta)
        [cols] = program.emit_stmt(
            pa.FirstColsOp(),
            [factor, rank],
            [pa.OutSpec("ffs_" + which + "_" + label, out_shape)],
            note=f"ffs-basis:{which}=first-δ({label})",
        )
    else:  # coker / ker: last (ambient - δ) cols
        comp = comp_atom if comp_atom is not None else _comp_dim(program, rank, ambient, label)
        [cols] = program.emit_stmt(
            pa.LastColsOp(),
            [factor, rank],
            [pa.OutSpec("ffs_" + which + "_" + label,
                        (factor_rows(factor, ambient), comp))],
            note=f"ffs-basis:{which}=last-(n-δ)({label})",
        )
    return cols


def factor_rows(factor: pa.SymArray, ambient: int) -> int:
    """Give the ambient row count of a subspace column block: the dimension it embeds into.

    Both SVD factors are square in their own ambient dimension, so this is that dimension
    whichever factor the block was sliced from.
    """
    return ambient


# A complement dimension ``n − δ`` is a runtime axis of its own.  A ``DimAtom`` resolves through
# its source's runtime binding, so it has to reference a Stmt that actually produces ``n − δ``,
# and the SVD's rank output produces ``δ``.  Hence a small ``CompRankOp`` Stmt computing
# ``ambient − rank`` as a fresh 0-d integer output, with the complement atom sourced from that.
def _comp_dim(
    program: pa.Program, rank: pa.SymArray, ambient: int, label: str,
) -> pa.DimAtom:
    """Mint the runtime dimension atom for a complement axis, ``ambient − rank``.

    See the note above for why this needs a statement of its own rather than referring to the
    decomposition's rank output.
    """
    [comp_rank] = program.emit_stmt(
        pa.CompRankOp(ambient), [rank],
        [pa.OutSpec(f"comp_rank_{label}", ())],
        note=f"ffs-basis:comp-rank({ambient}−δ)({label})",
    )
    comp_idx = len(program.statements) - 1
    return pa.DimAtom(name=f"comp(rank:{label},{ambient})", source=(comp_idx, 0))


# ---------------------------------------------------------------------------
# Small numeric ops used by FFS realization (frozen, hashable; numpy __call__).
# ---------------------------------------------------------------------------


def _eye(program: pa.Program, n: int, name: str) -> pa.SymArray:
    """Emit an ``n``-dimensional identity matrix, which stands in for an absent metric."""
    [eye] = program.emit_stmt(
        pa.EyeOp(n), [], [pa.OutSpec(name, (n, n))], note=f"metric:identity({n})",
    )
    return eye


def _shape_of(sa: pa.SymArray) -> tuple[int, ...]:
    """Give a ``SymArray``'s shape, from its bulk array when it has one and its cells otherwise."""
    if getattr(sa, "_bulk", None) is not None:
        return tuple(sa._bulk.shape)
    return tuple(np.asarray(sa.cells).shape)
