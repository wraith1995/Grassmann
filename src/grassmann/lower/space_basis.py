"""``space_basis`` — default (inner-product-orthonormal) basis + candidates (`41 §1`).

``space_basis(space) -> (default_basis, candidates)``.

The **default** is the paper's *standard basis*: orthonormal w.r.t. the space's inner
product, so the metric is the identity in coordinates (``Inner``/``Dagger``/pinv become
numeric — the major min-sym win, `41 §1`/`§1b`). It mirrors the Space tower onto the
Basis tower per the `41 §1` table:

```
Scalar      ⇒ ScalarBasis                ⊕(sᵢ) ⇒ BasisSum(default sᵢ)
                                         ⊗(sᵢ) ⇒ BasisProd(default sᵢ)
Dual s      ⇒ DualBasis(default s)       ⊕ⁿ s  ⇒ BasisRep(n, default s)
WithInner s g ⇒ default = g-orthonormal WithInnerBasis  (candidate: reuse default s)
Img f       ⇒ ProjectedBasis(f, "image") Ker/CoImg/CoKer f ⇒ ProjectedBasis(f, which, …)
SVar α      ⇒ BVar(β)                    (basis supplied at instantiation)
```

The **candidates** feed the symbolic-aware choice in :mod:`grassmann.lower.choose`
(`41 §1b`): the operator's SVD/structure-aligned basis, the FFS ``src``/``tgt`` bases,
and the g-orthonormal basis. Realizing an FFS basis is the SVD/QR seam (`41 §1`,
`50 §A1`): :class:`SvdOp` (+ runtime rank ``δ_f`` as a :class:`~polyarray.DimAtom`) for
the generic case, :class:`QrOp` for the injective / ``IsoSpace`` case, :class:`GSvdOp`
for the metric (``WithInner``) case — realized by :func:`realize_ffs_basis`.
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
    """Give the default inner-product-orthonormal "standard" basis for ``space``."""
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
            # as a candidate (`73 Topic 2`).
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
    """Structure-mirroring basis functor.

    A general basis CHOICE for ``space`` that mirrors the basis tower — ``choice(space) -> Basis | None`` is
    consulted at EVERY node first (``None`` = no override there); un-overridden structural wrappers
    (``Dual``/``OPlus``/``OProd``/``OPlusRep``/``WithInner``) mirror and recurse; every other un-overridden
    node falls to :func:`default_basis`. Consumers supply ONLY their leaf policy (e.g. pointwise's
    immersed-tangent ``ImageBasis`` pin) instead of hand-rolling the tower walk.
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
    """Structure-mirroring SPACE functor — the `Space -> Space` sibling of :func:`map_basis`.

    ``choice(space) -> Space | None`` is consulted at EVERY node first (``None`` = no override
    there); un-overridden structural wrappers (``Dual``/``OPlus``/``OProd``/``OPlusRep``/
    ``WithInner``) mirror and recurse; every other un-overridden node is returned unchanged.
    Consumers supply ONLY their leaf policy instead of hand-rolling the tower walk — the same
    argument that produced ``map_basis``.

    ``map_choice(f) -> Term | None`` (optional) is applied to an FFS node's DEFINING MAP, so a consumer
    can rewrite ``Img(f)`` by rewriting ``f`` itself — without matching over ``Space`` in its own code
    (which would be a partial ladder over a closed sum type). ``None`` = leave the map alone.

    Written for swapping ONE immersion for another inside a value space: a form-valued DOF read
    against a σ-PERMUTED chart is the same space with ``Img(J_canon)`` replaced by ``Img(J_perm)``
    everywhere it occurs, however deep it sits under ``Dual``/``⊗``/``⊕`` — or inside the map that
    DEFINES an FFS node, which is where ``Λᵏ(Img J) = Img(alt)`` keeps it.
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
    """Give an FFS node's defining map under the optional `map_choice`; unchanged when it declines."""
    if map_choice is None:
        return f
    out = map_choice(f)
    return f if out is None else out


def _map_opt(
    s: S.Space | None,
    choice: Callable[[S.Space], S.Space | None],
    map_choice: Callable[[T.Term], T.Term | None] | None = None,
) -> S.Space | None:
    """Run `map_space` on an FFS node's optional ENDPOINT space.

    An FFS node is not a leaf of the walk: a buried immersion reaches the tower only through the
    `source`/`target` of the map that defines it. `Λᵏ(Img J) = Img(alt)` keeps its `Img(J)` in `alt`'s
    endpoints, so treating `Img` as a leaf made a σ readout on a FACE (a 2-form moment) find nothing to
    substitute — `tgt == src`, no transport, and every `O_σ` came back `I`. `_immersed_basis`/
    `_immersed_endpoints` already recurse this way on the BASIS side; this is the Space-side match.
    """
    return None if s is None else map_space(s, choice, map_choice=map_choice)


def _maybe(s: S.Space | None) -> B.Basis | None:
    return None if s is None else default_basis(s)


# ---------------------------------------------------------------------------
# Candidate bases (fed to §1b choice).
# ---------------------------------------------------------------------------


def candidate_bases(space: S.Space) -> tuple[B.Basis, ...]:
    """Alternative bases to score against the term in `41 §1b`.

    Always includes the default; adds the operator's SVD/structure-aligned basis, the
    FFS ``src``/``tgt`` bases, and (for a metric space) both the g-orthonormal default
    and the plain reuse of the underlying space's basis.
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
            # alternatives the §1b heuristic can score against the term (`73 Topic 2`).
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
    if b not in acc:
        acc.append(b)


def space_basis(space: S.Space) -> tuple[B.Basis, tuple[B.Basis, ...]]:
    """Return ``(default_basis, candidates)`` for ``space`` (`41 §1`).

    ``default_basis`` is the inner-product-orthonormal standard basis; ``candidates``
    are the alternatives (including the default) that the `41 §1b` heuristic scores.
    """
    return default_basis(space), candidate_bases(space)


# ---------------------------------------------------------------------------
# FFS basis realization — the SVD / QR / GSVD seam (`41 §1`, `50 §A1`/`§A5`).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FfsRealization:
    """The polyarray realization of one FFS basis.

    ``cols`` is the SymArray whose columns are the (numeric, orthonormal) basis vectors
    in the ambient (src/tgt standard) coordinates. ``rank`` is the SymArray carrying the
    numerical rank δ_f (a 0-d output, ``None`` when static). ``rank_dim`` is the
    :class:`~polyarray.DimAtom` sizing the dynamic column axis (``None`` when the rank is
    static, e.g. the injective ``QrOp`` case). ``op`` names which op produced it
    (``"SvdOp"`` / ``"QrOp"`` / ``"GSvdOp"``) for provenance.
    """

    cols: pa.SymArray
    rank: pa.SymArray | None
    rank_dim: pa.DimAtom | None
    op: str
    """Which op produced ``cols``.  Read by :func:`ffs_cols_orthonormal` — the frame's
    ORTHONORMALITY is a property of the producing op, and consumers that need the
    coordinate map INTO the frame must branch on it (``Pᵀ`` only when orthonormal)."""
    # The upper-triangular ``R`` of the injective QR (``J = Q·R``), captured so a change of
    # basis between the literal ``ImageBasis`` and the orthonormal ``ProjectedBasis`` of the
    # SAME map is ``R`` directly — no realize-both-and-pseudo-invert (`Stage B`).  ``None``
    # for the SVD/GSVD seams.
    r: pa.SymArray | None = None
    # The SAME subspace frame expressed in the DECLARED ENDPOINT basis instead of the ambient
    # canonical one — i.e. the covariant realization's ``Q₀``, before the ``cols = T·Q₀`` that
    # exists only to satisfy this dataclass's "ambient standard coordinates" contract.  Set only by
    # :func:`~grassmann.lower.represent._realize_image_covariant`; ``None`` everywhere else.
    #
    # Why it is worth carrying: for a NATURAL ``f`` (the exterior ``alt`` under the transported
    # ⊗ᵏ frames — certified mechanically by ``try_constant_ffs`` folding ``A₀`` to a build-time
    # constant) ``Q₀`` is CONSTANT, while ``T`` carries all the cell dependence AND, for an
    # ``Img(J)`` endpoint, the QR radical ``R``.  A consumer whose operand already lives in
    # ``endpoint_basis`` can use ``endpoint_cols`` directly and skip ``T``, ``InvOp(T)`` and its own
    # operand cob entirely.  The two frames span the SAME subspace and yield the SAME subspace
    # coordinates — the ambient change of basis is applied consistently to frame and operand — but
    # ONLY when the projection is EXACT: with a residual, ``pinv`` projects along the ambient
    # orthogonal complement, which differs between the two ambient bases.  Consumers must therefore
    # guard on exactness (measured: for the Λᵏ trace the residual is identically zero because a
    # diagonal pullback of an antisymmetric tensor stays antisymmetric for ANY ι; for a general
    # operand it is 100%). See
    endpoint_cols: pa.SymArray | None = None
    endpoint_basis: object | None = None      # the `B.Basis` `endpoint_cols` is expressed in


# The realization ops whose ``cols`` are STANDARD-orthonormal (``PᵀP = I`` exactly), so the
# adjoint ``Pᵀ`` IS the matched dual ``pinv(P) = (PᵀP)⁻¹Pᵀ`` and a consumer may read
# coordinates with a transpose.  POSITIVE whitelist — anything absent needs the general
# ``pinv``: ``"GSvdOp"`` is *g*-orthonormal (``UᵀGU = I``, not ``UᵀU = I``), ``"MapApply"`` /
# ``"Gram"`` are the LITERAL map columns, and any ``"…+cov"`` frame is ``T·Q₀`` (orthonormal in
# the target basis's inner product, not the standard one).
_STD_ORTHONORMAL_FFS_OPS = frozenset({"QrOp", "SvdOp", "const", "const-svd", "supplied"})


def ffs_cols_orthonormal(real: FfsRealization) -> bool:
    """Whether ``real.cols`` is provably STANDARD-orthonormal (``PᵀP = I``).

    Conservative in the SAFE direction: ``False`` only costs a consumer the general
    ``pinv``/Gram solve, it is never wrong; ``True`` on a non-orthonormal frame is the
    documented silent-wrongness class (`docs/basis-honesty-handoff.md`), so this is a
    positive whitelist of the ops that provably orthonormalize.
    """
    return real.op in _STD_ORTHONORMAL_FFS_OPS


# block index of each SvdOp/GSvdOp output that carries the named subspace.
# SvdOp outputs: (U, S, Vh, rank); GSvdOp outputs: (U, UI, V, VI, S, rank).
_SVD_WHICH_TAKES_RIGHT = ("ker", "coimg")  # built from V (right factor)


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
    """Realize an FFS basis (``which`` ∈ ``"img"|"image"|"coker"|"coimg"|"ker"``) (`41 §1`).

    ``"image"`` (the :class:`~grassmann.core.basis.ProjectedBasis` spelling) is accepted as
    an alias of ``"img"`` — both name the **orthonormal range** (Q/SVD-U) basis.

    Emits the SVD/QR/GSVD ``Stmt``(s) into ``program`` and returns an
    :class:`FfsRealization`. Selection per `41 §1`/`50 §A1`:

    * **injective + image** ⇒ :class:`QrOp` — ``Q`` is the orthonormal range basis with a
      *static* ``rank = dom_dim`` (recovers the dropped ``IsoSpace``);
    * **metric present** ⇒ :class:`GSvdOp` (orthonormal in the metrics);
    * otherwise ⇒ :class:`SvdOp`, with the runtime rank ``δ_f`` sizing a dynamic axis.

    ``dim_atom`` (`51 §FFS-δ`) — a caller-supplied **canonical subspace-dimension** atom
    for the dynamic column axis.  The FFS dimension is metric-independent (the rank, or its
    complement for ker/coker), so a single atom is shared across the metric-free SVD and the
    metric GSVD of one ``(f, which)`` and across every FFS-typed Var of that space.  ``None``
    ⇒ the seam mints a fresh atom (the legacy behaviour) and reports it in
    :attr:`FfsRealization.rank_dim` (and, for ker/coker, the complement atom is the subspace
    axis of ``cols``) so the caller can record it and pass it back next time.

    ``literal_frame`` ("two realizations of the same ffs basis is fine") — on
    the injective-image route ONLY, realize the LITERAL columns (:func:`_realize_via_gram`)
    instead of the orthonormal ``Q``.  Same subspace, a different frame; see that function for
    the precondition and the conditioning price.  Default ``False`` = the orthonormal frame
    (byte-identical to the historical behaviour on every route).
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
    # SOUND STRUCTURAL constant-fold (dependency, NOT sampling): if `f_matrix`'s dependency
    # cone is a closed constant subprogram (no live input ref, only `stmt_out` generators — see
    # `_cone_is_structurally_constant`), its plain-SVD frame is a PROVABLE build-time constant.
    # Fold it once here — emitting the constant `cols` directly — instead of deferring an SvdOp
    # whose downstream `fold_numeric` chokes on the dynamic rank-δ axis (the high-degree `∫u·v`
    # blow-up: ~480 realizations of 2 structurally-constant operators, the 2×2 reference
    # tangent-frame identity and the 4×4 Λ² `alt` antisymmetrizer, OOM-ing codegen).  Only the
    # PLAIN-SVD path is folded (the metric GSVD and the sign-fixed injective QR produce a
    # DIFFERENT orthonormalization — a Euclidean-SVD constant would be the wrong frame there),
    # so the folded `cols` is byte-identical to what `_realize_via_svd` would produce for this
    # `M`.  `try_constant_ffs` (sampling) stays as-is for the covariant `A₀` seam, where
    # `f_matrix` is feed-invariant but dataflow-symbolic (vertex atoms cancel) — a case this
    # dependency check deliberately reports NON-constant (its cone reads vertex generators).
    if _cone_is_structurally_constant(program, f_matrix):
        return _fold_constant_ffs(program, which, f_matrix)
    return _realize_via_svd(
        program, which, f_matrix, dom_dim, cod_dim, label, dim_atom=dim_atom,
    )


def try_constant_ffs(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
    dom_dim: int, cod_dim: int, *, probes: int = 3, atol: float = 1e-7,
) -> FfsRealization | None:
    """Give the FFS ``which`` basis as a BUILD-TIME CONSTANT, or ``None``.

    Applies when ``f_matrix`` is value-INVARIANT under the program's inputs.  Invariance is
    verified by evaluating ONLY
    ``f_matrix``'s dependency cone (``polyarray.evaluate_cone`` — safe on a big/partially-built
    shared program: only the cone runs, so an unrelated singular op / a dynamic input OUTSIDE
    the cone is never touched) at ``probes`` generic feeds and checking the value is identical;
    then the range/complement basis is the SVD of that constant, taken once at build.

    A DYNAMIC (``DimAtom``-sized) input is bound with a size-1 placeholder along its dynamic
    axes: ``build_runtime_bindings`` requires a value for every declared input, but one outside
    the cone is never read — the placeholder just satisfies the binder (an FFS-typed value
    input, sized by a runtime rank, sits OUTSIDE the ``f_matrix`` cone of a geometry-only
    factorisation).
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
# Constant-fold of a structurally-constant FFS realization (dependency-based,
# NOT sampling).  Shared "given a numeric matrix's SVD, emit the constant frame
# for ``which``" core is :func:`_const_ffs_from_svd` (reused by the sampling
# ``try_constant_ffs`` above and the structural fold below).
# ---------------------------------------------------------------------------


def _canonical_subspace_basis(q: np.ndarray) -> np.ndarray:
    """Build a CANONICAL orthonormal basis of ``span(q)``.

    A function of the SUBSPACE (and the ambient index order) alone, not of the factorization that produced
    ``q``.

    Why this exists (`73 Topic 2`). ``U[:,:r]`` is only determined up to a
    rotation *inside* each degenerate singular subspace, so it is a **discontinuous** function of
    the operator: the Λᵏ antisymmetrizer's nonzero singular values are degenerate (measured
    ``σ = 1, 1, 1, 0…`` for the 3-dim ``Λ²`` case) and the operator we factor is invariant only
    BY CANCELLATION (~1e-15, not bitwise), so 1e-15 of rounding rotated the realized frame by
    O(1).  Two ``Space`` nodes naming the same subspace — ``Img(alt_canonical)`` and
    ``Img(alt_permuted)`` for a reoriented facet — then got unrelated frames, and the change of
    basis composed across them came out a rotation instead of the signed permutation a σ-oriented
    DOF owes.  No memo can tie those two: their keys are legitimately different (measured: 564 FFS
    realizations, 140 id-keys vs 137 structural keys — keying structurally buys ~3 duplicate
    factorizations and nothing else), and the memo cannot span compilations anyway.

    The fix is to stop asking the factorization for the frame.  The orthogonal **projector**
    ``P = q·qᵀ`` is unique for a given subspace and CONTINUOUS in it (unlike ``q`` itself), so a
    basis read off ``P`` in a fixed ambient order is stable at ~1e-15 where the SVD basis was
    stable at O(1).  This is pivoted Cholesky of a projector in index order: greedily orthogonalize
    ``P``'s columns ``P·e_j``, ``j = 0…n-1``, keeping those with a non-negligible residual.

    It is SELF-SIGN-FIXING, which is why no ad-hoc sign convention appears here: after
    orthogonalizing against the accepted columns, ``w = P'·e_j`` with ``P'`` still an orthogonal
    projector, so ``w[j] = e_jᵀP'ᵀP'e_j = ‖w‖² ≥ 0`` — the accepted column's own coordinate is
    positive by construction, and ``‖w‖² = w[j]`` exactly (used as the acceptance test).

    NOTE this deliberately does NOT carry orientation, and must not: a 3-cycle and a transposition
    of a facet's vertices span the SAME plane, so ``Λᵏ`` of it is the same subspace and any
    subspace-keyed rule returns the same vector for both.  The σ sign rides on the ambient
    chart-to-chart ``convert`` in the middle of ``_pb_ffs`` (``Project ∘ convert ∘ Embed``), where
    it belongs and where it was measured correct (single-factor convert, ``det = −1``): with
    ``P_S = P_T = p`` the composite is ``pᵀ(det M · p) = det M = ±1``.  Making the two endpoints
    agree is exactly what turns that into a clean sign.
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
    """Emit the BUILD-TIME-CONSTANT :class:`FfsRealization` for ``which`` from a full SVD.

    Takes a numeric matrix ``M``'s full SVD ``M = U·diag(S)·Vh`` and its rank ``r``, and uses
    the SAME range/complement column convention :func:`_realize_via_svd` slices from the
    runtime SVD factors:

    * ``img``   = first ``r`` columns of ``U``   (the orthonormal range basis);
    * ``coker`` = last  ``cod−r`` columns of ``U`` (its orthogonal complement);
    * ``coimg`` = first ``r`` columns of ``V = Vhᵀ`` (the right factor);
    * ``ker``   = last  ``dom−r`` columns of ``V``.

    ``cols`` is a build-time-constant numeric SymArray; ``rank``/``rank_dim`` are ``None``
    (the axis is static — the numeric rank is known at build).  Shared by the sampling
    (:func:`try_constant_ffs`) and structural (:func:`realize_ffs_basis`) constant paths so
    both produce an IDENTICAL frame for the same numeric ``M``.
    """
    if which in ("img", "coker"):
        cols = u[:, :r] if which == "img" else u[:, r:]
    else:                                                        # coimg / ker: right factor V = Vhᵀ
        v = vh.T
        cols = v[:, :r] if which == "coimg" else v[:, r:]
    # The SLICE fixes the SUBSPACE; the slice's own columns do not fix a frame (they are arbitrary
    # inside a degenerate singular subspace). Re-derive the frame canonically from that subspace so
    # one subspace has ONE frame everywhere — see `_canonical_subspace_basis`.
    cols = _canonical_subspace_basis(np.ascontiguousarray(cols))
    return FfsRealization(cols=pa.SymArray(np.ascontiguousarray(cols)),
                          rank=None, rank_dim=None, op=op)


def _cone_is_structurally_constant(program: pa.Program, f_matrix: pa.SymArray) -> bool:
    """Report whether ``f_matrix``'s value is a PROVABLE build-time constant.

    Sound, by DEPENDENCY — not by sampling.

    Thin delegate to :func:`polyarray.is_structurally_constant` (the check moved to
    polyarray, since it uses ONLY polyarray machinery — ``dependency_cone``, the ref
    types, and ``program.env`` provenance).  Returns ``True`` iff ``f_matrix``'s
    dependency cone is a *closed constant subprogram*: no Stmt input is a live feed
    (:class:`~polyarray.ir.InputRef` / :class:`~polyarray.ir.IntAtomRef`) and no cell
    generator anywhere in the cone (nor in ``f_matrix`` itself) has provenance ``kind``
    other than ``"stmt_out"``.

    EXACT, not heuristic: never mistakes a feed-varying operator for a constant.
    Conservative: any ref/generator that cannot be positively classified constant-safe —
    or a cone that fails to enumerate — forces ``False``, so a frame with any live input
    dependency is left to the SVD/QR/GSVD seam.
    """
    return pa.is_structurally_constant(program, f_matrix)


def _fold_constant_ffs(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
) -> FfsRealization:
    """Fold a PROVABLY-constant ``f_matrix`` to its build-time-constant FFS ``which`` frame.

    See :func:`_cone_is_structurally_constant`.  Evaluates ``f_matrix``'s cone ONCE at build
    (:func:`polyarray.evaluate_cone` — the cone has no live inputs, so any binding gives the
    same matrix; a zero binding is used) → the numeric operator ``M``, then emits the constant
    range/complement basis via the SHARED :func:`_const_ffs_from_svd` core using ``M``'s SVD.
    The result is byte-for-byte the frame :func:`_realize_via_svd` would produce for this ``M``
    (same full SVD, same ``matrix_rank`` tolerance as :class:`~polyarray.SvdOp`, same column
    convention) — but as a constant, with NO SvdOp/rank Stmt emitted.
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
    """Realize :class:`~grassmann.core.basis.ImageBasis` — the LITERAL map-application.

    Columns are ``f`` applied to the source basis (``f(eᵢ)``), i.e. the ``(cod, dom)``
    operator matrix itself — **not** orthonormalized (the R/fwd side, `73 Topic 2`).  No
    QR/SVD is emitted; the columns ARE ``f_matrix`` (passed in ``(cod, dom)`` orientation).
    The column axis is the static source dimension ``dom_dim`` (the ``f(eᵢ)`` per source
    basis vector); ``rank``/``rank_dim`` are ``None`` (no SVD δ).
    """
    [cols] = program.emit_stmt(
        pa.IdentityOp(), [f_matrix],
        [pa.OutSpec("imgapp_" + label, (cod_dim, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):map-application",
    )
    return FfsRealization(cols=cols, rank=None, rank_dim=None, op="MapApply")


def _sb_signs_zero_to_one(low: object, signs: object) -> object:
    """``where(signs == 0, 1.0, signs)``.

    The +diagonal-R sign convention (0 ⇒ +1) as a pyab expr (space_basis-local twin of the numeric ``s[s == 0]
    = 1.0``).
    """
    c = low.core
    zero = c.ArrayExpr(obj=c.FloatLit(value=0.0), dtype=low._dtype_f64())
    one = c.ArrayExpr(obj=c.FloatLit(value=1.0), dtype=low._dtype_f64())
    return c.WhereExpr(cond=c.BinaryExpr(op=c.BinaryOp.EQ, lhs=signs, rhs=zero), x=one, y=signs)


def _realize_via_qr(
    program: pa.Program, f_matrix: pa.SymArray, dom_dim: int, label: str,
) -> FfsRealization:
    # f : (cod, dom); injective ⇒ Q has dom orthonormal columns (the range basis).
    Q, R = program.emit_stmt(
        pa.QrOp(mode="reduced"),
        [f_matrix],
        [pa.OutSpec("Q_" + label, _qr_q_shape(f_matrix, dom_dim)),
         pa.OutSpec("R_" + label, (dom_dim, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):inj=>QrOp",
    )
    # Positive-diagonal-R sign convention (numpy's qr does not fix the sign): with s=sign(diag R),
    # Q·diag(s) and diag(s)·R is the unique QR whose Q keeps the orientation of f's columns.
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
    """Build the LITERAL-COLUMN (Gram) realization of an injective map's image.

    The SECOND realization of the SAME FFS basis ("two realizations of the
    same ffs basis is fine"): ``cols = J`` itself — the map's own columns — with ``r = None``
    (``J = cols·I``).  Same subspace ``Img(f)``, same ``FfsRealization`` contract, a
    DIFFERENT (non-orthonormal) frame, tagged ``op="Gram"`` so
    :func:`ffs_cols_orthonormal` reports ``False`` and every consumer that needs coordinates
    IN the frame takes its general route — the matched dual ``pinv(J) = (JᵀJ)⁻¹Jᵀ = G⁻¹Jᵀ``
    (the Gram solve this realization is named for) instead of ``Jᵀ``.

    WHY it exists: :func:`_realize_via_qr`'s ``Q`` is ALGEBRAIC-IRRATIONAL in the inputs (QR
    normalizes by ``√(…)``), so polyarray's exact rational lane cannot carry it and every
    certificate downstream of it degrades to probe-and-freeze.  ``J`` is RATIONAL in them, so
    the exact lane certifies through it.  The price is conditioning — coordinates come from
    ``G = JᵀJ`` whose condition number is ``κ(J)²`` — which is why this realization is OPT-IN
    (``literal_frame``) and never the numeric default.

    PRECONDITION: ``f`` INJECTIVE — the same precondition :func:`_realize_via_qr` already
    requires, and the reason no new assumption enters: injective ⇒ the columns are a basis of
    the image (full column rank) ⇒ ``G`` is invertible and ``pinv(J)·J = I`` identically.
    Callers select this route only under that guard (:func:`realize_ffs_basis`).
    """
    [cols] = program.emit_stmt(
        pa.IdentityOp(), [f_matrix],
        [pa.OutSpec("Jg_" + label, _qr_q_shape(f_matrix, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):inj=>literal-cols(Gram)",
    )
    return FfsRealization(cols=cols, rank=None, rank_dim=None, op="Gram")


@dataclass(frozen=True)
class _QrSignConventionOp:
    """Apply the +diagonal-R sign convention to a ``(Q, R)`` factorization.

    ``Q·diag(s)``, ``diag(s)·R`` with ``s = sign(diag R)`` (zeros ⇒ +1) ⇒ ``R`` has a positive diagonal.
    """

    def __call__(self, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        s = np.sign(np.diag(R))
        s[s == 0] = 1.0
        return Q * s, s[:, None] * R

    def __numpy_source__(self, args: list[str]) -> str:
        """``to_numpy_source`` op-carried emit hook, discovered by ``numpy_source._render_op``.

        Renders the +diagonal-R sign convention so ANY :class:`polyarray.Program` carrying this
        op emits standalone numpy without the caller threading ``op_renderers=`` — the emit twin
        of :meth:`__pyab_lower__`, and the same value body as :meth:`__call__`.

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
        """Pyab op-carried lowering.

        The +diagonal-R sign convention ``(Q·s, s[:,None]·R)`` with ``s = sign(diag R)`` (0 ⇒ +1) (twin of the
        ``_qr_sign_convention`` numpy renderer / :meth:`__call__`; same math as chartlib's ``QrSignFixOp`` but
        on given ``(Q, R)``).
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
    shp = _shape_of(f_matrix)
    return (shp[0], dom_dim)


def _realize_via_svd(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
    dom_dim: int, cod_dim: int, label: str,
    *, dim_atom: pa.DimAtom | None = None,
) -> FfsRealization:
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
    # For img/coimg the canonical subspace atom IS the rank atom δ; reuse a caller-supplied
    # one (so the metric-free SVD and the metric GSVD share it), else mint from this rank
    # output.  For ker/coker the rank atom is internal and the SUBSPACE atom is the
    # complement; ``dim_atom`` (when given) flows to ``_take_ffs_block`` as the comp atom.
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
    m_v = metric_dom if metric_dom is not None else _eye(program, dom_dim, label + "_Mv")
    m_w = metric_cod if metric_cod is not None else _eye(program, cod_dim, label + "_Mw")
    # GSvdOp's block widths are runtime (δ_f / its complement) and cannot be self-sized
    # by GSvdOp's own rank output (a DimAtom must reference a *prior* Stmt's output).  So
    # emit GSvdOp with STATIC full-width factors [U|UI] (cod×cod) and [V|VI] (dom×dom)
    # via a thin wrapper, then SLICE the runtime-δ block in a second Stmt that consumes
    # the now-prior rank output.
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
    # img/coimg: canonical subspace atom IS δ (shared with the metric-free SVD); ker/coker:
    # the subspace atom is the complement (see `51 §FFS-δ`).
    if which in ("img", "coimg") and dim_atom is not None:
        delta = dim_atom
    else:
        delta = pa.DimAtom(name=f"rank:{label}", source=(gsvd_idx, 3))
    # Block↔FFS: image = first δ of Ufull, coker = last (cod−δ) of Ufull;
    #            coimg = first δ of Vfull, ker = last (dom−δ) of Vfull.
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
    """Slice the rank-δ FFS block from the full SVD factors (`41 §1`).

    img  = first δ cols of U;   coker = last cod-δ cols of U;
    coimg= first δ cols of V;   ker   = last dom-δ cols of V   (V = Vhᵀ).
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
    return ambient


# A complement dimension n−δ is itself a runtime axis.  A ``DimAtom`` resolves from its
# ``source``'s run-time binding, so it must reference a Stmt that *actually produces n−δ*,
# not the SVD ``rank`` output (which produces δ).  We therefore emit a tiny ``_CompRankOp``
# Stmt computing ``ambient − rank`` as a fresh 0-d int output and source the complement
# ``DimAtom`` from THAT, so the dynamic axis resolves to ``ambient − δ`` at run time.
def _comp_dim(
    program: pa.Program, rank: pa.SymArray, ambient: int, label: str,
) -> pa.DimAtom:
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
    [eye] = program.emit_stmt(
        pa.EyeOp(n), [], [pa.OutSpec(name, (n, n))], note=f"metric:identity({n})",
    )
    return eye


def _shape_of(sa: pa.SymArray) -> tuple[int, ...]:
    if getattr(sa, "_bulk", None) is not None:
        return tuple(sa._bulk.shape)
    return tuple(np.asarray(sa.cells).shape)
