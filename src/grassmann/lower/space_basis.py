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

from typing import assert_never

from dataclasses import dataclass

import numpy as np
import polyarray as pa

from grassmann.core import basis as B
from grassmann.core import props as P
from grassmann.core import space as S

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
    """The default inner-product-orthonormal "standard" basis for ``space`` (`41 §1`)."""
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


def map_basis(space: S.Space, choice) -> B.Basis:
    """Structure-mirroring basis functor (2026-07-03, fem audit #18): a general basis CHOICE
    for ``space`` that mirrors the basis tower — ``choice(space) -> Basis | None`` is consulted
    at EVERY node first (``None`` = no override there); un-overridden structural wrappers
    (``Dual``/``OPlus``/``OProd``/``OPlusRep``/``WithInner``) mirror and recurse; every other
    un-overridden node falls to :func:`default_basis`. Consumers supply ONLY their leaf policy
    (e.g. pointwise's immersed-tangent ``ImageBasis`` pin) instead of hand-rolling the tower walk."""
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
        case _:  # audit: ok EX-WILDCARD — bail-default (§0): out[0]=default_basis; extras are optimization [approved Teo 2026-07-03]
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
    # The upper-triangular ``R`` of the injective QR (``J = Q·R``), captured so a change of
    # basis between the literal ``ImageBasis`` and the orthonormal ``ProjectedBasis`` of the
    # SAME map is ``R`` directly — no realize-both-and-pseudo-invert (`Stage B`).  ``None``
    # for the SVD/GSVD seams.
    r: pa.SymArray | None = None


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
    fold_constant: bool = False,
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
    """
    if which == "image":
        which = "img"
    if which not in ("img", "coker", "coimg", "ker"):
        raise ValueError(f"realize_ffs_basis: bad which {which!r}")

    injective = props is not None and props.inj
    has_metric = metric_dom is not None or metric_cod is not None

    if which == "img" and injective and not has_metric:
        return _realize_via_qr(program, f_matrix, dom_dim, label)
    if has_metric:
        return _realize_via_gsvd(
            program, which, f_matrix, dom_dim, cod_dim,
            metric_dom, metric_cod, label, dim_atom=dim_atom,
        )
    # `fold_constant` (OPT-IN, the FEEC covariant / exterior-power readout only — NOT the
    # general FFS realization, whose SVD-emission / memoization / dynamic-bulk contract other
    # callers rely on): a value-INVARIANT `f_matrix` folds to a build-time-constant range
    # basis.  See `try_constant_ffs`.
    if fold_constant:
        folded = try_constant_ffs(program, which, f_matrix, dom_dim, cod_dim)
        if folded is not None:
            return folded
    return _realize_via_svd(
        program, which, f_matrix, dom_dim, cod_dim, label, dim_atom=dim_atom,
    )


def try_constant_ffs(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
    dom_dim: int, cod_dim: int, *, probes: int = 3, atol: float = 1e-7,
) -> FfsRealization | None:
    """If ``f_matrix`` is value-INVARIANT under the program's inputs, its FFS ``which`` basis
    as a BUILD-TIME CONSTANT, else ``None``.  Invariance is verified by evaluating ONLY
    ``f_matrix``'s dependency cone (``polyarray.evaluate_cone`` — safe on a big/partially-built
    shared program: only the cone runs, so an unrelated singular op / a dynamic input OUTSIDE
    the cone is never touched) at ``probes`` generic feeds and checking the value is identical;
    then the range/complement basis is the SVD of that constant, taken once at build.

    A DYNAMIC (``DimAtom``-sized) input is bound with a size-1 placeholder along its dynamic
    axes: ``build_runtime_bindings`` requires a value for every declared input, but one outside
    the cone is never read — the placeholder just satisfies the binder (an FFS-typed value
    input, sized by a runtime rank, sits OUTSIDE the ``f_matrix`` cone of a geometry-only
    factorisation)."""
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
        if which in ("img", "coker"):
            cols = u[:, :r] if which == "img" else u[:, r:]
        else:                                                    # coimg / ker: right factor V = Vhᵀ
            v = vh.T
            cols = v[:, :r] if which == "coimg" else v[:, r:]
        return FfsRealization(cols=pa.SymArray(np.ascontiguousarray(cols)),
                              rank=None, rank_dim=None, op="const-svd")
    except Exception:  # noqa: BLE001 — best-effort; the dynamic SVD is the fallback
        return None


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
        _IdentityMatrixOp(), [f_matrix],
        [pa.OutSpec("imgapp_" + label, (cod_dim, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):map-application",
    )
    return FfsRealization(cols=cols, rank=None, rank_dim=None, op="MapApply")


@dataclass(frozen=True)
class _IdentityMatrixOp:
    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(A, dtype=float)


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


@dataclass(frozen=True)
class _QrSignConventionOp:
    """Apply the +diagonal-R sign convention to a ``(Q, R)`` factorization: ``Q·diag(s)``,
    ``diag(s)·R`` with ``s = sign(diag R)`` (zeros ⇒ +1) ⇒ ``R`` has a positive diagonal."""

    def __call__(self, Q: np.ndarray, R: np.ndarray):  # noqa: N803
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)
        s = np.sign(np.diag(R))
        s[s == 0] = 1.0
        return Q * s, s[:, None] * R


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
        _GSvdFullOp(),
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
            _FirstColsOp(), [factor, rank],
            [pa.OutSpec("ffs_" + which + "_" + label, (ambient, delta))],
            note=f"ffs-basis(metric):{which}=first-δ({label})",
        )
    else:
        comp = dim_atom if dim_atom is not None else _comp_dim(program, rank, ambient, label)
        [block] = program.emit_stmt(
            _LastColsOp(), [factor, rank],
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
            _TransposeOp(),
            [Vh],
            [pa.OutSpec("V_" + label, (dom_dim, dom_dim))],
            note=f"ffs-basis:transpose(Vh)({label})",
        )
        factor, ambient = V, dom_dim

    if which in ("img", "coimg"):
        out_shape: tuple[int | pa.DimAtom, ...] = (factor_rows(factor, ambient), delta)
        [cols] = program.emit_stmt(
            _FirstColsOp(),
            [factor, rank],
            [pa.OutSpec("ffs_" + which + "_" + label, out_shape)],
            note=f"ffs-basis:{which}=first-δ({label})",
        )
    else:  # coker / ker: last (ambient - δ) cols
        comp = comp_atom if comp_atom is not None else _comp_dim(program, rank, ambient, label)
        [cols] = program.emit_stmt(
            _LastColsOp(),
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
        _CompRankOp(ambient), [rank],
        [pa.OutSpec(f"comp_rank_{label}", ())],
        note=f"ffs-basis:comp-rank({ambient}−δ)({label})",
    )
    comp_idx = len(program.statements) - 1
    return pa.DimAtom(name=f"comp(rank:{label},{ambient})", source=(comp_idx, 0))


# ---------------------------------------------------------------------------
# Small numeric ops used by FFS realization (frozen, hashable; numpy __call__).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TransposeOp:
    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(A).T


@dataclass(frozen=True)
class _FirstColsOp:
    def __call__(self, A: np.ndarray, rank: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(A)[:, : int(rank)]


@dataclass(frozen=True)
class _LastColsOp:
    def __call__(self, A: np.ndarray, rank: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(A)[:, int(rank):]


@dataclass(frozen=True)
class _CompRankOp:
    """The complement rank ``ambient − δ`` as a 0-d int — sizes the FFS complement axis.

    A ``DimAtom`` resolves from its ``source`` Stmt's run-time value, so the complement
    (Ker/CoKer) axis must be sourced from a Stmt that produces ``ambient − δ`` rather than
    the SVD ``rank`` (which produces δ).  This op is that Stmt (`41 §1`).
    """

    ambient: int

    def __call__(self, rank: np.ndarray) -> np.ndarray:
        return np.asarray(self.ambient - int(rank))


@dataclass(frozen=True)
class _GSvdFullOp:
    """Run :class:`~polyarray.GSvdOp` and return full-width metric-orthonormal factors.

    Returns ``(Ufull=[U|UI], Vfull=[V|VI], S, rank)`` with *static* widths (cod×cod and
    dom×dom), so the runtime-δ block slice happens in a downstream Stmt (which can
    reference this op's rank output). Image/coimg blocks are the leading ``rank`` columns;
    coker/ker the trailing complement — the FFS block↔subspace correspondence (`50 §A1`).
    """

    rcond: float | None = None

    def __call__(self, A: np.ndarray, M_V: np.ndarray, M_W: np.ndarray):  # noqa: N803
        U, UI, V, VI, S, rank = pa.GSvdOp(rcond=self.rcond)(A, M_V, M_W)
        Ufull = np.concatenate([np.asarray(U), np.asarray(UI)], axis=1)
        Vfull = np.concatenate([np.asarray(V), np.asarray(VI)], axis=1)
        return Ufull, Vfull, np.asarray(S), np.asarray(rank)


@dataclass(frozen=True)
class _EyeOp:
    n: int

    def __call__(self) -> np.ndarray:
        return np.eye(self.n)


def _eye(program: pa.Program, n: int, name: str) -> pa.SymArray:
    [eye] = program.emit_stmt(
        _EyeOp(n), [], [pa.OutSpec(name, (n, n))], note=f"metric:identity({n})",
    )
    return eye


def _shape_of(sa: pa.SymArray) -> tuple[int, ...]:
    if getattr(sa, "_bulk", None) is not None:
        return tuple(sa._bulk.shape)
    return tuple(np.asarray(sa.cells).shape)
