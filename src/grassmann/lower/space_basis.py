"""``space_basis`` — default (inner-product-orthonormal) basis + candidates (`41 §1`).

``space_basis(space) -> (default_basis, candidates)``.

The **default** is the paper's *standard basis*: orthonormal w.r.t. the space's inner
product, so the metric is the identity in coordinates (``Inner``/``Dagger``/pinv become
numeric — the major min-sym win, `41 §1`/`§1b`). It mirrors the Space tower onto the
Basis tower per the `41 §1` table:

```
Scalar      ⇒ ScalarBasis                ⊕(sᵢ) ⇒ BasisSum(default sᵢ)
Base(_,_)   ⇒ BaseBasis(name)            ⊗(sᵢ) ⇒ BasisProd(default sᵢ)
Dual s      ⇒ DualBasis(default s)       ⊕ⁿ s  ⇒ BasisRep(n, default s)
WithInner s g ⇒ default = g-orthonormal WithInnerBasis  (candidate: reuse default s)
Img f       ⇒ ImageBasis(f)              Ker/CoImg/CoKer f ⇒ ProjectedBasis(f, which, …)
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
]


# ---------------------------------------------------------------------------
# Default (inner-product-orthonormal) basis.
# ---------------------------------------------------------------------------


def default_basis(space: S.Space) -> B.Basis:
    """The default inner-product-orthonormal "standard" basis for ``space`` (`41 §1`)."""
    match space:
        case S.Scalar():
            return B.scalar_basis()
        case S.Base(name, _):
            return B.base_basis(name)
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
        case S.Img(f, _, _):
            return B.image_basis(f)
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
        case _:  # pragma: no cover - defensive
            raise S.SpaceError(f"space_basis: unknown space {space!r}")


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
            # the image basis IS the SVD basis of f; also the codomain's standard basis.
            _add(out, B.image_basis(f))
            if tgt is not None:
                _add(out, default_basis(tgt))
        case S.Ker(f, src, tgt) | S.CoImg(f, src, tgt):
            if src is not None:
                _add(out, default_basis(src))
        case S.CoKer(f, src, tgt):
            if tgt is not None:
                _add(out, default_basis(tgt))
        case _:
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
) -> FfsRealization:
    """Realize an FFS basis (``which`` ∈ ``"img"|"coker"|"coimg"|"ker"``) (`41 §1`).

    Emits the SVD/QR/GSVD ``Stmt``(s) into ``program`` and returns an
    :class:`FfsRealization`. Selection per `41 §1`/`50 §A1`:

    * **injective + ``which == "img"``** ⇒ :class:`QrOp` — ``Q`` is the orthonormal range
      basis with a *static* ``rank = dom_dim`` (recovers the dropped ``IsoSpace``);
    * **metric present** ⇒ :class:`GSvdOp` (orthonormal in the metrics);
    * otherwise ⇒ :class:`SvdOp`, with the runtime rank ``δ_f`` sizing a dynamic axis.
    """
    if which not in ("img", "coker", "coimg", "ker"):
        raise ValueError(f"realize_ffs_basis: bad which {which!r}")

    injective = props is not None and props.inj
    has_metric = metric_dom is not None or metric_cod is not None

    if which == "img" and injective and not has_metric:
        return _realize_via_qr(program, f_matrix, dom_dim, label)
    if has_metric:
        return _realize_via_gsvd(
            program, which, f_matrix, dom_dim, cod_dim,
            metric_dom, metric_cod, label,
        )
    return _realize_via_svd(program, which, f_matrix, dom_dim, cod_dim, label)


def _realize_via_qr(
    program: pa.Program, f_matrix: pa.SymArray, dom_dim: int, label: str,
) -> FfsRealization:
    # f : (cod, dom); injective ⇒ Q has dom orthonormal columns (the range basis).
    Q, _R = program.emit_stmt(
        pa.QrOp(mode="reduced"),
        [f_matrix],
        [pa.OutSpec("Q_" + label, _qr_q_shape(f_matrix, dom_dim)),
         pa.OutSpec("R_" + label, (dom_dim, dom_dim))],
        note=f"ffs-basis:ImageBasis({label}):inj=>QrOp",
    )
    return FfsRealization(cols=Q, rank=None, rank_dim=None, op="QrOp")


def _qr_q_shape(f_matrix: pa.SymArray, dom_dim: int) -> tuple[int, ...]:
    shp = _shape_of(f_matrix)
    return (shp[0], dom_dim)


def _realize_via_svd(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
    dom_dim: int, cod_dim: int, label: str,
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
    delta = pa.DimAtom(name=f"rank:{label}", source=(svd_idx, 3))
    cols = _take_ffs_block(program, which, U, Vh, rank, delta, cod_dim, dom_dim, label)
    return FfsRealization(cols=cols, rank=rank, rank_dim=delta, op="SvdOp")


def _realize_via_gsvd(
    program: pa.Program, which: str, f_matrix: pa.SymArray,
    dom_dim: int, cod_dim: int,
    metric_dom: pa.SymArray | None, metric_cod: pa.SymArray | None, label: str,
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
        comp = pa.DimAtom(name=f"comp(rank:{label},{ambient})", source=(gsvd_idx, 3))
        [block] = program.emit_stmt(
            _LastColsOp(), [factor, rank],
            [pa.OutSpec("ffs_" + which + "_" + label, (ambient, comp))],
            note=f"ffs-basis(metric):{which}=last-(n-δ)({label})",
        )
    return FfsRealization(cols=block, rank=rank, rank_dim=delta, op="GSvdOp")


def _take_ffs_block(
    program: pa.Program, which: str, U: pa.SymArray, Vh: pa.SymArray,
    rank: pa.SymArray, delta: pa.DimAtom, cod_dim: int, dom_dim: int, label: str,
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
        [cols] = program.emit_stmt(
            _LastColsOp(),
            [factor, rank],
            [pa.OutSpec("ffs_" + which + "_" + label,
                        (factor_rows(factor, ambient), _CompDim(delta, ambient)))],
            note=f"ffs-basis:{which}=last-(n-δ)({label})",
        )
    return cols


def factor_rows(factor: pa.SymArray, ambient: int) -> int:
    return ambient


# A complement dimension n-δ is itself dynamic; represent with a DimAtom keyed to a
# dedicated complement-rank Stmt would be cleaner, but for the substrate we re-key the
# same δ DimAtom and let the op produce the right (numeric) width at run time.  We wrap
# it in a fresh DimAtom name so is_dynamic sees a runtime axis.
def _CompDim(delta: pa.DimAtom, ambient: int) -> pa.DimAtom:  # noqa: N802
    return pa.DimAtom(name=f"comp({delta.name},{ambient})", source=delta.source)


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
