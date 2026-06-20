"""``cob`` — change-of-basis between two bases of the same space (`41 §2`).

``cob(program, from_basis, to_basis) -> SymArray`` emits ``Stmt``s into ``program`` and
returns the change-of-basis matrix ``M`` such that, for a coordinate vector ``x`` in
``from_basis``, ``M @ x`` is the same vector's coordinates in ``to_basis``.

Port of FEMOracle ``convertBasisTowardsStandard``: realize each basis toward the ambient
*standard* coordinates (a matrix ``R_b`` whose columns are ``b``'s vectors in standard
coords), then compose — ``cob(from→to) = R_to⁻¹ · R_from`` (`41 §2`, impl shape "convert
toward standard, compose, then out"). The §2 structural rules are realized by recursion:

```
BasisProd(bᵢ)→BasisProd(b'ᵢ)  ⇒ Kron/Einsum of cob(bᵢ→b'ᵢ)
BasisSum(bᵢ) →BasisSum(b'ᵢ)   ⇒ block-diagonal of cob(bᵢ→b'ᵢ)
DualBasis b  →DualBasis b'    ⇒ (cob(b→b'))⁻ᵀ
BasisRep(n,b)                 ⇒ block-repeat of cob(b→b')
ProjectedBasis/ImageBasis     ⇒ from the SVD/QR factors (§1)
CompleteBasis(sub, comp)      ⇒ [ realize(sub) | realize(comp) ] stacked
```

**Min-sym (`41 §1b`):** these matrices are *numeric* ``Stmt``s — a well-chosen basis
keeps the cob numeric so symbolics never enter the change-of-basis; and an identical
``from``/``to`` short-circuits to the identity (no spurious cob, the "parametric in/out
⇒ same basis" lever).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polyarray as pa

from grassmann.core import basis as B

from ._dims import DynamicDim, basis_dim
from .space_basis import _shape_of

__all__ = ["cob", "realize_basis", "CobError"]


class CobError(Exception):
    """Raised when two bases cannot be related by a change of basis."""


def _numeric_dim(metric: object) -> int | None:
    """The dimension ``n`` of a numeric ``(n, n)`` metric, or ``None`` if not numeric."""
    try:
        a = np.asarray(metric, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        return int(a.shape[0])
    return None


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def cob(
    program: pa.Program, from_basis: B.Basis, to_basis: B.Basis,
) -> pa.SymArray:
    """Change-of-basis matrix ``M`` (from_basis-coords → to_basis-coords) (`41 §2`).

    Emits the matrix-building ``Stmt``s into ``program`` and returns the resulting
    :class:`~polyarray.SymArray`. Identical bases short-circuit to the identity.
    """
    if _basis_eq(from_basis, to_basis):
        return _identity(program, basis_dim(from_basis), "cob:id")

    # Structural §2 rules where both sides share the same outer constructor.
    match (from_basis, to_basis):
        case (B.BasisProd(fs), B.BasisProd(ts)) if len(fs) == len(ts):
            return _cob_prod(program, fs, ts)
        case (B.BasisSum(fs), B.BasisSum(ts)) if len(fs) == len(ts):
            return _cob_sum(program, fs, ts)
        case (B.DualBasis(fb), B.DualBasis(tb)):
            inner = cob(program, fb, tb)
            return _inv_transpose(program, inner)
        case (B.BasisRep(fn, fb), B.BasisRep(tn, tb)) if fn == tn:
            inner = cob(program, fb, tb)
            return _block_repeat(program, inner, _count_int(fn))
        # A ``WithInnerBasis`` shares its inner basis's *vectors* (the metric is metadata,
        # not a coordinate change), so a change of basis to/from it strips the wrapper —
        # e.g. relating a tangent-space value (metricized) to its base coordinates is the
        # identity, not a (generally unrealizable) abstract-basis cob. When the strip makes
        # the two sides equal we size the identity from the (numeric) metric, since the inner
        # basis of an *abstract* Base carries no explicit dimension.
        case (B.WithInnerBasis(fb, fg), _):
            inner_to = to_basis.inner if isinstance(to_basis, B.WithInnerBasis) else to_basis
            n = _numeric_dim(fg)
            if n is not None and _basis_eq(fb, inner_to):
                return _identity(program, n, "cob:wib-id")
            return cob(program, fb, to_basis)
        case (_, B.WithInnerBasis(tb, tg)):
            n = _numeric_dim(tg)
            if n is not None and _basis_eq(from_basis, tb):
                return _identity(program, n, "cob:wib-id")
            return cob(program, from_basis, tb)

    # General case: realize both toward standard coords and compose R_to⁻¹ · R_from.
    r_from = realize_basis(program, from_basis)
    r_to = realize_basis(program, to_basis)
    return _compose_via_standard(program, r_from, r_to, from_basis, to_basis)


# ---------------------------------------------------------------------------
# Realize a basis as a matrix of its vectors in the ambient standard coords.
# ---------------------------------------------------------------------------


def realize_basis(program: pa.Program, b: B.Basis) -> pa.SymArray:
    """Matrix whose columns are ``b``'s vectors in the ambient standard coords (`41 §2`).

    For an orthonormal standard basis this is the identity (the min-sym default). For a
    :class:`CompleteBasis` it is ``[ realize(sub) | complement ]`` stacked. Concrete
    :class:`BaseBasis` carries explicit vectors. Structured bases compose by the §2 rules
    (Kron for products, block-diag for sums, block-repeat for reps).
    """
    match b:
        case B.ScalarBasis():
            return _identity(program, 1, "realize:scalar")
        case B.BaseBasis(name, vectors):
            # Term-frame vectors / a backup-completed basis need ``_rep`` (the realization seam
            # in represent) — the pure-matrix path can't build them.
            if isinstance(vectors, tuple) or b.backup is not None:
                raise CobError(
                    f"realize_basis: BaseBasis({name!r}) is a term-frame / backup-completed "
                    "basis; realized via represent.realize_basis_cols, not as a static matrix"
                )
            if vectors is None:
                # The standard basis of a Base is the ambient identity; size it from a declared
                # ``space`` if present, else defer (cob short-circuits / the companion sizes it).
                try:
                    return _identity(program, basis_dim(b), "realize:base-std:" + name)
                except DynamicDim as exc:
                    raise CobError(
                        f"realize_basis: BaseBasis({name!r}) has no explicit vectors and no "
                        "static space; it IS the standard basis (identity) — sized from context"
                    ) from exc
            mat = np.asarray(vectors, dtype=float)
            if mat.ndim != 2:
                raise CobError(f"BaseBasis vectors must be a 2-D matrix, got {mat.shape}")
            # Convention: columns of `vectors` are the basis vectors in ambient coords.
            return _const(program, mat, "realize:base:" + name)
        case B.DualBasis(inner):
            # dual realization is (realize(inner))⁻ᵀ.
            r = realize_basis(program, inner)
            return _inv_transpose(program, r)
        case B.WithInnerBasis(inner, _):
            return realize_basis(program, inner)
        case B.BasisProd(bases):
            mats = [realize_basis(program, x) for x in bases]
            return _kron(program, mats)
        case B.BasisSum(bases):
            mats = [realize_basis(program, x) for x in bases]
            return _block_diag(program, mats)
        case B.BasisRep(count, inner):
            r = realize_basis(program, inner)
            return _block_repeat(program, r, _count_int(count))
        case B.CompleteBasis(sub_basis, complement):
            sub = realize_basis(program, sub_basis)
            comp = _const(program, np.asarray(complement, dtype=float), "realize:complement")
            return _hstack(program, [sub, comp])
        case B.ImageBasis() | B.ProjectedBasis():
            raise CobError(
                f"realize_basis: FFS basis {b!r} is realized via space_basis."
                "realize_ffs_basis (SvdOp/QrOp/GSvdOp), not as a static matrix"
            )
        case _:
            raise CobError(f"realize_basis: cannot realize {b!r}")


# ---------------------------------------------------------------------------
# Structural §2 combinators (each as a numeric Stmt).
# ---------------------------------------------------------------------------


def _cob_prod(program, fs, ts) -> pa.SymArray:
    """BasisProd → BasisProd: Kron of the per-factor cobs (`41 §2`)."""
    factor_cobs = [cob(program, f, t) for f, t in zip(fs, ts, strict=True)]
    return _kron(program, factor_cobs)


def _cob_sum(program, fs, ts) -> pa.SymArray:
    """BasisSum → BasisSum: block-diagonal of the per-summand cobs (`41 §2`)."""
    block_cobs = [cob(program, f, t) for f, t in zip(fs, ts, strict=True)]
    return _block_diag(program, block_cobs)


def _compose_via_standard(program, r_from, r_to, from_basis, to_basis) -> pa.SymArray:
    """cob = R_to⁻¹ · R_from (realize both toward standard, compose) (`41 §2`)."""
    [out] = program.emit_stmt(
        _ComposeViaStdOp(),
        [r_to, r_from],
        [pa.OutSpec("cob", (basis_dim(to_basis), basis_dim(from_basis)))],
        note=f"cob:R_to^-1·R_from ({_short(from_basis)}→{_short(to_basis)})",
    )
    return out


# ---------------------------------------------------------------------------
# Matrix-building numeric ops (frozen, hashable, numpy __call__).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ConstOp:
    key: str  # provenance only; the data rides as a captured numpy array
    data_bytes: bytes
    shape: tuple[int, ...]
    dtype: str

    def __call__(self) -> np.ndarray:
        return np.frombuffer(self.data_bytes, dtype=self.dtype).reshape(self.shape).copy()


@dataclass(frozen=True)
class _IdentityMatOp:
    n: int

    def __call__(self) -> np.ndarray:
        return np.eye(self.n)


@dataclass(frozen=True)
class _InvTransposeOp:
    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.linalg.inv(np.asarray(A, dtype=float)).T


@dataclass(frozen=True)
class _ComposeViaStdOp:
    def __call__(self, R_to: np.ndarray, R_from: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.linalg.solve(np.asarray(R_to, dtype=float), np.asarray(R_from, dtype=float))


@dataclass(frozen=True)
class _KronOp:
    n: int

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        out = np.asarray(mats[0], dtype=float)
        for m in mats[1:]:
            out = np.kron(out, np.asarray(m, dtype=float))
        return out


@dataclass(frozen=True)
class _BlockDiagOp:
    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        arrs = [np.asarray(m, dtype=float) for m in mats]
        rows = sum(a.shape[0] for a in arrs)
        cols = sum(a.shape[1] for a in arrs)
        out = np.zeros((rows, cols))
        r = c = 0
        for a in arrs:
            out[r:r + a.shape[0], c:c + a.shape[1]] = a
            r += a.shape[0]
            c += a.shape[1]
        return out


@dataclass(frozen=True)
class _BlockRepeatOp:
    n: int

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        a = np.asarray(A, dtype=float)
        out = np.zeros((self.n * a.shape[0], self.n * a.shape[1]))
        for i in range(self.n):
            out[i * a.shape[0]:(i + 1) * a.shape[0], i * a.shape[1]:(i + 1) * a.shape[1]] = a
        return out


@dataclass(frozen=True)
class _HStackOp:
    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        return np.hstack([np.asarray(m, dtype=float) for m in mats])


# ---------------------------------------------------------------------------
# Emitters.
# ---------------------------------------------------------------------------


def _identity(program: pa.Program, n: int, note: str) -> pa.SymArray:
    [m] = program.emit_stmt(
        _IdentityMatOp(n), [], [pa.OutSpec("I", (n, n))], note=note,
    )
    return m


def _const(program: pa.Program, mat: np.ndarray, note: str) -> pa.SymArray:
    mat = np.ascontiguousarray(np.asarray(mat, dtype=float))
    op = _ConstOp(key=note, data_bytes=mat.tobytes(), shape=mat.shape, dtype=str(mat.dtype))
    [m] = program.emit_stmt(op, [], [pa.OutSpec("const", mat.shape)], note=note)
    return m


def _inv_transpose(program: pa.Program, mat: pa.SymArray) -> pa.SymArray:
    shp = _shape_of(mat)
    [m] = program.emit_stmt(
        _InvTransposeOp(), [mat], [pa.OutSpec("invT", shp)], note="cob:DualBasis=(cob)^-T",
    )
    return m


def _kron(program: pa.Program, mats: list[pa.SymArray]) -> pa.SymArray:
    if len(mats) == 1:
        return mats[0]
    shapes = [_shape_of(m) for m in mats]
    rows = 1
    cols = 1
    for r, c in shapes:
        rows *= r
        cols *= c
    [m] = program.emit_stmt(
        _KronOp(len(mats)), mats, [pa.OutSpec("kron", (rows, cols))],
        note="cob:BasisProd=Kron",
    )
    return m


def _block_diag(program: pa.Program, mats: list[pa.SymArray]) -> pa.SymArray:
    if len(mats) == 1:
        return mats[0]
    shapes = [_shape_of(m) for m in mats]
    rows = sum(s[0] for s in shapes)
    cols = sum(s[1] for s in shapes)
    [m] = program.emit_stmt(
        _BlockDiagOp(), mats, [pa.OutSpec("blockdiag", (rows, cols))],
        note="cob:BasisSum=block-diag",
    )
    return m


def _block_repeat(program: pa.Program, mat: pa.SymArray, n: int) -> pa.SymArray:
    r, c = _shape_of(mat)
    [m] = program.emit_stmt(
        _BlockRepeatOp(n), [mat], [pa.OutSpec("blockrep", (n * r, n * c))],
        note="cob:BasisRep=block-repeat",
    )
    return m


def _hstack(program: pa.Program, mats: list[pa.SymArray]) -> pa.SymArray:
    shapes = [_shape_of(m) for m in mats]
    rows = shapes[0][0]
    cols = sum(s[1] for s in shapes)
    [m] = program.emit_stmt(
        _HStackOp(), mats, [pa.OutSpec("hstack", (rows, cols))],
        note="cob:CompleteBasis=[sub|comp]",
    )
    return m


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------


def _count_int(count) -> int:
    try:
        return basis_dim(B.basis_rep(count, B.scalar_basis()))
    except DynamicDim as exc:  # pragma: no cover - defensive
        raise CobError(f"BasisRep count must be static, got {count!r}") from exc


def _short(b: B.Basis) -> str:
    return type(b).__name__


def _backup_eq(a: object, b: object) -> bool:
    """Equality for an optional BaseBasis ``backup`` (a Basis or ``None``)."""
    if a is None or b is None:
        return a is b
    return _basis_eq(a, b)


def _basis_eq(a: B.Basis, b: B.Basis) -> bool:
    """Structural basis equality that tolerates numpy ``vectors`` fields.

    A plain ``a == b`` raises ``ValueError`` when a :class:`BaseBasis` /
    :class:`CompleteBasis` carries a numpy array (ambiguous array truth value), so
    compare those array-bearing fields with :func:`numpy.array_equal`.
    """
    if type(a) is not type(b):
        return False
    match (a, b):
        case (B.BaseBasis(), B.BaseBasis()):
            # ``space`` is sizing metadata, not part of the coordinate identity — compare
            # name + vectors + backup (so a standard basis is equal with or without a space).
            if a.name != b.name or not _backup_eq(a.backup, b.backup):
                return False
            va, vb = a.vectors, b.vectors
            if va is None or vb is None:
                return va is vb
            if isinstance(va, tuple) or isinstance(vb, tuple):
                return va is vb  # term-frame vectors compare by identity (Terms)
            return np.array_equal(np.asarray(va, dtype=float), np.asarray(vb, dtype=float))
        case (B.CompleteBasis(sa, ca), B.CompleteBasis(sb, cb)):
            return _basis_eq(sa, sb) and np.array_equal(
                np.asarray(ca, dtype=float), np.asarray(cb, dtype=float)
            )
        case (B.BasisProd(fs), B.BasisProd(ts)) | (B.BasisSum(fs), B.BasisSum(ts)):
            return len(fs) == len(ts) and all(_basis_eq(x, y) for x, y in zip(fs, ts, strict=True))
        case (B.DualBasis(x), B.DualBasis(y)) | (B.PrimalBasis(x), B.PrimalBasis(y)):
            return _basis_eq(x, y)
        case (B.BasisRep(nx, x), B.BasisRep(ny, y)):
            return nx == ny and _basis_eq(x, y)
        case (B.WithInnerBasis(x, _), B.WithInnerBasis(y, _)):
            return _basis_eq(x, y)
        case _:
            try:
                return bool(a == b)
            except ValueError:
                return False
