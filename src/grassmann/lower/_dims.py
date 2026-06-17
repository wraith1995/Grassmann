"""Static-dimension resolution for spaces and bases (`41 §0`).

A small helper shared by :mod:`space_basis` and :mod:`cob`: turn a structural
:class:`~grassmann.core.space.Space` (or :class:`~grassmann.core.basis.Basis`) into a
concrete ``int`` dimension *when statically known*, raising :class:`DynamicDim` when the
size is a runtime rank ``δ_f`` (a :class:`~grassmann.core.space.DimAtomRef`) or otherwise
symbolic. The substrate emits polyarray :class:`~polyarray.DimAtom`-sized (dynamic) axes
in those cases, so callers catch :class:`DynamicDim` and route to the bulk/dynamic lane.

This is *structural* dimension arithmetic only (the IW3 dim machinery does the real
inference); it suffices for the substrate's static numeric tests and for sizing the
``OutSpec``s of the FFS realizations.
"""

from __future__ import annotations

from grassmann.core import basis as B
from grassmann.core import space as S


class DynamicDim(Exception):
    """Raised when a space/basis dimension is not a static ``int`` (a runtime ``δ_f``)."""


def dim_of(space: S.Space) -> int:
    """Return the static integer dimension of ``space`` or raise :class:`DynamicDim`."""
    match space:
        case S.Scalar():
            return 1
        case S.Base(_, d):
            return _dim_int(d)
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
            # FFS dims are runtime ranks unless source/target + props pin them; the
            # substrate sizes these with a polyarray DimAtom (dynamic axis).
            raise DynamicDim(f"FFS space {space!r} has a runtime (δ_f) dimension")
        case S.SVar() | S.App():
            raise DynamicDim(f"space {space!r} has no static dimension")
        case _:  # pragma: no cover - defensive
            raise DynamicDim(f"cannot statically size {space!r}")


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
        case S.DimOf(s):
            return dim_of(s)
        case S.DimVar() | S.DimAtomRef():
            raise DynamicDim(f"dim {d!r} is symbolic / runtime")
        case _:  # pragma: no cover - defensive
            raise DynamicDim(f"cannot statically evaluate dim {d!r}")


def basis_dim(b: B.Basis) -> int:
    """Return the static integer dimension of a basis or raise :class:`DynamicDim`."""
    match b:
        case B.ScalarBasis():
            return 1
        case B.BaseBasis(_, vectors):
            if vectors is None:
                raise DynamicDim(f"BaseBasis {b!r} has no explicit vectors to size")
            import numpy as np

            # columns of `vectors` are the basis vectors ⇒ dim = #columns.
            return int(np.asarray(vectors, dtype=float).shape[1])
        case B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _):
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
        case B.CompleteBasis(_, _):
            # ambient dim = sub + complement; complement is a realized matrix.
            import numpy as np

            comp = np.asarray(b.complement, dtype=float)
            return basis_dim(b.sub_basis) + int(comp.shape[1] if comp.ndim == 2 else comp.shape[0])
        case B.ImageBasis() | B.ProjectedBasis() | B.BVar() | B.BLam() | B.BApp():
            raise DynamicDim(f"basis {b!r} has a runtime/symbolic dimension")
        case _:  # pragma: no cover - defensive
            raise DynamicDim(f"cannot statically size basis {b!r}")
