"""The lowering boundary: where a coordinate-free term meets a basis and becomes arrays.

Everything above this package is coordinate-free — no space in a term has a basis until
lowering gives it one. This package is where that happens, and it is the only place in
grassmann where a basis is realized::

    a Space  ──space_basis──►  its default basis and the candidates
                                      │
                                    choose        scores them against the term
                                      ▼
                                 one basis
                                      │
                                  represent       walks the term, emitting Stmts
                                      │
                                     cob          converts between two bases of one space
                                      ▼
                            a polyarray.Program

:func:`space_basis` answers what bases a space has: its default, the standard basis
orthonormal in the space's own inner product, so the metric is the identity in
coordinates, together with the alternatives worth weighing. :func:`choose` picks among
them, preferring the basis that leaves the fewest symbolic cells in the result.
:func:`cob` builds the change-of-basis matrix between two bases of the same space, which
is the only conversion grassmann admits — two bases of *different* spaces are an error,
never a silently inserted isomorphism. :func:`represent` drives all of it, and the
per-node rules live with it in :mod:`grassmann.lower.represent`.

:func:`realize_ffs_basis` and :func:`realize_basis` are the seam beneath ``cob``: the
point at which a fundamental subspace stops being a space and becomes concrete columns,
produced by an SVD, a QR or a generalized SVD of the map that defines it.
"""

from __future__ import annotations

from .choose import choose, score_basis
from .represent import (
    CobError,
    Rep,
    RepresentError,
    cob,
    basis_cols_program,
    numpy_op_renderers,
    realize_basis,
    represent,
)
from .space_basis import (
    FfsRealization,
    candidate_bases,
    default_basis,
    realize_ffs_basis,
    space_basis,
)

__all__ = [
    "space_basis",
    "default_basis",
    "candidate_bases",
    "cob",
    "realize_basis",
    "choose",
    "score_basis",
    "realize_ffs_basis",
    "FfsRealization",
    "CobError",
    "represent",
    "basis_cols_program",
    "numpy_op_renderers",
    "Rep",
    "RepresentError",
]
