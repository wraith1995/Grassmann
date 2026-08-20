"""The lowering substrate: basis selection and change of basis.

This package is where coordinates are chosen. It holds the basis machinery the
per-node representation rules build on as a term is lowered to a
:mod:`polyarray` program: how a space picks its default basis and its
candidates, how one basis is changed into another, and which candidate to prefer.
The per-node rules themselves live in :mod:`grassmann.lower.represent`.

Public API:

* :func:`space_basis` — ``space_basis(space) -> (default_basis, candidates)``. The
  default is the standard basis orthonormal in the space's inner product (the
  metric is the identity in coordinates); the candidates feed :func:`choose`.
* :func:`cob` — ``cob(program, from_basis, to_basis) -> SymArray``, the
  change-of-basis matrix, emitting numeric statements into a polyarray program.
* :func:`choose` — the symbolic-aware basis-choice heuristic: score the
  candidates and pick the one that minimizes the symbolic surface of the result.

Also re-exported: :func:`realize_basis` and :func:`realize_ffs_basis` (the
subspace SVD/QR/GSVD seam), and :func:`default_basis`, :func:`candidate_bases`,
and :func:`score_basis`.
"""

from __future__ import annotations

from .choose import choose, score_basis
from .represent import (
    CobError,
    Rep,
    RepresentError,
    cob,
    frame_cols_program,
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
    "frame_cols_program",
    "numpy_op_renderers",
    "Rep",
    "RepresentError",
]
