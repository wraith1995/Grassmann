"""Grassmann lowering substrate (IW4) — basis selection + change of basis (`41 §1`/`§2`).

This package is the **substrate** of the Grassmann→polyarray lowering (`60 §1–§2`): the
basis machinery the per-node ``representDown`` rules build on. It does **not** contain the
per-node rules (``represent.py``) — those need IW3 and come later (`60`, batch 3).

Public API:

* :func:`space_basis` (`41 §1`) — ``space_basis(space) -> (default_basis, candidates)``;
  the default is the inner-product-orthonormal standard basis (metric = identity in
  coords), candidates feed the §1b choice.
* :func:`cob` (`41 §2`) — ``cob(program, from_basis, to_basis) -> SymArray``; the
  change-of-basis matrix, emitting numeric ``Stmt``s into a polyarray ``Program``.
* :func:`choose` (`41 §1b`) — the symbolic-aware basis-choice heuristic (the ★ min-sym
  lever): score candidates and pick the one minimizing the result's symbolic surface.

Also re-exported: :func:`realize_basis` / :func:`realize_ffs_basis` (the FFS SVD/QR/GSVD
seam) and :func:`default_basis` / :func:`candidate_bases` / :func:`score_basis`.
"""

from __future__ import annotations

from .choose import choose, score_basis
from .cob import CobError, cob, realize_basis
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
]
