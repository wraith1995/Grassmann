"""Term→Term transformation passes (a new pass category).

Unlike :mod:`grassmann.infer` (which *types* a term) and :mod:`grassmann.lower`
(which lowers a term to :mod:`polyarray`), these passes rewrite a core
:class:`~grassmann.core.term.Term` into a new core ``Term``:

- :func:`pullback` — auto-transport a value into a target space, inserting
  projections/embeddings (and recursing structurally through ⊗/⊕/Dual).
- :func:`diff` — total covariant differentiation w.r.t. a single value variable.
"""

from __future__ import annotations

from .diff import DiffError, diff
from .pullback import PullbackError, pullback, transport_map

__all__ = [
    "pullback",
    "transport_map",
    "PullbackError",
    "diff",
    "DiffError",
]
