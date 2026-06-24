"""Term→Term transformation passes (a new pass category).

Unlike :mod:`grassmann.infer` (which *types* a term) and :mod:`grassmann.lower`
(which lowers a term to :mod:`polyarray`), these passes rewrite a core
:class:`~grassmann.core.term.Term` into a new core ``Term``:

- :func:`pullback` — auto-transport a value into a target space, inserting
  projections/embeddings (and recursing structurally through ⊗/⊕/Dual).
- :func:`diff` — total covariant differentiation w.r.t. a single value variable.
- :func:`simplify` / :func:`sum_of_products` — β-reduction, algebraic simplification,
  and a sum-of-products normal form (see :mod:`grassmann.transform.rewrite`).
"""

from __future__ import annotations

from .diff import DiffError, diff
from .pullback import PullbackError, pullback, transport_map
from .rewrite import (
    DEFAULT_SOP,
    RewriteError,
    SoPConfig,
    beta,
    is_normalized_sop,
    normal_form,
    rewrite,
    simplify,
    subst,
    sum_of_products,
    term_key,
)

__all__ = [
    "pullback",
    "transport_map",
    "PullbackError",
    "diff",
    "DiffError",
    # rewriting
    "rewrite",
    "subst",
    "beta",
    "simplify",
    "sum_of_products",
    "normal_form",
    "is_normalized_sop",
    "term_key",
    "SoPConfig",
    "DEFAULT_SOP",
    "RewriteError",
]
