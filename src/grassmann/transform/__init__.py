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

from .diff import (
    DerivInput,
    DiffError,
    clear_deriv_inputs,
    diff,
    enumerate_derivative_inputs,
)
from .pullback import PullbackError, pullback, transport_map
from .rewrite import (
    AGGRESSIVE_SOP,
    DEFAULT_SOP,
    RewriteError,
    SoPConfig,
    alpha_canon,
    beta,
    is_normalized_sop,
    normal_form,
    normalize_space,
    rewrite,
    simplify,
    space_key,
    spaces_iso,
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
    "enumerate_derivative_inputs",
    "DerivInput",
    "clear_deriv_inputs",
    # rewriting
    "rewrite",
    "subst",
    "beta",
    "simplify",
    "sum_of_products",
    "normal_form",
    "is_normalized_sop",
    "alpha_canon",
    "normalize_space",
    "space_key",
    "spaces_iso",
    "term_key",
    "SoPConfig",
    "DEFAULT_SOP",
    "AGGRESSIVE_SOP",
    "RewriteError",
]
