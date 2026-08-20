"""Term-to-term transformation passes.

Where :mod:`grassmann.infer` types a term and :mod:`grassmann.lower` lowers it to
:mod:`polyarray`, these passes rewrite a core
:class:`~grassmann.core.term.Term` into a new core term:

- :func:`pullback` — auto-transport a value into a target space, inserting
  projections/embeddings (and recursing structurally through ⊗/⊕/Dual).
- :func:`pushforward` / :func:`transport_basis` — the map-driven dual of
  ``pullback``: carry a value / basis *along* a chart map into its image
  (domain → image), the rational "transport" primitive (see
  :mod:`grassmann.transform.transport`).
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
    RIESZ_SOP,
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
from .transport import (
    TransportError,
    assert_rational,
    pull_along,
    pushforward,
    transport_basis,
    transport_cobasis,
)

__all__ = [
    "pullback",
    "transport_map",
    "PullbackError",
    "pushforward",
    "pull_along",
    "transport_basis",
    "transport_cobasis",
    "assert_rational",
    "TransportError",
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
    "RIESZ_SOP",
    "RewriteError",
]
