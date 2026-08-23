"""Passes that turn one core term into another, above the lowering boundary.

Inference types a term and lowering turns it into array code; these passes sit between,
rewriting a :class:`~grassmann.core.term.Term` into a different term that means the same
thing or means the derivative of it. They never realize a basis and never mutate the IR
they read — a pass returns a new term::

    Term ──▶ transform ──▶ Term          coordinate-free throughout
                 │
                 ├── diff        the total covariant derivative along one variable
                 ├── rewrite     β, simplification, sum-of-products normal form
                 ├── pullback    move a value between structurally related spaces
                 └── transport   carry a value or basis along a map, into its image

:mod:`~grassmann.transform.diff` differentiates a closed term with respect to a single
value variable, forward-mode, building the Levi-Civita connection where the metric moves.
:mod:`~grassmann.transform.rewrite` normalizes, from plain β-reduction up to a configurable
sum-of-products form. :mod:`~grassmann.transform.pullback` transports a value into a target
space using only the relations the spaces already have — subspace containment, a summand of
a direct sum, a factor of a tensor product — while
:mod:`~grassmann.transform.transport` does the map-driven counterpart, applying a chart map
and landing in its image.
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
