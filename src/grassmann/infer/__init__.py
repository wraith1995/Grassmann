"""The inference engine: bidirectional type synthesis and checking.

This engine types a term. It works bidirectionally — synthesizing a type where
one can be read off, checking against an expected type elsewhere — with
first-order unification over space terms, symbolic reasoning about dimensions
(static, or the runtime rank atom ``δ_f`` when a rank is unknown), the subspace
relation ``⊑``, and propagation of asserted map properties and the four
fundamental subspace dimensions. It reads the core IR
(:mod:`grassmann.core`) and does not modify it.

Public entry points
-------------------
- :func:`synth` / :func:`synth_space` — ``Γ ⊢ e ⇒ s`` (synthesize a Space/Type/Kind).
- :func:`check` — ``Γ ⊢ e ⇐ s`` (check against an expected Space/Type).
- :func:`unify` — first-order unification on space terms (mutates a :class:`MetaStore`).
- :func:`dim_of` / :func:`dim_eq` / :func:`rank_of` — symbolic dimensions + ``δ_f``.
- :func:`is_subspace` — the ``⊑`` relation.
- :func:`ffs_dims` — FFS dimensions from asserted props / runtime rank.
"""

from __future__ import annotations

from .assertions import (
    AssertFactError,
    DimFact,
    assert_property_dims,
    check_assert_property_consistency,
    consistent_dim_facts,
    map_facts_for_unify,
)
from .check import check, check_against
from .dims import (
    DimError,
    dim_eq,
    dim_of,
    is_runtime_dim,
    rank_of,
    space_mentions_runtime_dim,
)
from .linearity import (
    LinearityError,
    Usage,
    check_linearity,
    check_llam_resources,
)
from .properties import (
    FFSDims,
    ffs_dim,
    ffs_dims,
    is_bijective,
    is_injective,
    is_surjective,
    map_props,
)
from .subspace import (FfsKind, ffs_defining_map, ffs_kind, ffs_superspace, is_ffs,
                       is_subspace, split_map_space)
from .synth import Ctx, InferError, SynthResult, map_domain_codomain, synth, synth_space
from .unify import AssociationMismatch, MetaStore, SpaceMeta, UnifyError, fresh_meta, unify

__all__ = [
    # synthesis / checking
    "synth",
    "synth_space",
    "check",
    "check_against",
    "Ctx",
    "SynthResult",
    "InferError",
    "map_domain_codomain",
    # FFS / map-space probes (canonical)
    "is_ffs",
    "ffs_defining_map",
    "ffs_kind",
    "ffs_superspace",
    "split_map_space",
    "FfsKind",
    # unification
    "unify",
    "UnifyError",
    "AssociationMismatch",
    "MetaStore",
    "SpaceMeta",
    "fresh_meta",
    # dimensions
    "dim_of",
    "dim_eq",
    "rank_of",
    "is_runtime_dim",
    "space_mentions_runtime_dim",
    "DimError",
    # subspace
    "is_subspace",
    # properties / FFS
    "map_props",
    "is_bijective",
    "is_injective",
    "is_surjective",
    "ffs_dims",
    "ffs_dim",
    "FFSDims",
    # linearity (resource calculus)
    "LinearityError",
    "Usage",
    "check_linearity",
    "check_llam_resources",
    # asserted-fact unification aids
    "DimFact",
    "AssertFactError",
    "assert_property_dims",
    "consistent_dim_facts",
    "check_assert_property_consistency",
    "map_facts_for_unify",
]
