"""Grassmann inference / elaboration engine (IW3).

Bidirectional type synthesis + checking over the core IR, with first-order unification on
SPACE terms only, symbolic dimension reasoning (static OR the runtime SVD atom ``δ_f``),
the subspace relation ``⊑``, and asserted map-property / FFS-dimension propagation.

Spec: ``plans/25-embedding-inference.md`` (§0–§7). Operates over the IW1 core
(``grassmann.core``); does not modify it.

Public entry points
-------------------
- :func:`synth` / :func:`synth_space` — ``Γ ⊢ e ⇒ s`` (synthesize a Space/Type/Kind).
- :func:`check` — ``Γ ⊢ e ⇐ s`` (check against an expected Space/Type).
- :func:`unify` — first-order unification on space terms (mutates a :class:`MetaStore`).
- :func:`dim_of` / :func:`dim_eq` / :func:`rank_of` — symbolic dimensions + ``δ_f``.
- :func:`is_subspace` — the ``⊑`` relation (`25 §6`).
- :func:`ffs_dims` — FFS dimensions from asserted props / runtime rank (`25 §7`).
"""

from __future__ import annotations

from .check import check, check_against
from .dims import (
    DimError,
    dim_eq,
    dim_of,
    is_runtime_dim,
    rank_of,
    space_mentions_runtime_dim,
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
from .subspace import is_subspace
from .synth import Ctx, InferError, SynthResult, map_domain_codomain, synth, synth_space
from .unify import MetaStore, SpaceMeta, UnifyError, fresh_meta, unify

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
    # unification
    "unify",
    "UnifyError",
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
]
