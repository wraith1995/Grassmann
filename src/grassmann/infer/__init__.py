"""The inference engine: what space a term lives in, and whether it is well typed.

Inference reads the core IR (:mod:`grassmann.core`) and never modifies it. It is
bidirectional: :func:`synth` reads a classifier off a term where one can be read off
(``Γ ⊢ e ⇒ s``), and :func:`check` pushes an expected classifier in where the term alone
does not determine it (``Γ ⊢ e ⇐ s``). The two directions call each other, and the
expected space flowing inward is what recovers implicit parameters — the target of an
unannotated projection, the space of a bare ``Zero``.

Five kinds of reasoning meet in the middle::

    synth ⇒ / check ⇐        the two directions, over the closed Term sum
        │
        ├── unify            first-order unification of space metavariables ?α
        ├── dims             |s| and rank(f) — static, or the runtime atom δ_f
        ├── subspace         the ⊑ relation Project and Embed rest on
        ├── properties       asserted inj/surj/bij/rank ⇒ the four FFS dimensions
        └── linearity        exactly-once resource accounting for the ⊸ binder

Only spaces are unified. Dimensions are compared and never solved for, so a map asserted
bijective between spaces of unequal dimension is a contradiction rather than a constraint,
and a rank nobody asserted stays the opaque atom ``δ_f`` all the way to lowering rather
than being guessed. Asserted map properties are trusted facts throughout: they set the
dimensions of the four fundamental subspaces, license ``Solve``, and sharpen ``Pinv``.
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
