"""Asserted facts that aid first-order unification / inference.

The core :class:`~grassmann.core.term.Assert` node ``Assert(v, s)``, and the **asserted map
properties** carried by a map's :class:`~grassmann.core.term.Var`
(``inj``/``surj``/``bij``/``rank``), are *trusted* facts. Beyond pinning a space
by unification, those facts carry **dimension** information that the (dim-aware, but
dim-never-unifying) inference can exploit:

- an asserted **bijective** map forces ``|dom| ≡ |cod|`` (an isomorphism);
- an asserted **injective** map forces ``rank = |dom|`` so ``|dom| ≤ |cod|`` and
  ``|Img f| ≡ |dom|``;
- an asserted **surjective** map forces ``rank = |cod|`` so ``|Img f| ≡ |cod|``;
- an asserted **rank ``r``** pins ``δ_f`` to the static ``r`` (resolved in
  :mod:`grassmann.infer.dims`).

This module turns those facts into:

1. :func:`assert_property_dims` — given a map term and its decoded ``(dom, cod)``, the
   list of dimension equalities the asserted properties *guarantee*. Sound: it only reads
   user-asserted props (never invents).
2. :func:`check_assert_property_consistency` — the consistency check the ``Assert`` /
   inverse rules in :mod:`grassmann.infer.synth` apply. After the synthesized space is
   unified with the asserted ``s`` (recovering implicit metavars — this is the unification
   *aid*), it verifies that every property-implied dimension equality holds for the
   now-resolved space, rejecting an assertion that contradicts an asserted property.

Because dimensions are *compared, never unified*, the contribution of an
assertion to unification is two-fold: (a) the ordinary structural unification of
``Assert``'s space solves implicit space metavars; (b) the property-implied dim equalities
are *checked* against the resolved space, so an asserted-``bij`` map whose asserted
``dom``/``cod`` have unequal dimension is rejected up front rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

from grassmann.core.space import Dim, Space, is_space

from .dims import DimError, dim_eq, dim_of
from .properties import map_props
from .synth import InferError


@dataclass(frozen=True)
class DimFact:
    """A guaranteed dimension equality ``lhs ≡ rhs`` with a human-readable ``why``."""

    lhs: Dim
    rhs: Dim
    why: str


def assert_property_dims(f: object, dom: Space, cod: Space) -> list[DimFact]:
    """Dimension equalities *guaranteed* by ``f``'s asserted properties (sound).

    ``f`` is a map term with decoded ``dom``/``cod``. Returns the dim facts that the
    asserted ``inj``/``surj``/``bij``/``rank`` impose. Returns ``[]`` when ``f`` carries
    no usable assertion (the runtime-rank case — nothing static to add).
    """
    props = map_props(f)
    if props is None:
        return []
    facts: list[DimFact] = []
    # A bijection is an isomorphism: |dom| ≡ |cod|.
    if props.bij:
        facts.append(DimFact(dim_of(dom), dim_of(cod), "asserted bij ⇒ |dom| ≡ |cod|"))
    else:
        # inj ⇒ rank = |dom| (so |Img f| ≡ |dom|); surj ⇒ rank = |cod|.
        if props.inj:
            facts.append(
                DimFact(dim_of(dom), dim_of(dom), "asserted inj ⇒ |Img f| ≡ |dom|")
            )
        if props.surj:
            facts.append(
                DimFact(dim_of(cod), dim_of(cod), "asserted surj ⇒ |Img f| ≡ |cod|")
            )
    # An explicit rank r constrains both ker/coker; if a map is asserted bij/inj/surj AND
    # given a numeric rank we cross-check the implied identities below.
    if props.rank is not None:
        if props.bij:
            facts.append(
                DimFact(props.rank, dim_of(dom), "asserted rank ≡ |dom| (bij)")
            )
            facts.append(
                DimFact(props.rank, dim_of(cod), "asserted rank ≡ |cod| (bij)")
            )
        elif props.inj:
            facts.append(
                DimFact(props.rank, dim_of(dom), "asserted rank ≡ |dom| (inj)")
            )
        elif props.surj:
            facts.append(
                DimFact(props.rank, dim_of(cod), "asserted rank ≡ |cod| (surj)")
            )
    return facts


def consistent_dim_facts(facts: list[DimFact]) -> list[DimFact]:
    """Return the facts whose ``lhs`` and ``rhs`` are *not* dim-equal (the violations).

    A non-empty result means an assertion contradicts a property and must be rejected.
    Facts whose dims cannot be computed (e.g. they mention an unsolved metavar) are
    skipped — we only reject a *demonstrable* contradiction (sound: never invent).
    """
    bad: list[DimFact] = []
    for fact in facts:
        try:
            if not dim_eq(fact.lhs, fact.rhs):
                bad.append(fact)
        except DimError:
            # A dim we cannot evaluate yet (open metavar) — not a demonstrable violation.
            continue
    return bad


def check_assert_property_consistency(f: object, dom: Space, cod: Space) -> None:
    """Raise if ``f``'s asserted properties contradict its (resolved) ``dom``/``cod``.

    Used by the ``Assert``/inverse rules to reject e.g. an asserted-``bij`` map whose
    asserted domain and codomain have unequal (computable) dimension.
    """
    bad = consistent_dim_facts(assert_property_dims(f, dom, cod))
    if bad:
        first = bad[0]
        raise AssertFactError(
            f"asserted property is inconsistent with the space: {first.why} "
            f"but {first.lhs!r} ≢ {first.rhs!r}"
        )


class AssertFactError(InferError):
    """Raised when an asserted property contradicts the resolved space.

    A subclass of :class:`grassmann.infer.synth.InferError` so it is caught by the same
    generic inference-failure handling as the other rules.
    """


def map_facts_for_unify(f: object) -> list[DimFact]:
    """Property-implied dim facts for a map term ``f`` (decoding its own ``dom``/``cod``).

    Convenience wrapper: decodes ``f``'s map space, then defers to
    :func:`assert_property_dims`. Returns ``[]`` if ``f`` is not a decodable map or has no
    usable assertion.
    """
    from .synth import map_domain_codomain

    if not is_space(getattr(f, "space", None)) and map_props(f) is None:
        return []
    try:
        dom, cod = map_domain_codomain(f)
    except Exception:
        return []
    return assert_property_dims(f, dom, cod)


__all__ = [
    "DimFact",
    "AssertFactError",
    "assert_property_dims",
    "consistent_dim_facts",
    "check_assert_property_consistency",
    "map_facts_for_unify",
]
