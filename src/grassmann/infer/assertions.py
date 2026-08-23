"""What an assertion tells inference about dimensions, and how a contradiction is caught.

An :class:`~grassmann.core.term.Assert` node and the map properties carried on a map's
:class:`~grassmann.core.term.Var` — ``inj``, ``surj``, ``bij``, an explicit ``rank`` — are
trusted facts. Unifying ``Assert``'s space with the synthesized one already does the
obvious work of solving implicit space metavariables. The properties do something further:
each one guarantees a dimension equality that the space must satisfy::

    bij       ⇒  |dom| ≡ |cod|             the map is an isomorphism
    inj       ⇒  rank = |dom|,  so  |Img f| ≡ |dom|
    surj      ⇒  rank = |cod|,  so  |Img f| ≡ |cod|
    rank r    ⇒  δ_f is the static r       (resolved in grassmann.infer.dims)

Dimensions are compared and never unified, so these equalities cannot be *solved* — they
can only be *checked*. :func:`assert_property_dims` collects the ones a map's properties
guarantee, and :func:`check_assert_property_consistency` tests them against the space once
unification has resolved it. A map asserted bijective between spaces of demonstrably
unequal dimension is therefore rejected at the assertion, rather than accepted and left to
fail somewhere downstream.

Everything here is sound in one direction only: it reads what the user asserted and never
invents a property. A dimension that cannot be evaluated yet — one still mentioning an
unsolved metavariable — is not a demonstrable contradiction and is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

from grassmann.core.space import Dim, Space, is_space

from .dims import DimError, dim_eq, dim_of
from .properties import map_props
from .synth import InferError


@dataclass(frozen=True)
class DimFact:
    """One dimension equality an asserted property guarantees, with the reason it holds.

    ``why`` is prose naming the property the equality came from, so a violation can report
    which assertion it contradicts rather than only which dimensions disagreed.
    """

    lhs: Dim
    rhs: Dim
    why: str


def assert_property_dims(f: object, dom: Space, cod: Space) -> list[DimFact]:
    """Collect the dimension equalities ``f``'s asserted properties guarantee.

    ``f`` is a map term whose domain and codomain have already been decoded. A map with no
    usable assertion gives the empty list, which is the runtime-rank case: nothing static
    is known, so nothing static is claimed.
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
    # An explicit rank constrains both the kernel and the cokernel. When a map is asserted
    # bij/inj/surj and also given a rank, the two must agree, so the implied identities are
    # cross-checked here.
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
    """Give the facts that fail — those whose two sides are not dimension-equal.

    A non-empty result means an assertion contradicts a property and must be rejected. A
    fact whose dimensions cannot be evaluated, because one side still mentions an unsolved
    metavariable, is skipped rather than counted: only a demonstrable contradiction is a
    violation.
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
    """Raise when ``f``'s asserted properties contradict its resolved domain and codomain.

    The ``Assert`` and inverse rules apply this once unification has pinned the space down,
    so that a map asserted bijective between spaces of unequal computable dimension is
    refused at the assertion.
    """
    bad = consistent_dim_facts(assert_property_dims(f, dom, cod))
    if bad:
        first = bad[0]
        raise AssertFactError(
            f"asserted property is inconsistent with the space: {first.why} "
            f"but {first.lhs!r} ≢ {first.rhs!r}"
        )


class AssertFactError(InferError):
    """Raised when an asserted map property contradicts the space it was asserted against.

    It subclasses :class:`grassmann.infer.synth.InferError`, so a caller handling inference
    failures generically catches this one too, while a caller that cares can distinguish a
    contradicted assertion from an ordinary mismatch.
    """


def map_facts_for_unify(f: object) -> list[DimFact]:
    """Collect ``f``'s property-implied dimension facts, decoding its endpoints first.

    This is :func:`assert_property_dims` for a caller that holds only the map term: it
    synthesizes and decodes ``f``'s map space to find the domain and codomain. A term that
    is not a decodable map, or that carries no usable assertion, gives the empty list.
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
