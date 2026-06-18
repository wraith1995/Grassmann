"""Bidirectional checking ``Γ ⊢ e ⇐ s`` (`25 §4`).

Default rule: synthesize ``e ⇒ s'`` then unify ``s' ~ s``. Propagation rules:

- a ``Lam`` checked against ``Π(b:σ).τ`` pushes ``σ`` in and checks ``body ⇐ τ``;
- an application checked against an expected ``s`` synthesizes and unifies the result
  with ``s`` to **recover implicit params that appear only in the result** (`25 §4`).

``check_against`` accepts either a Space or a Type as the expected classifier.
"""

from __future__ import annotations

from grassmann.core.space import Space, is_space
from grassmann.core.term import Lam, Term, is_term
from grassmann.core.types import Pi, SpaceAsType, Type, is_type

from .synth import Ctx, InferError, _check_subspace_obligations, synth
from .unify import unify


def check(term: Term, expected: Space | Type, ctx: Ctx | None = None) -> None:
    """Public ``⇐`` entry point. Raises on a mismatch (`25 §4`)."""
    top = ctx is None
    if ctx is None:
        ctx = Ctx()
    check_against(term, expected, ctx)
    if top:
        # B1: validate any deferred Project(v, None) obligations now that the expected
        # space has flowed in and solved their target metavars.
        _check_subspace_obligations(ctx)


def check_against(term: object, expected: Space | Type, ctx: Ctx) -> None:
    if not is_term(term):
        raise InferError(f"check: expected a Term, got {term!r}")

    # Lam ⇐ Π : push the domain in and check the body against the codomain.
    if isinstance(term, Lam) and isinstance(expected, Pi):
        _check_lam_against_pi(term, expected, ctx)
        return

    # Default: synthesize, then unify the result with the expected classifier.
    got = synth(term, ctx)

    if is_space(expected):
        if not is_space(got):
            # A value was expected but a function/Π was produced.
            if isinstance(got, SpaceAsType):
                got = got.space
            else:
                raise InferError(
                    f"check: expected value of space {expected!r}, got {got!r}"
                )
        unify(got, expected, ctx.store)  # type: ignore[arg-type]
        return

    if is_type(expected):
        _check_type_eq(got, expected, ctx)
        return

    raise InferError(f"check: expected a Space or Type, got {expected!r}")


def _check_lam_against_pi(term: Lam, expected: Pi, ctx: Ctx) -> None:
    # v1: the binders' domains must structurally agree with the Π's; then check body.
    # We synthesize the Lam and unify against the full Π (handles the common nested case
    # for these examples). A deeper push-in is unnecessary while codomains are
    # value-independent (`25` note in §3/§4).
    got = synth(term, ctx)
    _check_type_eq(got, expected, ctx)


def _check_type_eq(got: object, expected: Type, ctx: Ctx) -> None:
    """Structural type equality with space unification at the leaves (`25 §4`)."""
    match expected:
        case SpaceAsType():
            if isinstance(got, SpaceAsType):
                unify(got.space, expected.space, ctx.store)
                return
            if is_space(got):
                unify(got, expected.space, ctx.store)  # type: ignore[arg-type]
                return
            raise InferError(f"check: expected {expected!r}, got {got!r}")
        case Pi():
            if not isinstance(got, Pi):
                raise InferError(f"check: expected a Π {expected!r}, got {got!r}")
            if got.domain != expected.domain:
                raise InferError(
                    f"check: Π domain mismatch {got.domain!r} vs {expected.domain!r}"
                )
            _check_type_eq(got.codomain, expected.codomain, ctx)
            return
        case _:
            # TypeApp: require structural equality.
            if got != expected:
                raise InferError(f"check: type mismatch {got!r} vs {expected!r}")


__all__ = ["check", "check_against"]
