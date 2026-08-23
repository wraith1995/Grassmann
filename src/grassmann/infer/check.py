"""The check direction of inference, ``Γ ⊢ e ⇐ s`` — pushing an expected classifier in.

Checking asks whether a term fits a classifier that the context already knows, which is
the direction that recovers information the term alone does not carry. The default rule
synthesizes ``e ⇒ s'`` and unifies ``s' ~ s``, and the unification is the point: an
implicit parameter appearing only in the result of an application is solved by meeting the
expected space there. A ``Lam`` checked against a ``Π(b:σ).τ`` instead pushes the domain
inward and checks the body against the codomain.

:func:`~grassmann.infer.synth.synth` runs the opposite direction. The two call each other,
so a term is typed by whichever direction has the information at each node. Both a Space
and a Type are accepted as the expected classifier; a Type is compared structurally, with
spaces unified at its leaves.
"""

from __future__ import annotations

from grassmann.core.space import Space, is_space
from grassmann.core.term import Lam, Term, is_term
from grassmann.core.types import Pi, SpaceAsType, Type, is_type

from .synth import Ctx, InferError, _check_subspace_obligations, synth
from .unify import unify


def check(term: Term, expected: Space | Type, ctx: Ctx | None = None) -> None:
    """Check ``term`` against the expected classifier, discharging deferred obligations.

    At the top level this also validates the subspace obligations that an unannotated
    ``Project`` deferred, now that the expected space has flowed in and solved their target
    metavariables.

    Parameters
    ----------
    term
        The term to check.
    expected
        The expected classifier — a Space or a Type.
    ctx
        Typing context; a fresh one is created at the top level.

    Raises
    ------
    InferError
        If ``term`` does not check against ``expected``.
    """
    top = ctx is None
    if ctx is None:
        ctx = Ctx()
    check_against(term, expected, ctx)
    if top:
        # Validate any deferred Project(v, None) obligations now that the expected
        # space has flowed in and solved their target metavars.
        _check_subspace_obligations(ctx)


def check_against(term: object, expected: Space | Type, ctx: Ctx) -> None:
    """Check ``term`` against ``expected`` in ``ctx``, leaving obligations for the caller.

    This is the rule the inference calls recursively: it applies the propagation rule for a
    ``Lam`` against a ``Π`` and otherwise synthesizes and unifies, but it does not discharge
    the deferred subspace obligations — :func:`check` does that once, at the top.

    Parameters
    ----------
    term
        Candidate term; a non-term raises.
    expected
        The expected classifier — a Space or a Type.
    ctx
        Typing context, carrying metavariable solutions.

    Raises
    ------
    InferError
        If ``term`` is not a Term, or does not check against ``expected``.
    """
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
    # The binder domains must agree structurally with the Π's, and then the body must check
    # against the codomain. Synthesizing the Lam and comparing against the whole Π achieves
    # both, including for nested binders; pushing the domain in one binder at a time buys
    # nothing while codomains do not depend on the values bound.
    got = synth(term, ctx)
    _check_type_eq(got, expected, ctx)


def _check_type_eq(got: object, expected: Type, ctx: Ctx) -> None:
    """Compare types structurally, unifying spaces at the leaves."""
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
