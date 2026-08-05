"""``choose`` — symbolic-aware basis-choice heuristic (`41 §1b`, the ★ min-sym lever).

``choose(space, candidates, *, term=None, in_basis=None)`` scores the candidate bases by
*expected symbolic content of the result* and returns the winner. This is the v1
**heuristic** (`41 §1b`): a global basis search is deferred.

The levers, as scored here (`41 §1b`):

* **Absorb operator structure into the basis.** A value derived from a symbolic operator
  ``f`` scores best in ``f``'s SVD / structure-aligned basis (``ImageBasis`` /
  ``ProjectedBasis``): there the projection is the identity and the coordinates are
  numeric. So an FFS basis of the *same* operator gets the top score.
* **Numeric result wins.** A candidate that makes the result numeric is preferred even
  at the cost of a (numeric) change-of-basis ``Stmt``.
* **Keep ``cob`` numeric.** The inner-product-orthonormal default makes the metric the
  identity and the cob numeric, so it is the baseline winner absent operator structure.
* **Parametric in/out ⇒ same basis.** If an ``in_basis`` is supplied and is among the
  candidates (an endomorphism / identified parametric spaces), prefer it — one basis for
  both sides, so no spurious cob (hence no spurious symbolics) appears.

The score is a small integer (higher = better); :func:`score_basis` exposes it for tests.
"""

from __future__ import annotations

from grassmann.core import basis as B
from grassmann.core import space as S

__all__ = ["choose", "score_basis"]


def choose(
    space: S.Space,
    candidates: tuple[B.Basis, ...],
    *,
    term: object | None = None,
    in_basis: B.Basis | None = None,
) -> B.Basis:
    """Pick the symbolic-minimizing basis from ``candidates`` for ``space`` (`41 §1b`).

    ``candidates`` is typically the second element of
    :func:`grassmann.lower.space_basis.space_basis`. ``term`` is the (optional) core-IR
    term being represented — used to recognize when a candidate is the SVD/structure
    basis of the *same* operator. ``in_basis`` (optional) is the basis already chosen for
    the input side, enabling the "parametric in/out ⇒ same basis" lever.
    """
    if not candidates:
        raise ValueError("choose: no candidate bases")

    best = candidates[0]
    best_score = score_basis(space, best, term=term, in_basis=in_basis)
    for cand in candidates[1:]:
        sc = score_basis(space, cand, term=term, in_basis=in_basis)
        if sc > best_score:
            best, best_score = cand, sc
    return best


def score_basis(
    space: S.Space,
    candidate: B.Basis,
    *,
    term: object | None = None,
    in_basis: B.Basis | None = None,
) -> int:
    """Heuristic score for ``candidate`` (higher = fewer expected symbolics) (`41 §1b`)."""
    score = 0

    # ── Parametric in/out ⇒ same basis: a candidate matching the input side wins big.
    if in_basis is not None and _eq(candidate, in_basis):
        score += 100

    # ── Absorb operator structure: an FFS/SVD basis of the operator behind `space`.
    op = _operator_of(space)
    if op is not None and _is_operator_basis_of(candidate, op):
        # numeric-result-wins: in the operator's own basis the coordinates are numeric.
        score += 40

    # ── Keep cob numeric / identity metric: the orthonormal standard default.
    #   The default is orthonormal (metric = identity), so it is the baseline.
    if _is_orthonormal_default(space, candidate):
        score += 10

    # ── g-orthonormal default for a metric space (identity metric ⇒ numeric Inner/dot).
    if isinstance(space, S.WithInner) and isinstance(candidate, B.WithInnerBasis):
        score += 20

    return score


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _operator_of(space: S.Space) -> object | None:
    """The linear-map term whose structure ``space`` is built from (an FFS), or None."""
    from grassmann.infer.subspace import ffs_defining_map

    return ffs_defining_map(space)


def _is_operator_basis_of(candidate: B.Basis, op: object) -> bool:
    match candidate:
        case B.ImageBasis(f):
            return f is op or f == op
        case B.ProjectedBasis(f, _, _, _):
            return f is op or f == op
        case _:  # audit: ok EX-WILDCARD — safe direction: False only skips a scoring bonus [approved 2026-07-03]
            return False


def _is_orthonormal_default(space: S.Space, candidate: B.Basis) -> bool:
    """True iff ``candidate`` is the inner-product-orthonormal standard basis of ``space``.

    Imported lazily to avoid a cycle with :mod:`space_basis`.
    """
    from .space_basis import default_basis

    try:
        return _eq(candidate, default_basis(space))
    except Exception:  # pragma: no cover - SVar/App have no static default
        return False


def _eq(a: B.Basis, b: B.Basis) -> bool:
    """Numpy-tolerant structural basis equality (shared with represent's cob)."""
    from .represent import _basis_eq

    return _basis_eq(a, b)
