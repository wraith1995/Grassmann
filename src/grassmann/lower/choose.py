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
    is_op_basis = op is not None and _is_operator_basis_of(candidate, op)
    if is_op_basis:
        # numeric-result-wins: in the operator's own basis the coordinates are numeric.
        score += 40

    # ── Keep cob numeric / identity metric: the orthonormal standard default.
    #   The orthonormal default makes the metric the identity (a numeric ``Inner``/
    #   ``Dagger``/``Solve``), which is a CONVENIENCE, not a correctness requirement.  At an
    #   un-pinned FFS ``Img(f)`` leaf that default is the ``ProjectedBasis`` — the
    #   ORTHONORMALIZING QR/SVD frame whose ``R``/``√`` factor is ALGEBRAIC-IRRATIONAL in the
    #   inputs, so it kills polyarray's exact (rational) lane (the "fold").  The literal
    #   ``ImageBasis(f)`` (``f(eᵢ)`` columns) spans the SAME subspace, is RATIONAL, and keeps
    #   the fold alive.  So award the +10 orthonormal bonus only where it is FREE — a space
    #   with no rational fold-preserving alternative (its default is already rational) — or
    #   where ``term`` genuinely pairs the value through a metric/orthonormality-consuming op
    #   (`_demands_orthonormal`); otherwise route the +10 to the rational frame (below).
    #
    #   GATED ON THE PROPAGATED RATIONAL-FRAME LANE (`literal_ffs_frames`).  Off by default the
    #   scorer is BYTE-IDENTICAL to the historical orthonormal-default choice — the numeric lane
    #   must not move, and a global leaf-flip is UNSOUND (it re-frames un-pinned ``Img`` leaves
    #   that feed the metric/gauge/Λᵏ-inner paths, which currently require the orthonormal frame).
    #   A consumer that wants the exact/rational lane opts in by PROPAGATION (pointwise's
    #   immersed-FEEC certificate path), and only then does the fold-preserving ``ImageBasis`` win.
    demands = _demands_orthonormal(term)
    fold_alt = _rational_frame_lane() and _fold_preserving_alternative_exists(space)
    if _is_orthonormal_default(space, candidate) and (not fold_alt or demands):
        score += 10
    # ── Fold-preservation: the RATIONAL literal image frame of an un-pinned ``Img`` leaf, when
    #   no metric/orthonormality demand forces the orthonormal frame.  Symmetric with the +10
    #   above (exactly one of the two operator bases earns it), so the rational ``ImageBasis``
    #   now WINS (50 vs 40) where the orthonormal ``ProjectedBasis`` used to (`73 Topic 2`).
    elif is_op_basis and isinstance(candidate, B.ImageBasis) and fold_alt and not demands:
        score += 10

    # ── g-orthonormal default for a metric space (identity metric ⇒ numeric Inner/dot).
    if isinstance(space, S.WithInner) and isinstance(candidate, B.WithInnerBasis):
        score += 20

    return score


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _operator_of(space: S.Space) -> object | None:
    """Give the linear-map term whose structure ``space`` is built from (an FFS), or None."""
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


def _rational_frame_lane() -> bool:
    """Whether the PROPAGATED rational (literal-FFS) frame lane is active for this lowering.

    The scorer's fold-preservation preference is opt-in — the exact/certificate consumer
    propagates it (``represent(literal_ffs_frames=True)`` / the ``literal_ffs_frames`` context),
    exactly the same switch that makes the frame REALIZATION rational.  OFF by default, so the
    orthonormal-default choice is byte-identical to the historical behaviour.  Imported lazily to
    avoid the ``represent`` ⇄ ``choose`` import cycle.
    """
    from .represent import _literal_ffs_frames

    return _literal_ffs_frames()


def _fold_preserving_alternative_exists(space: S.Space) -> bool:
    """Whether ``space`` offers a RATIONAL, fold-preserving frame beside its orthonormal default.

    An un-pinned ``Img(f)`` of an **injective** map is the one case: its orthonormal default
    (``ProjectedBasis``) is the QR frame whose ``R`` is algebraic-irrational (kills the exact
    lane), while the literal ``ImageBasis(f)`` (``f(eᵢ)`` columns) is RATIONAL and — because
    ``f`` is injective, full column rank — a genuine basis of the SAME image.  A NON-injective
    image's literal columns are rank-deficient (not a basis), so its SVD ``ProjectedBasis``
    stays the honest default; ``Ker``/``CoImg``/``CoKer`` and non-FFS spaces offer no literal
    frame at all (``candidate_bases`` adds ``ImageBasis`` only for ``Img``).
    """
    from grassmann.infer.properties import map_props

    if not isinstance(space, S.Img):
        return False
    p = map_props(space.f)
    return p is not None and p.inj


# Term nodes whose coordinates are simplest (numeric) in an ORTHONORMAL frame — the metric /
# orthonormality-consuming ops.  Their presence in ``term`` is what makes the orthonormal
# default's identity-metric a genuine win worth the QR/SVD radical; absent them, the rational
# fold-preserving frame is preferred at an ``Img`` leaf.  (Trace/Det/LogDet are basis-independent
# scalars; Project/Embed/Corestrict act in any basis — none demands orthonormality.)
def _demands_orthonormal(term: object | None) -> bool:
    """Whether ``term`` pairs the value through a metric/orthonormality-consuming op.

    Scans ``term`` (a core-IR Term) for an ``Inner``/``Dagger``/``Solve``/``Pinv``/
    ``SolveApply``/``PinvApply``/``Flat``/``Sharp``/``Remetric`` node.  Conservative: ``None``
    or an unrecognized shape ⇒ ``False`` (no demand), which keeps the fold-preserving frame —
    the SAFE direction for the exact lane.  Uses the generic reflective child walk
    (:func:`grassmann.core.term._children`), so it is a membership predicate, not a partial
    dispatch ladder over the ``Term`` sum type.
    """
    if term is None:
        return False
    from grassmann.core import term as T

    demanding = (T.Inner, T.Dagger, T.Solve, T.Pinv, T.SolveApply, T.PinvApply,
                 T.Flat, T.Sharp, T.Remetric)
    seen: set[int] = set()
    stack: list[object] = [term]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, demanding):
            return True
        stack.extend(T._children(node))
    return False


def _is_orthonormal_default(space: S.Space, candidate: B.Basis) -> bool:
    """Report whether ``candidate`` is the inner-product-orthonormal standard basis of ``space``.

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
