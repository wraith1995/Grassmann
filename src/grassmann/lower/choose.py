"""The basis-choice policy: which of a space's candidate bases lowering represents it in.

Every space offers several honest coordinatizations, and they are not equally cheap. The
one this module picks is the one expected to leave the fewest symbolic cells in the
lowered result, scored by an additive heuristic over the candidates
:func:`~grassmann.lower.space_basis.space_basis` supplies::

    candidates ──► score each ──► the highest wins
      standard orthonormal        +10   metric is the identity, so the cob is numeric
      the operator's own basis    +40   its projection is the identity there
      the basis the input is in  +100   one basis on both sides, so no cob at all
      the g-orthonormal basis     +20   of a metric space, so ⟨·,·⟩ is numeric

Four considerations produce those scores. A value derived from a symbolic operator ``f``
is cheapest in a basis aligned to ``f``'s own structure — an ``ImageBasis`` or a
``ProjectedBasis`` of that same map — because the projection onto the subspace is the
identity there and the coordinates come out numeric. A candidate that makes the result
numeric is worth taking even when it costs a change of basis, provided that cob is itself
numeric. The inner-product-orthonormal standard basis makes the metric the identity, so
``Inner``, ``Dagger`` and the solves lower to numeric operations, which makes it the
baseline winner where no operator structure is available. And when an input basis is
offered and is among the candidates — an endomorphism, or two parametrically identified
spaces — using it on both sides avoids a cob entirely, and with it the symbolic entries a
cob would introduce.

The choice is local: each space is scored on its own, with no search over the combinations
a whole term admits. :func:`score_basis` exposes the score, a small integer where higher is
better.
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
    """Pick the basis of ``space`` expected to leave the fewest symbolics, from ``candidates``.

    ``candidates`` is normally the second element of
    :func:`grassmann.lower.space_basis.space_basis`. ``term`` is the core term being
    represented, and it is what lets the scorer recognize that a candidate is the structure
    basis of the very operator the value came from, and whether the term consumes a metric.
    ``in_basis`` is the basis already chosen for the input side, and a candidate matching it
    wins outright, since one basis on both sides means no change of basis at all.

    Ties go to the earliest candidate, which is the space's default basis.
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
    """Score one candidate basis of ``space``: higher means fewer expected symbolic cells.

    The score is additive over the considerations described in the module docstring, so a
    candidate can collect several bonuses at once — the g-orthonormal basis of a metric space
    that also happens to be the input basis, say.
    """
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

    # ── Keep the cob and the metric numeric: the orthonormal standard default.
    #   An orthonormal basis makes the metric the identity, so ``Inner``, ``Dagger`` and
    #   ``Solve`` lower to numeric operations.  That is a convenience, not a correctness
    #   requirement, and at an un-pinned ``Img(f)`` leaf it has a price: the default there is
    #   the ``ProjectedBasis``, the orthonormalizing QR/SVD basis, whose ``R`` factor is
    #   algebraic-irrational in the program inputs and therefore cannot be carried by
    #   polyarray's exact rational lane.  The literal ``ImageBasis(f)``, the ``f(eᵢ)`` columns,
    #   spans the same subspace and is rational in the inputs, so the exact lane runs through
    #   it.  The +10 is awarded where orthonormality is free — a space whose default is already
    #   rational, so no alternative is being given up — or where ``term`` genuinely pairs the
    #   value through an operation that consumes a metric (`_demands_orthonormal`).  Otherwise
    #   it goes to the rational basis instead, below.
    #
    #   The alternative is only ever considered under the propagated rational-basis lane
    #   (`literal_ffs_frames`), which is off by default, so ordinarily the orthonormal default
    #   simply wins.  Flipping every leaf unconditionally would be unsound: an un-pinned ``Img``
    #   leaf feeding a metric, a gauge or a Λᵏ inner product needs the orthonormal basis.  A
    #   consumer that wants exactness opts in, and only then does the fold-preserving
    #   ``ImageBasis`` take the bonus.
    demands = _demands_orthonormal(term)
    fold_alt = _rational_frame_lane() and _fold_preserving_alternative_exists(space)
    if _is_orthonormal_default(space, candidate) and (not fold_alt or demands):
        score += 10
    # ── The rational literal image basis of an un-pinned ``Img`` leaf, where no metric or
    #   orthonormality demand forces the orthonormal one.  This is symmetric with the +10
    #   above — exactly one of the two operator bases earns it — so the rational ``ImageBasis``
    #   takes the leaf at 50 against the orthonormal ``ProjectedBasis``'s 40.
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
    """Give the map a fundamental subspace is defined by, or ``None`` for any other space.

    ``Img(f)``, ``Ker(f)``, ``CoImg(f)`` and ``CoKer(f)`` are all built from ``f``, and it is
    ``f``'s own structure basis that makes their coordinates numeric.
    """
    from grassmann.infer.subspace import ffs_defining_map

    return ffs_defining_map(space)


def _is_operator_basis_of(candidate: B.Basis, op: object) -> bool:
    """Report whether ``candidate`` is a structure basis of the map ``op`` itself.

    Only an ``ImageBasis`` or a ``ProjectedBasis`` carries a defining map, and it counts here
    only when that map is ``op``: a basis of some *other* operator buys none of the structure
    the bonus is paying for.
    """
    match candidate:
        case B.ImageBasis(f):
            return f is op or f == op
        case B.ProjectedBasis(f, _, _, _):
            return f is op or f == op
        case _:  # audit: ok EX-WILDCARD — safe direction: False only skips a scoring bonus [approved 2026-07-03]
            return False


def _rational_frame_lane() -> bool:
    """Report whether the rational (literal) basis lane is active for this lowering.

    The scorer's preference for a rational basis is opt-in, and the switch is the same one that
    makes the frame's *realization* rational: ``represent(literal_frames=True)`` or the
    :func:`~grassmann.lower.represent.literal_ffs_frames` context. It is off by default, so the
    orthonormal default ordinarily wins unopposed. The import is deferred to the call to keep
    :mod:`represent` and :mod:`choose` from importing each other at module load.
    """
    from .represent import _literal_ffs_frames

    return _literal_ffs_frames()


def _fold_preserving_alternative_exists(space: S.Space) -> bool:
    """Report whether ``space`` has a rational basis available beside its orthonormal default.

    The image of an injective map is the one case. Its orthonormal default, the
    ``ProjectedBasis``, is the QR basis whose ``R`` factor is algebraic-irrational and so
    cannot be carried by the exact lane, while the literal ``ImageBasis(f)`` — the columns
    ``f(eᵢ)`` — is rational in the inputs and, because injectivity means full column rank, is a
    genuine basis of that same image.

    Nothing else qualifies. A non-injective map's literal columns are rank-deficient and hence
    not a basis at all, so the SVD ``ProjectedBasis`` remains the honest default there. ``Ker``,
    ``CoImg``, ``CoKer`` and the non-subspace spaces are never offered a literal basis in the
    first place: :func:`~grassmann.lower.space_basis.candidate_bases` adds an ``ImageBasis``
    only for an ``Img``.
    """
    from grassmann.infer.properties import map_props

    if not isinstance(space, S.Img):
        return False
    p = map_props(space.f)
    return p is not None and p.inj


# The nodes whose coordinates are numeric only in an orthonormal basis: the ones that consume a
# metric or an orthonormality.  Their presence in a term is what makes the orthonormal default's
# identity metric worth the QR/SVD radical; without them the rational basis is preferred at an
# ``Img`` leaf.  Trace, Det and LogDet are basis-independent scalars, and Project, Embed and
# Corestrict act in any basis, so none of those demands orthonormality.
def _demands_orthonormal(term: object | None) -> bool:
    """Report whether ``term`` pairs its value through an operation that consumes a metric.

    It searches the term for an ``Inner``, ``Dagger``, ``Solve``, ``Pinv``, ``SolveApply``,
    ``PinvApply``, ``Flat``, ``Sharp`` or ``Remetric`` node. The walk goes through the generic
    reflective child enumeration :func:`grassmann.core.term._children`, so this is a membership
    test over the term rather than a partial dispatch ladder over the ``Term`` sum type, and a
    node kind it has never seen costs it nothing.

    A missing term, or a shape it cannot walk, answers ``False``. That is the conservative
    direction: it keeps the rational basis, which the exact lane can carry.
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

    A space with no static default basis — an ``SVar`` or an unreduced ``App`` — answers
    ``False``, since there is no standard basis for the candidate to be. The import is deferred
    to the call to avoid a cycle with :mod:`space_basis`.
    """
    from .space_basis import default_basis

    try:
        return _eq(candidate, default_basis(space))
    except Exception:  # pragma: no cover - SVar/App have no static default
        return False


def _eq(a: B.Basis, b: B.Basis) -> bool:
    """Compare two bases structurally, tolerating the numpy arrays a basis may carry.

    This is the same equality :mod:`~grassmann.lower.represent` uses to decide whether a cob is
    needed, so a candidate the scorer calls equal to the input basis is exactly one that would
    cost no change of basis.
    """
    from .represent import _basis_eq

    return _basis_eq(a, b)
