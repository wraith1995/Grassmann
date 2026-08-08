"""``unify`` must CLASSIFY a ⊗-arity failure, not resolve it — and the two ⊗-leaf readers
(space-level and basis-level) must agree on leaf ORDER.

Background.  ``OProd`` is a flat tuple of factors, so the NESTING of a ⊗-tower is significant:
``lower.represent._shape_of_space`` makes each top-level factor its own array axis, so
``V*⊗V*⊗V*`` is ``(3,3,3)`` while ``V*⊗(V*⊗V*)`` is ``(3,9)``.  Two producers disagreed about which
they build — ``tensor_power(V, k)`` is FLAT, ``linear_map_space(dom, cod) = oprod(dual(dom), cod)``
RIGHT-NESTS as soon as ``cod`` is compound — and at k = 3 the two met and ``unify`` refused.

The refusal is CORRECT and stays: ⊗ associativity makes the two one *space* but not one *layout*,
and unifying them would let mismatched layouts meet silently.  What this file pinned first was the
diagnosis — both rows raise; the success set of ``unify`` is untouched.  What then changed is the
PRODUCER: ``transform.diff``'s derivative tower, the only nested ⊗-tower anything built, is now
flat (``_tower_space``, `§flat-tower`), so the association row is unreachable from it —
``test_the_derivative_tower_is_flat_and_never_reaches_the_association_row``.  A genuine map space
(``Alt₃``, an opaque Jacobian) is still a two-factor ``linear_map_space`` MATRIX, so the row itself
stays live and stays the right answer for anyone who respells one by hand.

⚠ **k = 2 is invisible** — both spellings are arity 2 there — which is exactly why the space level
looked healthy while the BASIS level was silently wrong (the rank-2 defect fixed in ``b7af6db``:
``_reconcile_pairing`` matched by top-level factor count, refused, and contracted the operand RAW).
``test_leaf_readers_agree_on_order`` is the tripwire that keeps ``core.space.tensor_leaves`` and
``lower.represent._basis_prod_leaves`` — now a matched pair — from drifting apart again.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.core import space as S
from grassmann.infer.unify import AssociationMismatch, MetaStore, UnifyError, unify
from grassmann.lower.represent import _basis_prod_leaves, _shape_of_space
from grassmann.lower.space_basis import default_basis, map_basis
from grassmann.stdlib.exterior import alt

V3 = S.OPlusRep(S.DimLit(3), S.Scalar())
V5 = S.OPlusRep(S.DimLit(5), S.Scalar())


def _flat(k: int, leaf: S.Space) -> S.Space:
    """``leaf* ⊗ … ⊗ leaf*`` — the FLAT spelling, as ``tensor_power`` builds it."""
    return S.oprod(*[S.dual(leaf)] * k)


def _nested(k: int, leaf: S.Space) -> S.Space:
    """``leaf* ⊗ (leaf* ⊗ (… ))`` — the RIGHT-NESTED spelling, as currying builds it."""
    sp: S.Space = S.dual(leaf)
    for _ in range(k - 1):
        sp = S.oprod(S.dual(leaf), sp)
    return sp


# ===========================================================================
# Phase 1 — `tensor_leaves` itself.
# ===========================================================================


def test_tensor_leaves_flattens_only_oprod():
    assert S.tensor_leaves(_flat(3, V3)) == [S.dual(V3)] * 3
    assert S.tensor_leaves(_nested(3, V3)) == [S.dual(V3)] * 3
    assert S.tensor_leaves(V3) == [V3]          # a non-product is its own single leaf


def test_tensor_leaves_treats_dual_as_a_leaf_not_a_distribution_point():
    """``Dual(A⊗B)`` is ONE axis of size ``|A|·|B|``; ``Dual(A)⊗Dual(B)`` is two.  Different
    layouts ⇒ the reader must not identify them (and ``_basis_prod_leaves`` does not either —
    ``default_basis(Dual(A⊗B))`` is a ``DualBasis``, not a ``BasisProd``)."""
    fused = S.dual(S.oprod(V3, V5))
    split = S.oprod(S.dual(V3), S.dual(V5))
    assert S.tensor_leaves(fused) == [fused]
    assert S.tensor_leaves(split) == [S.dual(V3), S.dual(V5)]


def test_tensor_leaves_keeps_the_tensor_unit():
    """The ⊗-unit occupies no array axis but it DOES occupy a factor position, and
    ``_basis_prod_leaves`` keeps its ``ScalarBasis`` tail.  Dropping it here would put the two
    readers out of step, which is the drift Phase 4 exists to catch."""
    sp = S.oprod(S.dual(V3), S.oprod(S.dual(V3), S.Scalar()))
    assert S.tensor_leaves(sp) == [S.dual(V3), S.dual(V3), S.Scalar()]


# ===========================================================================
# Phase 2 — the two failure rows.
# ===========================================================================


def test_association_mismatch_is_reported_as_its_own_failure():
    """Same 3 leaves, different nesting ⇒ the NEW error, naming both spellings and both
    layouts.  It is still a ``UnifyError`` subclass: nothing that used to fail now succeeds."""
    with pytest.raises(AssociationMismatch) as excinfo:
        unify(_flat(3, V3), _nested(3, V3), MetaStore())
    msg = str(excinfo.value)

    assert isinstance(excinfo.value, UnifyError)
    assert "ASSOCIATION mismatch" in msg
    assert "(3, 3, 3)" in msg and "(3, 9)" in msg     # both `_shape_of_space` layouts
    assert repr(_flat(3, V3)) in msg                  # both spellings, verbatim
    assert repr(_nested(3, V3)) in msg
    assert "tensor_power" in msg and "linear_map_space" in msg   # flat vs nested producer


def test_association_mismatch_is_symmetric():
    for a, b in ((_flat(3, V3), _nested(3, V3)), (_nested(3, V3), _flat(3, V3))):
        with pytest.raises(AssociationMismatch) as excinfo:
            unify(a, b, MetaStore())
        assert "(3, 3, 3)" in str(excinfo.value) and "(3, 9)" in str(excinfo.value)


def test_genuine_arity_mismatch_keeps_the_old_message():
    """Different leaves ⇒ the spaces really differ; the pre-existing text is the right one and
    is left byte-identical (and is NOT the association subclass)."""
    a, b = _flat(3, V3), _flat(2, V5)
    with pytest.raises(UnifyError) as excinfo:
        unify(a, b, MetaStore())
    assert not isinstance(excinfo.value, AssociationMismatch)
    assert str(excinfo.value) == (
        "⊗ arity mismatch: 3 ≢ 2 factors, and neither side "
        "has a Scalar (⊗-unit) factor to drop, so there is no rewrite that makes "
        "progress.\nThis is usually ONE space spelled two ways — a flat "
        "`tensor_power(V, k)` product meeting a right-nested map space "
        "`linear_map_space` builds by currying. They are isomorphic under the tensor "
        "encoding of maps but have different AXIS structure, so they must be made to "
        "agree at construction, not here.\n"
        f"  a = {a!r}\n  b = {b!r}"
    )


def test_equal_leaf_count_but_mismatched_leaves_is_still_the_old_message():
    """The leaf COUNT agreeing is not enough — the leaves must unify pairwise.  ``V*⊗V*⊗W*``
    against ``V*⊗(V*⊗V*)`` has three leaves a side and is still a genuine mismatch."""
    a = S.oprod(S.dual(V3), S.dual(V3), S.dual(V5))
    with pytest.raises(UnifyError) as excinfo:
        unify(a, _nested(3, V3), MetaStore())
    assert not isinstance(excinfo.value, AssociationMismatch)
    assert "arity mismatch" in str(excinfo.value)


def test_classification_does_not_bind_metavars():
    """The classification asks "would these unify?" on a COPY of ``Θ``.  Asking must not solve
    anything: the store the caller handed in is unchanged whichever row fires."""
    m = S.SpaceMeta()
    store = MetaStore()
    with pytest.raises(AssociationMismatch):
        unify(S.oprod(S.dual(V3), m, S.dual(V3)), _nested(3, V3), store)
    assert store.get(m) is None


def test_success_set_is_untouched():
    """Every pair that unified before still unifies — the ⊗-unit and 1-ary rewrites in
    particular, which reach the same ``OProd`` arm and must not be diverted."""
    unify(_flat(3, V3), _flat(3, V3), MetaStore())
    unify(S.oprod(S.dual(V3), S.Scalar()), S.dual(V3), MetaStore())
    unify(S.oprod(V3), V3, MetaStore())
    m = S.SpaceMeta()
    store = MetaStore()
    unify(_nested(3, V3), m, store)
    assert store.get(m) == _nested(3, V3)


# ===========================================================================
# Phase 4 — the two leaf readers are a matched pair.
# ===========================================================================


@pytest.mark.parametrize(
    "tower",
    [
        _flat(3, V3),
        _nested(3, V3),
        S.oprod(S.dual(V3), S.oprod(S.dual(V3), S.Scalar())),          # right-nested, unit tail
        S.oprod(S.oprod(S.dual(V3), S.dual(V5)), S.dual(V3)),          # LEFT-nested
        S.oprod(S.dual(S.oprod(V3, V5)), S.dual(V3)),                  # a fused Dual leaf
        S.oprod(V3, S.oprod(S.oplus(V3, V5), S.Scalar())),             # a ⊕ leaf
    ],
)
def test_leaf_readers_agree_on_order(tower):
    """``core.space.tensor_leaves`` and ``represent._basis_prod_leaves`` induce the SAME leaf
    sequence on the same tower — positionally, not as a multiset.

    ``default_basis`` mirrors the space tower node for node, so the basis reader's leaves must be
    exactly the default bases of the space reader's leaves, in order.  A multiset comparison would
    be blind to a transposition, which is the failure mode that matters: ⊗ is associative but NOT
    commutative, and a reordering silently permutes array axes.
    """
    space_leaves = S.tensor_leaves(tower)
    basis_leaves = _basis_prod_leaves(default_basis(tower))
    assert basis_leaves == [default_basis(s) for s in space_leaves]


def test_leaf_readers_agree_under_a_non_default_leaf_policy():
    """Not an artifact of ``default_basis``: with a leaf CHOICE overriding one leaf, the two
    readers still line up position for position (``map_basis`` mirrors the same tower)."""
    tower = S.oprod(S.dual(V3), S.oprod(S.dual(V5), S.Scalar()))
    pinned = G.basis("β", V5)

    def choice(sp):
        return pinned.node if sp == V5 else None

    space_leaves = S.tensor_leaves(tower)
    basis_leaves = _basis_prod_leaves(map_basis(tower, choice))
    assert basis_leaves == [map_basis(s, choice) for s in space_leaves]


# ===========================================================================
# The splice guard: a map space stays a MATRIX.
# ===========================================================================


def test_alt3_still_realizes_as_a_27_by_27_matrix():
    """``Alt₃ : V⊗³ ⊸ V⊗³`` is ``Dual(V⊗³) ⊗ V⊗³`` — TWO axes, ``(27, 27)``, which is what the
    SVD/QR seam consumes.  Splicing ⊗ at construction would flatten it to six ``(3,…)`` axes; this
    pins that nothing here moved in that direction."""
    V = G.osum(*[G.R] * 3)
    out = G.compile(alt(V, 3)).run({})["result"]
    assert np.asarray(out).shape == (27, 27)


# ===========================================================================
# The producer that used to reach the association row.
# ===========================================================================


def test_the_derivative_tower_is_flat_and_never_reaches_the_association_row():
    """``diff``'s derivative tower was the ONE nested ⊗-tower producer, and this row is the reason
    it had to stop being one.

    An order-``n`` derivative input is ``X ⊸ (X ⊸ (… ⊸ V))``.  Spelled with ``linear_map_space``
    that right-nests, so from order 3 up it met the FLAT products it is contracted against
    (``tensor_power``, ``Kron``'s spliced endpoints) and landed exactly on the
    ``AssociationMismatch`` row above — an error with no repair, because ⊗ associativity makes the
    two one space but ``_shape_of_space`` makes them two LAYOUTS.  ``transform.diff._tower_space``
    now builds the tower FLAT, so both sides agree by construction (`§flat-tower`).

    Assert it on the ARTIFACT the producer mints, not on a hand-written space: what matters is that
    nothing in the stack still spells a tower nested."""
    from grassmann.core import term as T
    from grassmann.transform.diff import clear_deriv_inputs, diff, enumerate_derivative_inputs

    clear_deriv_inputs()
    try:
        u, x = T.Var("u", S.Scalar()), T.Var("x", V3)
        deps: dict[int, object] = {u.uid: None, x.uid: None}
        e: object = u
        for _ in range(3):
            e = G.beta(diff(e, x, deps))
            e = e.node if hasattr(e, "node") else e
            for rec in enumerate_derivative_inputs(e):
                deps.setdefault(rec.var.uid, None)
        (d3,) = [r for r in enumerate_derivative_inputs(e) if r.order == 3]
    finally:
        clear_deriv_inputs()

    # FLAT: three seed factors then the value's ⊗-unit — four factors, four axes, not two.
    assert d3.var.space == S.oprod(S.dual(V3), S.dual(V3), S.dual(V3), S.Scalar())
    assert _shape_of_space(d3.var.space) == (3, 3, 3, 1)
    # and it unifies with the flat spelling the contraction partners build (the ⊗-unit drops).
    unify(d3.var.space, _flat(3, V3), MetaStore())
