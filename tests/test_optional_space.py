"""Optional space annotations on ``Zero`` / ``Project``, unified later.

The explicit space annotation is OPTIONAL (``None``) on :class:`~grassmann.core.term.Zero`
and :class:`~grassmann.core.term.Project`; inference recovers it via first-order space
unification (a fresh ``SpaceMeta`` solved during synthesis / checking). The explicit-space
path is unchanged.

Scope note: ``Zero(None)`` is fully recoverable by unification (a Sum sibling, a
check/Assert against an expected space). ``Project(v, None)``'s *target subspace* is not
unique, so it is recoverable only in the ⇐ check direction (an expected space flows in);
a standalone ``Project(v, None)`` is reported as ambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann.core as C
import grassmann.core.props as PR
from grassmann.infer import InferError, UnifyError, check, synth_space
from grassmann.lower.represent import represent

V = C.base("V", 3)
W = C.base("W", 4)


# ---------------------------------------------------------------------------
# Zero — optional space, recovered by unification.
# ---------------------------------------------------------------------------


def test_zero_optional_space_allowed_structurally():
    z = C.Zero()  # no space
    assert z.space is None
    assert C.zero().space is None  # smart ctor default


def test_zero_explicit_space_unchanged():
    assert synth_space(C.Zero(V)) == V


def test_zero_none_recovered_in_sum():
    v = C.var("v", V)
    assert synth_space(C.Sum((C.Zero(), v))) == V


def test_zero_none_recovered_by_check():
    z = C.Zero()
    check(z, V)  # solves the fresh meta to V; no error
    with pytest.raises((InferError, UnifyError)):
        check(C.Sum((C.Zero(), C.var("w", W))), V)  # W ≠ V


def test_zero_none_lowers_in_sum_context():
    v = C.var("v", V)
    prog = represent(C.Sum((C.Zero(), v)))
    out = prog.run({f"v_{v.uid}": np.array([1.0, 2.0, 3.0])})["result"]
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_zero_explicit_lowers_identically():
    v = C.var("v", V)
    none_prog = represent(C.Sum((C.Zero(), v)))
    expl_prog = represent(C.Sum((C.Zero(V), v)))
    inputs = {f"v_{v.uid}": np.array([4.0, 5.0, 6.0])}
    np.testing.assert_allclose(
        none_prog.run(inputs)["result"], expl_prog.run(inputs)["result"]
    )


# ---------------------------------------------------------------------------
# Project — optional target subspace, recovered in the ⇐ check direction.
# ---------------------------------------------------------------------------


def _injective_image():
    f = C.var("f", C.linear_map_space(C.base("Vi", 2), W), props=PR.inj)
    return C.img(f, source=C.base("Vi", 2), target=W)


def test_project_optional_space_allowed_structurally():
    w = C.var("w", W)
    p = C.Project(w)
    assert p.space is None
    assert C.project(w).space is None


def test_project_explicit_space_unchanged():
    from grassmann.core.space import Img

    imgf = _injective_image()
    w = C.var("w", W)
    assert isinstance(synth_space(C.Project(w, imgf)), Img)


def test_project_none_recovered_by_check():
    imgf = _injective_image()
    w = C.var("w", W)
    check(C.Project(w), imgf)  # recovers the omitted target, validates ⊑ W


def test_project_none_standalone_is_ambiguous():
    w = C.var("w", W)
    with pytest.raises(InferError):
        synth_space(C.Project(w))


def test_project_none_rejects_non_subspace_target():
    # An expected space that is NOT a subspace of the operand's space is rejected.
    w = C.var("w", W)
    other = C.base("Z", 5)
    with pytest.raises((InferError, UnifyError)):
        check(C.Project(w), other)


def test_malformed_space_still_rejected():
    with pytest.raises(C.TermError):
        C.Zero("not a space")
    with pytest.raises(C.TermError):
        C.Project(C.var("w", W), "not a space")
