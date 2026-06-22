"""``WithInner`` metric must be a Term — not a raw array (the design: ``metric: Term``).

A concrete numeric metric is supplied as a ``Var`` fed at run time, never embedded as an array.
This keeps the space/basis hashable (so the metric-Eval operand dedup works) and the lowering
uniform (a single ``_rep(metric)`` path).  Regression for the array escape hatch + the unhashable
``WithInner`` residual it caused.
"""

import numpy as np
import pytest

import grassmann.core as C
from grassmann.core.space import SpaceError
from grassmann.lower import represent


def _vk(v):
    return f"{v.name}_{v.uid}"


def _metric_var(V):
    return C.var("g", C.oprod(C.dual(V), C.dual(V)))  # g : V*⊗V*


def test_array_metric_is_rejected_at_construction():
    V = C.base("V", 3)
    with pytest.raises(SpaceError, match="Term"):
        C.with_inner(V, np.eye(3))


def test_non_term_sentinel_metric_is_rejected():
    V = C.base("V", 3)
    with pytest.raises(SpaceError, match="Term"):
        C.with_inner(V, object())


def test_var_metric_withinner_is_hashable():
    # A Term (Var) metric ⇒ the space and its default basis are hashable — the root cause of the
    # `_operand_metric_basis` residual is gone (no unhashable numpy array embedded in the IR).
    V = C.base("V", 3)
    g = _metric_var(V)
    Vg = C.with_inner(V, g)
    assert hash(Vg)  # does not raise
    assert hash(C.WithInnerBasis(C.list_basis(C.var("z", V)), g))


def test_var_metric_withinner_dot_uses_the_gram():
    # The residual was a CRASH lowering a metric Eval over a WithInner operand; with a Var metric
    # it lowers, runs, and computes the true metric inner product ⟨a,b⟩ = acᵀ·G·bc (fed at run).
    V0 = C.base("V", 3)
    g = _metric_var(V0)
    Vg = C.with_inner(V0, g)
    a, b = C.var("a", Vg), C.var("b", Vg)
    dot = C.eval_(C.inner(Vg), (0, 1), a, b)
    G = np.array([[2.0, 0, 0], [0, 3, 0], [0, 0, 1.0]])
    ac, bc = np.array([1.0, 1, 2]), np.array([0.0, 2, 1])
    out = np.asarray(
        represent(dot).run({_vk(g): G, _vk(a): ac, _vk(b): bc})["result"], float
    )
    assert np.allclose(out, ac @ G @ bc)
    assert not np.allclose(out, ac @ bc)
