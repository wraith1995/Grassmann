"""IW5 tests: the basic nonlinear scalar stdlib ops (the old ``BaseNonLinear``).

Each op (``sqrt``/``absval``/``sign``/``div``/``reciprocal``) builds a core
:class:`~grassmann.core.term.External` over ``Scalar`` operands, lowers through
:func:`grassmann.lower.represent.represent`, runs via :meth:`polyarray.Program.run`, and
matches the numpy reference (``np.sqrt``/``np.abs``/``np.sign``/division/reciprocal).

A composite then ties them together — vector normalization
``x · reciprocal(sqrt(dot(x, x)))`` (and the scalar division ``a / b``) — verified against
the numpy normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
import grassmann.core.space as S
import grassmann.stdlib as L
import grassmann.stdlib.linalg as LA
from grassmann.stdlib.scalar import _is_euclidean


def _key(v: G.Expr) -> str:
    node = v.node
    return f"{node.name}_{node.uid}"


def test_sqrt_matches_numpy() -> None:
    x = G.var("x", G.R)
    out = G.run(L.sqrt(x), {_key(x): np.asarray(9.0)})["result"]
    np.testing.assert_allclose(out, np.sqrt(9.0))


def test_absval_matches_numpy() -> None:
    x = G.var("x", G.R)
    for v in (-3.5, 0.0, 2.25):
        out = G.run(L.absval(x), {_key(x): np.asarray(v)})["result"]
        np.testing.assert_allclose(out, np.abs(v))


def test_sign_matches_numpy() -> None:
    x = G.var("x", G.R)
    for v in (-7.0, 0.0, 4.0):
        out = G.run(L.sign(x), {_key(x): np.asarray(v)})["result"]
        np.testing.assert_allclose(out, np.sign(v))


def test_div_matches_numpy() -> None:
    x = G.var("x", G.R)
    y = G.var("y", G.R)
    out = G.run(L.div(x, y), {_key(x): np.asarray(6.0), _key(y): np.asarray(4.0)})["result"]
    np.testing.assert_allclose(out, 6.0 / 4.0)


def test_reciprocal_matches_numpy() -> None:
    x = G.var("x", G.R)
    out = G.run(L.reciprocal(x), {_key(x): np.asarray(8.0)})["result"]
    np.testing.assert_allclose(out, np.reciprocal(8.0))


@pytest.mark.parametrize(
    ("op", "ref", "val"),
    [
        (L.exp, np.exp, 1.4),
        (L.log, np.log, 3.0),
        (L.square, np.square, -2.5),
        (L.sin, np.sin, 0.9),
        (L.cos, np.cos, 0.9),
        (L.tan, np.tan, 0.9),
        (L.sinh, np.sinh, 0.7),
        (L.cosh, np.cosh, 0.7),
        (L.tanh, np.tanh, 0.7),
        (L.atan, np.arctan, 0.7),
        (L.asin, np.arcsin, 0.4),
        (L.acos, np.arccos, 0.4),
        (L.sigmoid, lambda a: 1.0 / (1.0 + np.exp(-a)), 0.6),
        (L.softplus, lambda a: np.logaddexp(0.0, a), 0.6),
    ],
)
def test_extended_scalar_ops_match_numpy(op, ref, val) -> None:
    x = G.var("x", G.R)
    out = G.run(op(x), {_key(x): np.asarray(val)})["result"]
    np.testing.assert_allclose(out, ref(val))


@pytest.mark.parametrize(
    "name",
    ["exp", "log", "square", "sin", "cos", "tan", "sinh", "cosh", "tanh",
     "atan", "asin", "acos", "sigmoid", "softplus"],
)
def test_extended_op_provenance(name) -> None:
    x = G.var("x", G.R)
    prog = G.compile(getattr(L, name)(x))
    notes = [s.note for s in prog.statements]
    assert any(n == f"external:{name}" for n in notes), notes


def test_elementwise_on_vector_matches_numpy() -> None:
    """Scalar ops run elementwise on a Euclidean VECTOR value (not just on ``Scalar``)."""
    V = S.base("V", 3)
    x = G.var("x", V)
    xv = np.array([0.1, 0.4, -0.6])
    for op, ref in [(L.exp, np.exp), (L.sin, np.sin), (L.tanh, np.tanh), (L.square, np.square)]:
        out = G.run(op(x), {_key(x): xv})["result"]
        np.testing.assert_allclose(out, ref(xv))


def test_elementwise_on_matrix_matches_numpy() -> None:
    """And on a rank-2 (⊗) Euclidean value — elementwise, shape-preserving."""
    V, W = S.base("V", 3), S.base("W", 2)
    m = G.var("m", S.oprod(S.dual(V), W))
    mat = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    out = G.run(L.cosh(m), {_key(m): mat})["result"]
    np.testing.assert_allclose(out, np.cosh(mat))
    assert out.shape == (3, 2)


def test_hadamard_forward_vector_and_matrix() -> None:
    """The pointwise (Hadamard) product runs elementwise over Euclidean vectors and matrices."""
    V = S.base("V", 3)
    a, b = G.var("a", V), G.var("b", V)
    av, bv = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    out = G.run(L.hadamard(a, b), {_key(a): av, _key(b): bv})["result"]
    np.testing.assert_allclose(out, av * bv)
    # matrix (⊗) operands
    W = S.base("W", 2)
    p, q = G.var("p", S.oprod(S.dual(V), W)), G.var("q", S.oprod(S.dual(V), W))
    pv = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    qv = np.array([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
    outm = G.run(L.hadamard(p, q), {_key(p): pv, _key(q): qv})["result"]
    np.testing.assert_allclose(outm, pv * qv)


def test_euclidean_predicate() -> None:
    V = S.base("V", 3)
    assert _is_euclidean(S.Scalar())
    assert _is_euclidean(V)
    assert _is_euclidean(S.dual(V))
    assert _is_euclidean(S.oprod(S.dual(V), V))
    assert _is_euclidean(S.oplus(V, V))
    # NOT Euclidean: a re-metricized space.
    g = G.var("g", S.oprod(S.dual(V), S.dual(V))).node
    assert not _is_euclidean(S.with_inner(V, g))


def test_noneuclidean_operand_rejected() -> None:
    """A value in a re-metricized (non-Euclidean) space is refused, not silently applied."""
    V = S.base("V", 3)
    g = G.var("g", S.oprod(S.dual(V), S.dual(V))).node
    xm = G.var("xm", S.with_inner(V, g))
    with pytest.raises(ValueError, match="not Euclidean"):
        L.exp(xm)


def test_external_provenance_note() -> None:
    x = G.var("x", G.R)
    prog = G.compile(L.sqrt(x))
    notes = [s.note for s in prog.statements]
    assert any(n == "external:sqrt" for n in notes), notes


def test_normalize_composite_matches_numpy() -> None:
    """``x · reciprocal(sqrt(dot(x, x)))`` == numpy ``x / ‖x‖`` — sqrt + reciprocal + dot."""
    V = G.space("V", 3)
    x = G.var("x", V)
    unit = G.scale(L.reciprocal(L.sqrt(LA.dot(x, x))), x)
    xv = np.array([1.0, 2.0, 2.0])
    out = G.run(unit, {_key(x): xv})["result"]
    np.testing.assert_allclose(out, xv / np.linalg.norm(xv))


def test_scalar_division_composite() -> None:
    """``div`` ties to ``sqrt``: ``a / sqrt(b)`` == numpy reference."""
    a = G.var("a", G.R)
    b = G.var("b", G.R)
    expr = L.div(a, L.sqrt(b))
    out = G.run(expr, {_key(a): np.asarray(10.0), _key(b): np.asarray(4.0)})["result"]
    np.testing.assert_allclose(out, 10.0 / np.sqrt(4.0))


def test_lit_is_a_rank0_scalar() -> None:
    """``Lit`` lowers to a rank-0 scalar (``()``), matching genuine scalars — so it can be
    summed/combined with them. Regression for the ``np.ascontiguousarray`` 0-d→1-d promotion
    bug in ``_const`` that made constants ``(1,)`` instead of ``()``."""
    from grassmann.core import space as S
    from grassmann.core import term as T

    out = G.run(G.wrap(T.lit(2.0)), {})["result"]
    assert np.asarray(out).shape == ()
    # 1 + ⟨x,x⟩ : a Lit summed with a genuine (rank-0) scalar must agree on shape
    V = S.base("V", 2)
    x = T.var("x", V)
    dotxx = T.Eval(T.inner(V), (0, 1), (x, x))
    expr = T.sum_(T.lit(1.0), dotxx)
    val = G.run(G.wrap(expr), {f"x_{x.uid}": np.array([1.0, 2.0])})["result"]
    np.testing.assert_allclose(val, 1.0 + 5.0)
