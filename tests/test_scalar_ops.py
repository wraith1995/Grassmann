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

import grassmann as G
import grassmann.stdlib as L
import grassmann.stdlib.linalg as LA


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
