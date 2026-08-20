"""Top-level ``import grassmann as G`` pipeline.

Proves the single-import end-to-end works: build a small program purely through the
``G.*`` surface, then ``G.compile`` / ``G.run`` (embed → infer → represent → run) and
match a numpy reference.

Storage convention (see ``tests/test_represent.py``): a map space ``V >> W``
mirrors its factors onto axes ``(|V|, |W|)`` and ``Eval`` contracts the leading (dual)
axis, so a stored map matrix is the transpose of the usual ``(cod, dom)`` matrix.
"""

from __future__ import annotations

import numpy as np

import grassmann as G


def _key(v: G.Expr) -> str:
    node = v.node
    return f"{node.name}_{node.uid}"


def test_single_import_surface() -> None:
    """The intended top-level surface is all reachable off one import."""
    for name in ("space", "var", "llam", "osum", "tensor", "dual", "inner", "dot",
                 "pinv", "compile", "run", "core", "infer", "lower", "stdlib",
                 "wedge", "alt", "sqrt", "div"):
        assert hasattr(G, name), name


def test_matvec_pipeline_matches_numpy() -> None:
    """A linear map applied to a vector, built + run through the top-level surface."""
    V, W = G.space("V", 2), G.space("W", 3)
    f = G.var("f", V >> W)
    x = G.var("x", V)
    y = f(x)  # apply  → W

    fv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # stored (|V|, |W|)
    xv = np.array([1.0, 1.0])
    out = G.run(y, {_key(f): fv, _key(x): xv})["result"]
    np.testing.assert_allclose(out, fv.T @ xv)


def test_inner_product_pipeline_matches_numpy() -> None:
    """An inner product built via ``G.dot`` lowers + runs to the numpy dot product."""
    V = G.space("V", 4)
    x = G.var("x", V)
    z = G.var("z", V)
    s = G.dot(x, z)  # Scalar

    xv = np.array([1.0, -2.0, 3.0, 0.5])
    zv = np.array([2.0, 1.0, -1.0, 4.0])
    out = G.run(s, {_key(x): xv, _key(z): zv})["result"]
    np.testing.assert_allclose(out, xv @ zv)


def test_compile_accepts_core_term() -> None:
    """``G.compile`` unwraps an embed ``Expr`` and also takes a raw core ``Term``."""
    V = G.space("V", 2)
    x = G.var("x", V)
    prog_expr = G.compile(x)            # Expr
    prog_node = G.compile(x.node)       # raw core Term
    assert prog_expr is not None and prog_node is not None
