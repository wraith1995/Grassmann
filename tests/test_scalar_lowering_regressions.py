"""Scalar-lowering regressions (fem #13 matrix findings G1–G3, fixed 2026-07-03).

The R ≅ R¹ empty-product law: a Scalar-space value is stored 0-d, but contraction and
change-of-basis seams consume the coordinate vector ``[x]`` in the standard basis {1}.
Two seams were missing the law (G1/G3), and the LLam vmap counted codomain axes off the
SPACE instead of the body ARRAY (G2 — a nested map's inner scalar cod is FUSED away).
"""
from __future__ import annotations

import numpy as np
import grassmann as G


def _run(prog, feeds: dict) -> np.ndarray:
    binding = {}
    for inp in prog.inputs:
        base = inp.name.rsplit("_", 1)[0]
        binding[inp.name] = feeds[base]
    return np.asarray(prog.run(binding)["result"])


def test_scalar_dot_compiles_and_runs():
    # G1: dot of two SCALARS — the humblest inner product — through represent's Eval.
    u = G.var("u", G.R)
    v = G.var("v", G.R)
    prog = G.compile(G.dot(u, v))
    assert np.allclose(_run(prog, {"u": 2.0, "v": 3.0}), 6.0)


def test_scalar_product_plus_scalar_dot():
    # G3's grassmann half: a SUM whose operands carry different (1-dim) scalar bases —
    # the sum-cob seam must apply the 1×1 cob through the R¹ coordinate vector.
    u = G.var("u", G.R)
    v = G.var("v", G.R)
    term = u * v + G.dot(u, v)                     # 2·u·v once lowered
    prog = G.compile(term)
    assert np.allclose(_run(prog, {"u": 2.0, "v": 3.0}), 12.0)
