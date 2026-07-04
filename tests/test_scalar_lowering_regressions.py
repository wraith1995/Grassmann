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


def test_transpose_map_types_dual_and_is_metric_free():
    """fem #15 F1: the covector-factor transport is the metric-free TRANSPOSE
    ``λω. ω∘f : Dual(B) ⊸ Dual(A)`` (LLam + Compose — defined in the calculus, Teo),
    NOT ``dagger`` (the metric adjoint, typed PRIMAL ``B ⊸ A`` — pairing a covector
    against it was the Dual(Img J) ~ Img J unify refusal). Applied to a covector it
    computes exactly ``fᵀω`` with no Gram anywhere."""
    from grassmann.core import term as T
    from grassmann.core.space import Dual
    from grassmann.embed.ops import Expr
    from grassmann.infer.synth import synth_space
    from grassmann.transform.pullback import _transpose_map

    A = G.osum(*[G.R] * 2)
    B = G.osum(*[G.R] * 3)
    f = G.var("f", A >> B)                       # f : A ⊸ B (stored (2, 3))
    tp = _transpose_map(f.node, B.node if hasattr(B, "node") else B,
                        A.node if hasattr(A, "node") else A)
    w = G.var("w", ~B)
    applied = T.Eval(tp, (0,), (w.node,))
    assert isinstance(synth_space(applied), Dual)     # lands in Dual(A)
    prog = G.compile(Expr(applied))
    F = np.arange(6, dtype=float).reshape(2, 3)
    W = np.array([1.0, 2.0, 3.0])
    feed = {i.name: (F if i.name.startswith("f") else W) for i in prog.inputs}
    out = np.asarray(prog.run(feed)["result"])
    assert np.allclose(out.ravel(), F @ W)            # fᵀω in stored-(dom,cod) convention


import pytest


@pytest.mark.xfail(reason="fem #15 P(T) tail: dual-FFS STD readout delivers IN-SPACE "
                          "(Q-dual, R^-T) coordinates under the ambient-STD label — "
                          "basis-honesty bug; blocks Whitney P(T)=I (see FINDINGS-2026-07-03)",
                   strict=True)
def test_dual_image_pin_reads_out_ambient_std():
    """A covector pinned to dual(ImageBasis(J)) read out at TargetBasis.STD must give the
    AMBIENT standard coordinates of θ1 — the pinv(J) row (θ_i(J e_j) = δ_ij). It currently
    gives R⁻ᵀ column 1 = θ1 in the in-space Q-dual frame, silently mislabeled as ambient.
    Downstream, Embed's ImageBasis fast path consumes the mislabeled coordinates and the
    Whitney edge-trace W picks up a spurious |J_edge| factor per edge (P(T) ≠ I)."""
    from grassmann.core import basis as B
    from grassmann.lower.represent import represent

    R2 = G.osum(G.R, G.R)
    jc = G.var("jc", R2 >> R2, props=G.inj)
    w = G.var("w", ~G.img(jc))
    pin = B.dual_basis(B.image_basis(jc.node))
    prog = represent(w.node, basis_env={w.node.uid: pin}, target=B.TargetBasis.STD)
    Jc = np.array([[2.0, -0.5], [1.0, 2.0]])
    feed = {}
    for i in prog.inputs:
        feed[i.name] = {"jc": Jc.T, "w": np.array([1.0, 0.0])}[i.name.rsplit("_", 1)[0]]
    out = np.asarray(prog.run(feed)["result"]).ravel()
    assert np.allclose(out, np.linalg.pinv(Jc)[0], atol=1e-12), out
