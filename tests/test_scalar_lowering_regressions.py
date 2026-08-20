"""Scalar-lowering regressions.

The R ≅ R¹ empty-product law: a Scalar-space value is stored 0-d, but contraction and
change-of-basis seams consume the coordinate vector ``[x]`` in the standard basis {1}.
Two seams were missing the law, and the LLam vmap counted codomain axes off the
SPACE instead of the body ARRAY (a nested map's inner scalar cod is FUSED away).
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
    # dot of two SCALARS — the humblest inner product — through represent's Eval.
    u = G.var("u", G.R)
    v = G.var("v", G.R)
    prog = G.compile(G.dot(u, v))
    assert np.allclose(_run(prog, {"u": 2.0, "v": 3.0}), 6.0)


def test_scalar_product_plus_scalar_dot():
    # a SUM whose operands carry different (1-dim) scalar bases —
    # the sum-cob seam must apply the 1×1 cob through the R¹ coordinate vector.
    u = G.var("u", G.R)
    v = G.var("v", G.R)
    term = u * v + G.dot(u, v)                     # 2·u·v once lowered
    prog = G.compile(term)
    assert np.allclose(_run(prog, {"u": 2.0, "v": 3.0}), 12.0)


def test_transpose_map_types_dual_and_is_metric_free():
    """The covector-factor transport is the metric-free TRANSPOSE
    ``λω. ω∘f : Dual(B) ⊸ Dual(A)`` (LLam + Compose — defined in the calculus),
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


def test_dual_image_wrapper_trace_chain_exact():
    """ P(T)=I seam regression: the ambient-immersion seam
    (embed/complete/backup) must ALWAYS receive standard-ambient columns —
    `ImageBasis.target` is representation-only and is never honored there, and the
    operator matrix in `_realize_image_basis` is coerced to its default slot bases.
    Before the fix, embedding a Λ¹ value pinned to the transported (dual-image) frame
    passed the pin-frame coefficients through under the ambient label, and the traced
    covector picked up a spurious |J_edge| per edge (Whitney P(T) ≠ I).

    The pinned end-to-end chain: ω = θ1 (the dual-image coframe covector) on the cell,
    traced through the honest inclusion ι to the edge, read out in the edge's dual-image
    frame — the coordinate is ω(Δx_edge) = 1 EXACTLY, on any cell."""
    from grassmann.core import basis as B
    from grassmann.core import term as T
    from grassmann.infer.synth import linear_map_space, map_domain_codomain
    from grassmann.lower.represent import ExternalOp, represent
    from grassmann.lower.space_basis import map_basis
    from grassmann.core import space as S
    from grassmann.transform.pullback import pullback

    R1, R2 = G.osum(G.R), G.osum(G.R, G.R)
    jc = G.var("jc", R2 >> R2, props=G.inj)
    je = G.var("je", R1 >> R2, props=G.inj)
    img_c, img_e = G.img(jc).node, G.img(je).node
    L1c = G.exterior_power(G.dual(G.img(jc)), 1)
    L1e = G.exterior_power(G.dual(G.img(je)), 1)
    w = G.var("w", L1c)
    imm = frozenset({jc.node.uid, je.node.uid})

    def immersed(space):
        def leaf(sp):
            match sp:
                case S.Img(f, _src, _tgt) if isinstance(f, T.Var) and f.uid in imm:
                    return B.image_basis(f)
                case S.Img(f, _src, _tgt):
                    return B.image_basis(f, None, None)
                case _:  # audit: ok EX-WILDCARD — choice-protocol: None = no override; map_basis mirrors/defaults exhaustively [approved 2026-07-03]
                    return None
        return map_basis(space, leaf)

    def prim(s_, t_):
        if s_ == img_e and t_ == img_c:
            dom_e, cod_e = map_domain_codomain(je.node)
            dom_c, _ = map_domain_codomain(jc.node)
            iota_std = T.External(linear_map_space(dom_e, dom_c),
                                  ExternalOp("iota_ref", lambda: np.array([[1.0, 0.0]])), ())
            v = T.var("v", img_e)
            return T.LLam((v,), T.Project(T.Eval(jc.node, (0,), (T.Eval(iota_std, (0,), (
                T.Eval(T.Pinv(je.node), (0,), (T.Embed(v, cod_e),)),)),)), img_c))
        return None

    pin_c = immersed(L1c.node if hasattr(L1c, "node") else L1c)
    tgt_e = immersed(L1e.node if hasattr(L1e, "node") else L1e)
    traced = pullback(w.node, L1e.node if hasattr(L1e, "node") else L1e, primitives=prim)
    prog = represent(traced, basis_env={w.node.uid: pin_c}, target=tgt_e)
    for Jc, Je in ((np.array([[2.0, -0.5], [1.0, 2.0]]), np.array([[2.0], [1.0]])),
                   (np.eye(2), np.array([[1.0], [0.0]])),
                   (2 * np.eye(2), np.array([[2.0], [0.0]]))):
        feed = {}
        for i in prog.inputs:
            feed[i.name] = {"jc": Jc.T, "je": Je.T, "w": np.array([1.0, 0.0])}[i.name.rsplit("_", 1)[0]]
        out = np.asarray(prog.run(feed)["result"]).ravel()
        assert np.allclose(out, [1.0], atol=1e-12), (Jc, Je, out)
