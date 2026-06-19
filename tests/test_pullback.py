"""Tests for the auto-pullback transform (:mod:`grassmann.transform.pullback`).

``pullback(v, T)`` transports a value into a target space, inserting projections /
embeddings and recursing structurally through ⊗ / ⊕ / Dual (contravariant). Each case
lowers through :func:`grassmann.run` and is checked against the numpy reference.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.transform.pullback import PullbackError, pullback, transport_map


def _key(uid: int, name: str) -> str:
    return f"{name}_{uid}"


def _run(term: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(G.compile(term).run(feed)["result"])


def test_equal_space_is_identity() -> None:
    V = S.base("V", 3)
    x = T.var("x", V)
    # equal spaces ⇒ the value is returned unchanged (no spurious Eval).
    assert pullback(x, V) is x
    assert isinstance(transport_map(V, V), T.Identity)


def test_project_onto_summand() -> None:
    V, W = S.base("V", 3), S.base("W", 2)
    U = S.oplus(V, W)
    u = T.var("u", U)
    out = _run(pullback(u, V), {_key(u.uid, "u"): np.array([1.0, 2, 3, 4, 5])})
    np.testing.assert_allclose(out, [1, 2, 3])


def test_embed_into_summand_pads() -> None:
    V, W = S.base("V", 3), S.base("W", 2)
    U = S.oplus(V, W)
    v = T.var("v", V)
    out = _run(pullback(v, U), {_key(v.uid, "v"): np.array([7.0, 8, 9])})
    np.testing.assert_allclose(out, [7, 8, 9, 0, 0])


def test_embed_then_project_is_identity_on_subspace() -> None:
    V, W = S.base("V", 3), S.base("W", 2)
    U = S.oplus(V, W)
    v = T.var("v", V)
    roundtrip = pullback(pullback(v, U), V)
    out = _run(roundtrip, {_key(v.uid, "v"): np.array([7.0, 8, 9])})
    np.testing.assert_allclose(out, [7, 8, 9])


def test_tensor_pullback_is_componentwise() -> None:
    a, b = S.base("a", 2), S.base("b", 1)
    A = S.oplus(a, b)  # dim 3
    z = T.var("z", S.oprod(A, A))  # dim 9, shape (3,3)
    M = np.arange(9.0).reshape(3, 3)
    out = _run(pullback(z, S.oprod(a, a)), {_key(z.uid, "z"): M})
    # project each factor onto its `a` block (first two coords).
    np.testing.assert_allclose(out.reshape(2, 2), M[:2, :2])


def test_oplus_block_pullback() -> None:
    a, b = S.base("a", 2), S.base("b", 1)
    A = S.oplus(a, b)  # 3
    b2 = S.base("b2", 4)
    src, dst = S.oplus(A, b2), S.oplus(a, b2)  # 7 -> 6
    y = T.var("y", src)
    yv = np.arange(7.0)
    out = _run(pullback(y, dst), {_key(y.uid, "y"): yv})
    np.testing.assert_allclose(out, np.concatenate([yv[:2], yv[3:7]]))


def test_dual_pullback_is_contravariant_adjunction() -> None:
    # ⟨pullback(c, ~V), x⟩ == ⟨c, embed(x, U)⟩  — the defining adjunction.
    V, W = S.base("V", 3), S.base("W", 2)
    U = S.oplus(V, W)
    c = T.var("c", S.dual(U))
    x = T.var("x", V)
    pc = pullback(c, S.dual(V))  # a covector in ~V
    paired = T.Eval(pc, (0,), (x,))  # ⟨pullback(c), x⟩
    cval = np.array([1.0, 2, 1, 3, 1])
    xval = np.array([2.0, 3, 4])
    out = _run(paired, {_key(c.uid, "c"): cval, _key(x.uid, "x"): xval})
    np.testing.assert_allclose(out, cval @ np.array([2, 3, 4, 0, 0]))


def test_dim_equal_unrelated_passes_through() -> None:
    P, Q = S.base("P", 3), S.base("Q", 3)
    p = T.var("p", P)
    out = _run(pullback(p, Q), {_key(p.uid, "p"): np.array([5.0, 6, 7])})
    np.testing.assert_allclose(out, [5, 6, 7])


def test_unequal_unrelated_raises() -> None:
    with pytest.raises(PullbackError):
        transport_map(S.base("P", 3), S.base("Q", 4))


def test_symbolic_oplusrep_raises() -> None:
    V = S.base("V", 2)
    rep = S.oplus_rep(S.DimVar("n"), V)
    with pytest.raises(PullbackError):
        transport_map(rep, V)


def test_exterior_power_roundtrip() -> None:
    # A Λᵏ(V) = Img(alt) value embeds into its ambient V⊗ᵏ and projects back — identity.
    import grassmann.stdlib as L

    V = G.space("V", 3)
    a, b = G.var("a", L.exterior_power(V, 1)), G.var("b", L.exterior_power(V, 1))
    w = L.wedge(a, b, 1, 1)  # a∧b : Λ²(V)
    lam2 = G.infer.synth_space(w.node)
    vk = L.tensor_power(V, 2).node
    amb = pullback(w.node, vk)  # Λ² ↪ V⊗²
    back = pullback(amb, lam2)  # project back
    feed = {_key(a.node.uid, "a"): np.random.randn(3), _key(b.node.uid, "b"): np.random.randn(3)}
    np.testing.assert_allclose(_run(back, feed), _run(w.node, feed))
    # the ambient is the antisymmetric tensor a⊗b − b⊗a.
    amb_val = _run(amb, feed)
    np.testing.assert_allclose(amb_val, -amb_val.T, atol=1e-8)


def test_exterior_functoriality_across_base_spaces() -> None:
    # ι*(a∧b) = (ιa)∧(ιb): pull a 2-form on V up to Λ²(U) along V ↪ U = V ⊕ W.
    import grassmann.stdlib as L

    V, W = G.space("V", 2), G.space("W", 2)
    U = G.osum(V, W)
    a, b = G.var("a", L.exterior_power(V, 1)), G.var("b", L.exterior_power(V, 1))
    w = L.wedge(a, b, 1, 1)
    up = pullback(w.node, L.exterior_power(U, 2).node)  # Λ²(V) → Λ²(U)
    av, bv = np.random.randn(2), np.random.randn(2)
    feed = {_key(a.node.uid, "a"): av, _key(b.node.uid, "b"): bv}
    assert _run(up, feed).shape == (6,)  # |Λ²(U)| = C(4,2) = 6
    up_amb = _run(pullback(up, L.tensor_power(U, 2).node), feed)  # back to U⊗²
    aU, bU = np.concatenate([av, [0, 0]]), np.concatenate([bv, [0, 0]])
    np.testing.assert_allclose(up_amb, np.outer(aU, bU) - np.outer(bU, aU), atol=1e-8)


def test_surface_pullback_reachable() -> None:
    assert callable(G.pullback)
    V, W = G.space("V", 3), G.space("W", 2)
    U = G.osum(V, W)
    u = G.var("u", U)
    out = np.asarray(G.run(G.pullback(u, V), {_key(u.node.uid, "u"): np.array([1.0, 2, 3, 4, 5])})["result"])
    np.testing.assert_allclose(out, [1, 2, 3])
