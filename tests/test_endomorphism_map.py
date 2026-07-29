"""Tests for :func:`grassmann.transform.pullback.endomorphism_map` — the functorial
extension of leaf SELF-maps (automorphisms) to composite spaces.

``transport_map`` short-circuits ``S == T`` to ``Identity``, so an automorphism — a
nontrivial self-map invisible to structural equality — needs this sibling functor. The
motivating consumer is the σ-branch reorientation of a FEEC weight's value space
(pointwise ``_TiledScalar``): the reference-cell automorphism's linear part ``A`` acts on
the tangent ground, and every composite value space (``Dual``/``⊗``/``⊕``/``Λᵏ``) must
carry the INDUCED action — ``Dual`` factors automatically become the pullback ``ω ↦ ω∘A``.

Conventions pinned here (empirically cross-checked on the savo two-tet S₃ face meshes):

* grassmann map matrices are ``(dom, cod)``, so the PUSHFORWARD leaf ``v ↦ A·v`` (column
  convention) is supplied as the External payload ``A.T``;
* the induced Dual action then stores exactly ``A``: ``(ω∘A)_i = Σ_j ω_j A_ji``;
* no determinant factor anywhere (the automorphism is unimodular; orientation signs ride
  the consumer's frame correction, not this functor).
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.lower.represent import ExternalOp
from grassmann.infer.synth import linear_map_space
from grassmann.transform.pullback import PullbackError, endomorphism_map

V = S.base("V", 3)
A = np.array([[1.0, 2.0, 0.5],
              [0.0, 1.5, -1.0],
              [2.0, 0.0, 1.0]])


def _prim(sp):
    """The leaf self-map: the pushforward ``v ↦ A·v`` on the ground ``V`` ((dom, cod) = A.T)."""
    if sp == V:
        return T.External(linear_map_space(V, V), ExternalOp("A", lambda m=A.T: m), ())
    return None


def _run(term: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(G.compile(term).run(feed)["result"])


def _apply(endo: T.Term, space: S.Space, coords: np.ndarray) -> np.ndarray:
    v = T.var("v", space)
    return _run(T.Eval(endo, (0,), (v,)), {f"v_{v.uid}": coords})


def test_primal_leaf_is_the_pushforward() -> None:
    endo = endomorphism_map(V, primitives=_prim)
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(_apply(endo, V, x), A @ x)


def test_dual_is_the_pullback_transpose_action() -> None:
    endo = endomorphism_map(S.Dual(V), primitives=_prim)
    w = np.array([0.7, -1.3, 0.4])
    np.testing.assert_allclose(_apply(endo, S.Dual(V), w), w @ A)   # (ω∘A)_i = Σ_j ω_j A_ji


def test_oprod_is_the_kron_of_factor_actions() -> None:
    VV = S.OProd((V, V))
    endo = endomorphism_map(VV, primitives=_prim)
    M = np.arange(9.0).reshape(3, 3) + 1.0
    np.testing.assert_allclose(_apply(endo, VV, M), A @ M @ A.T)    # (A⊗A)(x⊗y) = Ax ⊗ Ay


def test_oplus_is_block_diagonal() -> None:
    W = S.oplus(V, V)
    endo = endomorphism_map(W, primitives=_prim)
    x = np.array([1.0, -2.0, 3.0, 0.5, 4.0, -1.0])
    expected = np.concatenate([A @ x[:3], A @ x[3:]])
    np.testing.assert_allclose(_apply(endo, W, x), expected)


def test_mixed_dual_tensor_factor() -> None:
    """``Dual(V) ⊗ V`` — one covariant (pullback) and one contravariant (pushforward) slot."""
    DV = S.OProd((S.Dual(V), V))
    endo = endomorphism_map(DV, primitives=_prim)
    M = np.arange(9.0).reshape(3, 3) - 4.0
    np.testing.assert_allclose(_apply(endo, DV, M), A.T @ M @ A.T)  # slot0: ω↦ωA; slot1: v↦Av


def test_lambda2_carries_the_induced_compound_action() -> None:
    """``Λ²(V)`` (via stdlib ``exterior_power``) — the induced action is the rank-2 compound
    of the leaf action, exercised through the projector-``Img(Alt)`` conjugation arm.

    Oracle through the ambient: for an ANTISYMMETRIC ``M ∈ V⊗²``, project into ``Λ²``, act,
    embed back — the round trip must equal ``(A⊗A)M = A·M·Aᵀ`` (antisymmetry is preserved by
    the Kron action, so nothing is lost in the projection)."""
    from grassmann.stdlib.exterior import exterior_power

    node = exterior_power(_expr(V), 2).node
    endo = endomorphism_map(node, primitives=_prim)
    VV = S.OProd((V, V))
    M = np.array([[0.0, 2.0, -1.0],
                  [-2.0, 0.0, 3.0],
                  [1.0, -3.0, 0.0]])
    x = T.var("x", VV)
    roundtrip = T.Embed(T.Eval(endo, (0,), (T.Project(x, node),)), VV)
    out = _run(roundtrip, {f"x_{x.uid}": M})
    np.testing.assert_allclose(out, A @ M @ A.T, atol=1e-12)


def test_ground_with_no_leaf_action_is_the_identity() -> None:
    """A Scalar ground (``S.base``) with no supplied primitive reduces to the identity — the
    documented convention (an endomorphism with no leaf action IS the identity)."""
    W = S.base("W", 4)
    endo = endomorphism_map(W, primitives=lambda sp: None)
    x = np.array([1.0, -2.0, 3.0, 0.5])
    np.testing.assert_allclose(_apply(endo, W, x), x)


def test_unsupported_ffs_raises() -> None:
    """``Ker`` has no structural induced self-map — a loud PullbackError, not a silent identity."""
    f = T.var("f", linear_map_space(V, V))
    with pytest.raises(PullbackError):
        endomorphism_map(S.Ker(f), primitives=lambda sp: None)


def _expr(node):
    from grassmann.embed.ops import Expr
    return Expr(node)
