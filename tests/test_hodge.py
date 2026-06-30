"""The Hodge star ``⋆ : Λᵏ(V) → Λⁿ⁻ᵏ(V)`` (`stdlib.exterior.hodge_star`).

Built coordinate-free as ``⋆ = pinv(M) ∘ flat`` with ``M`` the wedge-into-volume pairing; the
metric is induced by the ``Λᵏ ⊑ V⊗ᵏ`` embedding, and the exterior powers carry RUNTIME (SVD-rank)
dimensions — so this also exercises the dynamic-shape lowering of ``pinv``/``llam``-vmap/``eval``."""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
import grassmann.embed.builders as B
from grassmann.lower.represent import ExternalOp
from grassmann.stdlib.exterior import exterior_power, hodge_star, wedge


def _pair(s, a, b):
    return B.inner(s).at([0, 1])(a, b)


def _vol_unit(V, n):
    """The unit volume form ``e₁∧…∧eₙ`` (standard orientation), normalised so ``⟨vol,vol⟩=1``."""
    Ln = exterior_power(V, n)

    def e(i):
        a = np.zeros(n); a[i] = 1.0
        return G.external(V, ExternalOp(f"e{i}", (lambda a=a: a)))

    vol = G.project(G.tensor(*[e(i) for i in range(n)]), Ln)
    return Ln, G.scale(G.reciprocal(G.sqrt(_pair(Ln, vol, vol))), vol)


CASES = [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)]


@pytest.mark.parametrize("n,k", CASES)
def test_hodge_defining_property(n, k):
    """``α ∧ ⋆β == ⟨α, β⟩ · vol`` for all ``α, β ∈ Λᵏ`` (the characterising identity)."""
    from math import comb

    V = G.space(f"R{n}", n)
    Lk = exterior_power(V, k)
    Ln, vol_unit = _vol_unit(V, n)
    alpha, beta = G.var("alpha", Lk), G.var("beta", Lk)
    star = hodge_star(beta, k)
    lhs = _pair(Ln, wedge(alpha, star, k, n - k), vol_unit)   # ⟨α∧⋆β, vol⟩
    rhs = _pair(Lk, alpha, beta)                              # ⟨α, β⟩
    ak, bk = f"alpha_{alpha.node.uid}", f"beta_{beta.node.uid}"
    rng = np.random.default_rng(10 * n + k)
    d = comb(n, k)
    for _ in range(4):
        av, bv = rng.standard_normal(d), rng.standard_normal(d)
        lo = float(G.run(lhs, {ak: av, bk: bv})["result"])
        ro = float(G.run(rhs, {ak: av, bk: bv})["result"])
        assert np.isclose(lo, ro, atol=1e-8)


@pytest.mark.parametrize("n,k", CASES)
def test_hodge_double_star(n, k):
    """``⋆`` squares to a SIGNED multiple of the identity on ``Λᵏ``: ``⋆⋆ = (-1)^{k(n-k)}/C(n,k)·id``.

    The sign ``(-1)^{k(n-k)}`` is the textbook Euclidean double-star sign (from the wedge's
    anticommutativity). The ``1/C(n,k)`` factor is the engine's INDUCED ``Λᵏ`` metric
    normalization (``⟨a∧b,a∧b⟩ = 2|a×b|²``, i.e. ``k!``-scaled vs. the textbook orthonormal
    metric) — consistent with the defining property above, which uses that same metric. Checked
    basis-independently via ``⟨⋆⋆β, γ⟩ == c·⟨β, γ⟩`` (coords would be fragile to the FFS basis)."""
    from math import comb

    V = G.space(f"R{n}", n)
    Lk = exterior_power(V, k)
    beta, gamma = G.var("beta", Lk), G.var("gamma", Lk)
    ss = hodge_star(hodge_star(beta, k), n - k)
    c = (-1) ** (k * (n - k)) / comb(n, k)
    lhs = _pair(Lk, ss, gamma)
    rhs = G.scale(float(c), _pair(Lk, beta, gamma))
    bk, gk = f"beta_{beta.node.uid}", f"gamma_{gamma.node.uid}"
    rng = np.random.default_rng(100 * n + k)
    d = comb(n, k)
    for _ in range(3):
        bv, gv = rng.standard_normal(d), rng.standard_normal(d)
        lo = float(G.run(lhs, {bk: bv, gk: gv})["result"])
        ro = float(G.run(rhs, {bk: bv, gk: gv})["result"])
        assert np.isclose(lo, ro, atol=1e-7)


def test_hodge_grade_bounds():
    V = G.space("R3", 3)
    with pytest.raises(ValueError):
        hodge_star(G.var("x", exterior_power(V, 1)), 0)
    with pytest.raises(ValueError):
        hodge_star(G.var("x", exterior_power(V, 1)), 3)
