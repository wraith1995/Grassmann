"""Cauchy–Binet: ⟨u₁∧…∧u_k, v₁∧…∧v_k⟩ = k!·det[⟨uᵢ, vⱼ⟩] equals the Λᵏ-coordinate pairing."""
from __future__ import annotations

import math

import numpy as np
import pytest

import grassmann as G
import polyarray as pa
from grassmann.stdlib.exterior import exterior_power, inner_of_wedges, wedge


def _value(term, vals):
    prog = pa.strip_asserts(G.compile(term))
    feed = {i.name: vals[i.name.rsplit("_", 1)[0]] for i in prog.inputs}
    return float(np.asarray(prog.run(feed)["result"]))


@pytest.mark.parametrize("n,k", [(3, 2), (4, 2), (4, 3)])
def test_cauchy_binet_matches_coordinate_pairing(n, k):
    V = G.space(f"R{n}", n)
    names = "abcdef"[: 2 * k]
    vs = {s: G.var(s, V) for s in names}
    us, ws = [vs[s] for s in names[:k]], [vs[s] for s in names[k:]]

    def wedge_all(xs):
        w = xs[0]
        for j, x in enumerate(xs[1:], start=1):
            w = wedge(w, x, j, 1)
        return w

    coord = G.inner(exterior_power(V, k)).at([0, 1])(wedge_all(us), wedge_all(ws))
    vals = {s: np.random.default_rng(7).standard_normal(n) * (i + 1) for i, s in enumerate(names)}
    want, got = _value(coord, vals), _value(inner_of_wedges(V, us, ws), vals)
    assert abs(got - want) <= 1e-12 * max(1.0, abs(want))
    # and the closed form itself: k!·det of the Gram matrix of the plain dot products
    gram = np.array([[vals[u] @ vals[w] for w in names[k:]] for u in names[:k]])
    assert abs(got - math.factorial(k) * np.linalg.det(gram)) <= 1e-12 * max(1.0, abs(got))
