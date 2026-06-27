"""`G.dot` on a DERIVED operand whose space is an exterior power `ΛᵏV = Img(alt(V,k))`.

`dot` reads the metric from the operand's space; a wedge VALUE carries no explicit `.space`, so
`_space_of` synthesizes it (keying `Inner` on the operand's own `alt` node — no memoization). This
asserts `dot(w, w)` compiles and runs for a wedge, via the native dynamic-rank inner lane.
"""

import numpy as np
import grassmann as G


def test_dot_on_a_wedge_value_compiles_and_runs():
    Re = G.space("Re", 3)
    a = G.var("a", G.exterior_power(Re, 1))
    b = G.var("b", G.exterior_power(Re, 1))
    w = G.wedge(a, b, 1, 1)                          # a 2-vector in Λ²(R³); no explicit .space
    prog = G.compile(G.dot(w, w))                    # _space_of synthesizes Λ²(R³) for the operand

    def key(v):
        return f"{v.node.name}_{v.node.uid}"
    out = prog.run({key(a): np.array([1.0, 0.0, 0.0]), key(b): np.array([0.0, 1.0, 0.0])})
    # Grassmann's tensor inner: ‖e₁∧e₂‖² = k!·det(Gram) = 2!·1 = 2 (the un-normalized convention)
    assert np.isclose(float(np.asarray(out["result"])), 2.0)
