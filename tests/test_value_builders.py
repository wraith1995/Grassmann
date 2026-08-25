"""Value-level builders that read their space off the value.

``const``, ``proj_of``, ``sharp_of`` and ``flat_of`` let a consumer act on a value
without naming the space a second time; these tests pin their results against the
space-first forms they abbreviate.
"""

import numpy as np

import grassmann as G


def _run(term):
    return G.run(term, {})["result"]


def test_const_builds_the_coordinate_vector():
    V = G.space("V", 3)
    got = _run(G.const(V, [0.0, 1.0, 0.0]))
    np.testing.assert_allclose(got, [0.0, 1.0, 0.0])


def test_proj_of_selects_the_summand():
    V, W = G.space("V", 2), G.space("W", 3)
    z = G.ostack(G.const(V, [0.0, 1.0]), G.const(W, [2.0, 3.0, 4.0]))
    np.testing.assert_allclose(_run(G.proj_of(z, 0)), [0.0, 1.0])
    np.testing.assert_allclose(_run(G.proj_of(z, 1)), [2.0, 3.0, 4.0])


def test_of_forms_match_the_space_first_forms():
    V = G.space("V", 3)
    x = G.const(V, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(_run(G.flat_of(x)), _run(G.flat(V)(x)))
    xflat = G.flat_of(x)
    np.testing.assert_allclose(_run(G.sharp_of(xflat)), _run(x))
