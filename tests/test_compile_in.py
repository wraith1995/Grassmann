"""The public frame API: ``standard_basis`` and ``compile_in``.

``standard_basis`` builds a concrete frame that records the space it
coordinatizes; ``compile_in`` pins each variable and the result to a frame and
routes into ``represent``, rejecting a frame built for the wrong space.
"""

import numpy as np
import pytest

import grassmann as G
from grassmann.core.basis import BasisError


def _run(prog):
    feed = {"f": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), "x": np.array([2.0, 3.0])}
    out = prog.run({i.name: feed[i.name.split("_")[0]] for i in prog.inputs})
    return np.asarray(out["result"], float)


def test_standard_basis_records_its_space():
    W = G.space("W", 3)
    assert G.standard_basis(W).get_note("space") == W.node


def test_compile_in_runs_in_given_frames():
    V, W = G.space("V", 2), G.space("W", 3)
    f, x = G.var("f", V >> W), G.var("x", V)
    prog = G.compile_in(
        f(x),
        inputs={f: G.standard_basis(V >> W), x: G.standard_basis(V)},
        target=G.standard_basis(W),
    )
    assert np.allclose(_run(prog), [2.0, 3.0, 5.0])


def test_compile_in_default_target_chooses_a_basis():
    V, W = G.space("V", 2), G.space("W", 3)
    f, x = G.var("f", V >> W), G.var("x", V)
    prog = G.compile_in(f(x), inputs={f: G.standard_basis(V >> W), x: G.standard_basis(V)})
    assert np.allclose(_run(prog), [2.0, 3.0, 5.0])


def test_compile_in_rejects_wrong_space_input_frame():
    V, W = G.space("V", 2), G.space("W", 3)
    f, x = G.var("f", V >> W), G.var("x", V)
    with pytest.raises(BasisError, match="that variable lives in"):
        G.compile_in(
            f(x),
            inputs={f: G.standard_basis(V >> W), x: G.standard_basis(W)},
            target=G.standard_basis(W),
        )


def test_compile_in_rejects_wrong_space_target_frame():
    V, W = G.space("V", 2), G.space("W", 3)
    f, x = G.var("f", V >> W), G.var("x", V)
    with pytest.raises(BasisError, match="the result lives in"):
        G.compile_in(
            f(x),
            inputs={f: G.standard_basis(V >> W), x: G.standard_basis(V)},
            target=G.standard_basis(V),
        )
