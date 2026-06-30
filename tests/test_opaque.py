"""Opaque functions: declare a black-box map ``f : A → B`` (optional numpy kernel), apply it in a
term, and differentiate ``f(x)`` via an engine-auto-allocated Jacobian tracer input.

These tests pin, with a concrete opaque map whose true derivative is known (``sin``, with the engine
treating it as fully opaque):
* primal — ``f(x)`` lowers + runs to the kernel value ``sin(x0)``;
* synth — ``f(x) : V``; a wrong-space argument is rejected;
* derivative — ``diff(⟨f(x), y⟩, x)`` references ONE Jacobian input ``Df`` (``source.name == "f"``,
  order 1), surfaced by :func:`enumerate_derivative_inputs`, and fed the true Jacobian
  ``diag(cos x0)`` reproduces the analytic ``⟨diag(cos x0)·dx, y⟩``;
* no regression — diffing a term without any opaque var still works.
"""

from __future__ import annotations

import numpy as np

import grassmann as G
from grassmann.transform import clear_deriv_inputs, enumerate_derivative_inputs


def test_opaque_primal_runs_kernel() -> None:
    n = 4
    V = G.space("V", n)
    f = G.opaque("f", V, V, fn=np.sin)
    x = G.var("x", V)

    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(n)
    out = np.asarray(G.run(f(x), {f"x_{x.node.uid}": x0})["result"])
    np.testing.assert_allclose(out, np.sin(x0), atol=1e-12)


def test_opaque_synth_space_and_wrong_arg() -> None:
    from grassmann.infer.synth import InferError, synth_space

    n = 3
    V = G.space("V", n)
    W = G.space("W", n + 1)
    f = G.opaque("f", V, V)
    x = G.var("x", V)
    w = G.var("w", W)

    assert synth_space(f(x).node) == V.node  # f(x) : V

    try:
        synth_space(f(w).node)  # wrong domain
    except InferError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected an InferError for a wrong-space opaque argument")


def test_opaque_no_kernel_cannot_run() -> None:
    from grassmann.lower.represent import RepresentError

    n = 2
    V = G.space("V", n)
    f = G.opaque("f", V, V)  # no fn
    x = G.var("x", V)
    try:
        G.compile(f(x))
    except RepresentError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected a RepresentError running an opaque op with no kernel")


def test_opaque_derivative_enumerates_jacobian_and_matches_analytic() -> None:
    clear_deriv_inputs()
    n = 4
    V = G.space("V", n)
    f = G.opaque("f", V, V, fn=np.sin)
    x = G.var("x", V)
    y = G.var("y", V)

    e = G.dot(f(x), y)  # ⟨f(x), y⟩
    jd = G.diff(e, x)  # an LLam over the seed h

    recs = enumerate_derivative_inputs(jd.node)
    assert len(recs) == 1
    (rec,) = recs
    assert rec.source.name == "f"
    assert rec.order == 1

    seed = jd.node.vars[0]
    rng = np.random.default_rng(11)
    x0 = rng.standard_normal(n)
    y0 = rng.standard_normal(n)
    dF = np.diag(np.cos(x0))  # the TRUE Jacobian of sin at x0, fed numerically

    body = G.compile(jd.node.body)
    feed_base = {
        f"x_{x.node.uid}": x0,
        f"y_{y.node.uid}": y0,
        f"{rec.var.name}_{rec.var.uid}": dF,
    }
    for i in range(n):
        h = np.zeros(n)
        h[i] = 1.0
        got = float(np.asarray(body.run({**feed_base, f"{seed.name}_{seed.uid}": h})["result"]))
        want = (dF @ h) @ y0  # ⟨diag(cos x0)·dx, y⟩
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_diff_without_opaque_var_still_works() -> None:
    clear_deriv_inputs()
    n = 3
    V = G.space("V", n)
    x = G.var("x", V)
    y = G.var("y", V)

    e = G.dot(x, y)  # ⟨x, y⟩
    jd = G.diff(e, x)
    assert enumerate_derivative_inputs(jd.node) == []

    seed = jd.node.vars[0]
    rng = np.random.default_rng(5)
    x0, y0 = rng.standard_normal(n), rng.standard_normal(n)
    body = G.compile(jd.node.body)
    for i in range(n):
        h = np.zeros(n)
        h[i] = 1.0
        feed = {f"x_{x.node.uid}": x0, f"y_{y.node.uid}": y0, f"{seed.name}_{seed.uid}": h}
        got = float(np.asarray(body.run(feed)["result"]))
        np.testing.assert_allclose(got, h @ y0, atol=1e-12)  # d⟨x,y⟩·h = ⟨h, y⟩
