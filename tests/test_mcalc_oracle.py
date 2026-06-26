"""Validate the ``mcalc`` ASCII frontend's derivatives against matrixcalculus.org.

matrixcalculus.org is used here as a **numeric oracle**: for each gold example we ask its
``/_show`` endpoint to symbolically differentiate the expression, ``exec`` the returned
``codeExport.python`` to get a runnable ``fAndG(...) -> (value, gradient)``, evaluate it on
fixed-seed random inputs, and compare its ``gradient`` to OUR gradient (obtained by lowering
+ running the mcalc-built derivative map). See ``tools/mcalc_oracle.py`` for the client +
offline fixture cache.

Running:
    pytest tests/test_mcalc_oracle.py -q                 # offline, uses cached fixtures
    GRASSMANN_MCALC_LIVE=1 pytest tests/test_mcalc_oracle.py -q -m network   # refresh live

Two conventions were pinned during bring-up (see ``test_matrix_apply_orientation`` and the
``A*exp(x)`` xfail below):

* **Matrix-apply orientation.** Our ``mat * vec`` (``A*x``) lowers as ``Aᵀ·x`` — a matrix
  *variable* is consumed transposed relative to matrixcalculus/numpy (``A·x``). So for any
  matrix-bearing scalar example we feed the matrix *transposed* into our engine to align the
  two linear maps before comparing gradients (``MAT_TRANSPOSE`` per example). Symmetric forms
  (``0.5*x'*A*x``) match either way.
* **Vector-valued full-Jacobian.** The full Jacobian *matrix* returned for a vector-valued
  map is mis-oriented (``A*exp(x)`` → ``diag(exp x)·A`` instead of ``A·diag(exp x)``), so that
  example is ``xfail``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import grassmann as G
from grassmann.syntax.mcalc import McalcBuildError, build_problem

# Make ``tools/mcalc_oracle.py`` importable.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))
import mcalc_oracle as O  # noqa: E402

LIVE = O.live_enabled()
N = 4          # concrete dimension all symbolic dims are instantiated to
RTOL = 1e-5
ATOL = 1e-7


# ---------------------------------------------------------------------------
# Gold examples.  ``decls`` is an *ordered* name -> type map (insertion order is the
# varList order we send; the server alphabetizes it and ``fAndG`` follows that order, which
# we recover from the response's ``varList``).  ``mat_transpose`` aligns the matrix-apply
# convention (see module docstring).
# ---------------------------------------------------------------------------
GOLD = [
    dict(id="bilinear",  expr="0.5*x'*A*x",     wrt="x",
         decls={"A": "matrix", "x": "vector"},                       mat_transpose=False),
    dict(id="sin_dot",   expr="sin(x)'*y",      wrt="x",
         decls={"x": "vector", "y": "vector"},                       mat_transpose=False),
    dict(id="hadamard",  expr="(y.*v)'*x",      wrt="x",
         decls={"x": "vector", "y": "vector", "v": "vector"},        mat_transpose=False),
    dict(id="power",     expr="a^b",            wrt="a",
         decls={"a": "scalar", "b": "scalar"},                       mat_transpose=False),
    dict(id="lsq",       expr="norm2(A*x-y)^2", wrt="x",
         decls={"A": "matrix", "x": "vector", "y": "vector"},        mat_transpose=True),
]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _rand(ty: str, n: int, rng: np.random.Generator):
    """A random input for a declared var type. Scalars are kept positive so ``a^b`` (which
    lowers through ``log a``) is well defined."""
    if ty == "scalar":
        return float(rng.uniform(0.5, 2.5))
    if ty == "vector":
        return rng.standard_normal(n)
    M = rng.standard_normal((n, n))
    return (M + M.T) / 2 if ty == "symmetric matrix" else M


def _inputs(decls: dict[str, str], n: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {name: _rand(ty, n, rng) for name, ty in decls.items()}


def _our_gradient(expr, wrt, decls, inputs, n, mat_transpose):
    """OUR gradient: build + diff with mcalc, instantiate symbolic dims to ``n``, run the
    derivative map (the full-Jacobian lowering returns the gradient/Jacobian directly)."""
    r = build_problem(expr, wrt=wrt, decls=decls)
    der = r.instantiate(r.derivative, {d: n for d in r.dim_names})
    feed = {}
    for name, ty in decls.items():
        val = inputs[name]
        if mat_transpose and ty in ("matrix", "symmetric matrix"):
            val = np.asarray(val).T
        feed[r.input_key(name)] = val
    return np.asarray(G.run(der, feed)["result"])


def _oracle(example):
    """Fetch (cached) the oracle response; skip cleanly when offline + no fixture."""
    decls = example["decls"]
    varList = [{"name": k, "type": t} for k, t in decls.items()]
    wrt = {"name": example["wrt"], "type": decls[example["wrt"]]}
    try:
        return O.query_show(example["expr"], wrt, varList, allow_network=LIVE)
    except FileNotFoundError as e:
        pytest.skip(f"no cached oracle fixture (set GRASSMANN_MCALC_LIVE=1 to populate): {e}")


def _oracle_gradient(resp, inputs):
    """Evaluate the oracle's ``fAndG`` on ``inputs``. ``fAndG`` takes vars in the *response*
    ``varList`` order (the server alphabetizes the names)."""
    fandg = O.load_fandg(resp)
    args = [inputs[name] for name in resp["varList"]]
    _value, gradient = fandg(*args)
    return np.asarray(gradient, dtype=float)


# ---------------------------------------------------------------------------
# The gold sweep.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("example", GOLD, ids=[e["id"] for e in GOLD])
def test_gradient_matches_oracle(example):
    resp = _oracle(example)
    assert not resp.get("errorString"), resp.get("errorString")
    inputs = _inputs(example["decls"], N, seed=20240626)
    ours = _our_gradient(example["expr"], example["wrt"], example["decls"],
                         inputs, N, example["mat_transpose"])
    oracle = _oracle_gradient(resp, inputs)
    np.testing.assert_allclose(np.reshape(ours, oracle.shape), oracle, rtol=RTOL, atol=ATOL)


@pytest.mark.xfail(strict=True, reason=(
    "vector-valued full-Jacobian is mis-oriented: A*exp(x) lowers to diag(exp x)·A instead "
    "of the correct A·diag(exp x) (engine seam, not mcalc). Tracked for represent.py."))
def test_vector_valued_jacobian_A_exp_x():
    decls = {"A": "matrix", "x": "vector"}
    example = dict(expr="A*exp(x)", wrt="x", decls=decls)
    resp = _oracle(example)
    inputs = _inputs(decls, N, seed=20240626)
    ours = _our_gradient("A*exp(x)", "x", decls, inputs, N, mat_transpose=False)
    oracle = _oracle_gradient(resp, inputs)
    np.testing.assert_allclose(ours, oracle, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Convention pin (no network): documents that ``A*x`` lowers as ``Aᵀ·x``.
# ---------------------------------------------------------------------------
def test_matrix_apply_orientation():
    """``mat * vec`` consumes the matrix variable *transposed* vs numpy/matrixcalculus:
    running ``A*x`` yields ``Aᵀ·x`` (so the lsq example feeds ``Aᵀ`` to align). This pins
    the convention the oracle sweep relies on — purely our engine, no oracle call."""
    rng = np.random.default_rng(0)
    x, A = rng.standard_normal(N), rng.standard_normal((N, N))
    r = build_problem("A*x", decls={"A": "matrix", "x": "vector"})
    prim = r.instantiate(r.primal, {d: N for d in r.dim_names})
    got = np.asarray(G.run(prim, {r.input_key("x"): x, r.input_key("A"): A})["result"])
    assert np.allclose(got, A.T @ x)
    assert not np.allclose(got, A @ x)


# ---------------------------------------------------------------------------
# ``det``/``logdet`` now build, diff, and run (gold example 9). Value vs numpy here; full
# oracle gradient comparison for matrix-wrt derivatives is covered in tests/test_mcalc.py.
# ---------------------------------------------------------------------------
def test_log_det_inv_runs():
    r = build_problem("log(det(inv(X)))", wrt="X", decls={"X": "matrix(n,n)"})
    assert r.derivative is not None
    rng = np.random.default_rng(9)
    X = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    assert np.allclose(r.value({"X": X}, {"n": 4}),
                       np.log(np.linalg.det(np.linalg.inv(X))))


# ---------------------------------------------------------------------------
# Live refresh — only runs with ``-m network`` AND GRASSMANN_MCALC_LIVE=1.
# ---------------------------------------------------------------------------
@pytest.mark.network
@pytest.mark.skipif(not LIVE, reason="set GRASSMANN_MCALC_LIVE=1 to hit matrixcalculus.org")
@pytest.mark.parametrize("example", GOLD, ids=[e["id"] for e in GOLD])
def test_refresh_fixture_live(example):
    decls = example["decls"]
    varList = [{"name": k, "type": t} for k, t in decls.items()]
    wrt = {"name": example["wrt"], "type": decls[example["wrt"]]}
    resp = O.query_show(example["expr"], wrt, varList, allow_network=True)
    assert not resp.get("errorString")
    assert O.fixture_path(example["expr"], wrt, varList).exists()
