"""Smoke + teeth tests for the random differential fuzzer (:mod:`gfuzz`).

Two things are asserted:

* **No silent-wrong** over a small deterministic batch — the fuzzer finds no bug on the
  current tree (the permanent-guard direction; a real regression would turn this red).

* **The harness has teeth** — with a deliberately corrupted ``diff`` (a Jacobian scaled by
  1.5), the AD-vs-FD oracle *does* report a bug.  This guards the guard: a fuzzer that can
  never fail is worthless, so we prove it fails when the thing under test is wrong.

``gfuzz`` lives at the grassmann repo root (not under ``src``), so it imports only when the
tests run with the repo root on ``sys.path`` (pytest's rootdir insertion handles this).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gfuzz", reason="gfuzz is a repo-root tool package; needs rootdir on sys.path")

import grassmann.core.term as T  # noqa: E402
import gfuzz.oracles as O  # noqa: E402
from gfuzz.generate import generate  # noqa: E402
from gfuzz.runner import run  # noqa: E402


def test_no_silent_wrong_small_batch():
    """A small seeded batch across all three methods finds no silent-wrong result."""
    rep = run(seed=0, n_trials=60, do_shrink=False)
    assert rep.n_bugs == 0, rep.summary()
    # sanity: each method actually did work (not all skipped)
    for m in ("ad_vs_fd", "rewrite", "identity"):
        c = rep.counts[m]
        assert c["ok"] > 0, f"{m} produced no ok results: {c}"


def test_ad_oracle_has_teeth(monkeypatch):
    """A 1.5×-wrong Jacobian must be caught by ad_vs_fd on at least one random case."""
    orig = O.diff

    def wrong(e, x, deps=None):
        J = orig(e, x, deps)
        if isinstance(J, T.Zero):
            return J
        return T.LLam(J.vars, body=T.scale(1.5, J.body))

    monkeypatch.setattr(O, "diff", wrong)
    caught = 0
    for i in range(40):
        r = np.random.default_rng([0, i])
        case = generate(r)
        if O.ad_vs_fd(case, r).status == "bug":
            caught += 1
    assert caught > 0, "planted 1.5x Jacobian bug went undetected — the oracle is blind"


def test_reproducible_by_index():
    """A ``(seed, index)`` reproduces the same case: same structure, names, and feed values.

    (``Var.uid`` is a process-global counter, so raw ``repr`` differs between runs; the
    *structure* — which ``unparse`` renders by name — and the sampled values do not.)"""
    import grassmann as G

    r1 = np.random.default_rng([0, 7])
    r2 = np.random.default_rng([0, 7])
    c1, c2 = generate(r1), generate(r2)
    assert G.unparse(c1.term) == G.unparse(c2.term)
    assert np.array_equal(c1.x0, c2.x0)
    v1 = sorted((v.name, val.tolist()) for v, val in c1.consts.items())
    v2 = sorted((v.name, val.tolist()) for v, val in c2.consts.items())
    assert v1 == v2
