"""Smoke + teeth tests for the random differential fuzzer (:mod:`gfuzz`).

* **No *unexpected* silent-wrong** over a small deterministic batch.  The fuzzer currently
  surfaces one *known* grassmann defect (``diff`` of a direct-sum-valued term that is
  independent of the differentiation variable builds an invalid ``LLam`` instead of the
  zero map — see ``KNOWN_BUGS``).  The test asserts every bug found matches a known
  signature, so a *new* regression turns it red while the documented one does not.

* **The harness has teeth** — with a deliberately corrupted ``diff`` (a Jacobian scaled by
  1.5), the AD-vs-FD oracle *does* report a bug.  A fuzzer that can never fail is worthless.

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

# Substrings that identify a *known, reported* grassmann defect (not a fuzzer bug).  Each is a
# real crash the fuzzer surfaces on the current tree; the test allows them so a *new*
# regression still turns red.  When one is fixed upstream, its entry simply goes unused.
KNOWN_BUGS = (
    # (A — diff(OPlusV, x) for x-independent F — is FIXED in transform/diff.py; a regression
    #  would re-introduce "LLam: linear variable …" and this test would then correctly fail.)
    # B: an OPlus/direct-sum lowers to an inconsistent total dimension (synth accepts)
    "bulk Stmt output",
    # C: the same space realizes to different numeric shapes (unit tensor axes) → Sum assert
    "assert:sum-shapes",
)


def _is_known(detail: str) -> bool:
    return any(k in detail for k in KNOWN_BUGS)


def test_only_known_bugs_in_small_batch():
    """A small seeded batch finds no *new* silent-wrong result (known defects are allowed)."""
    rep = run(seed=0, n_trials=80, do_shrink=False)
    unexpected = [f for f in rep.failures if not _is_known(f.detail)]
    assert not unexpected, rep.summary()
    for m in ("ad_vs_fd", "rewrite", "typecheck", "identity"):
        assert rep.counts[m]["ok"] > 0, f"{m} produced no ok results: {rep.counts[m]}"


def test_ad_oracle_has_teeth(monkeypatch):
    """A 1.5×-wrong Jacobian must be caught by ad_vs_fd.  Uses a fixed term with a known
    nonzero Jacobian (⟨Mx, x⟩) so the test does not depend on the generator's draws."""
    import grassmann.core.space as S
    from gfuzz.generate import Case

    orig = O.diff

    def wrong(e, x, deps=None):
        J = orig(e, x, deps)
        if isinstance(J, T.Zero):
            return J
        return T.LLam(J.vars, body=T.scale(1.5, J.body))

    monkeypatch.setattr(O, "diff", wrong)
    V = S.base("X", 3)
    x = T.var("x", V)
    M = T.var("M", S.oprod(S.dual(V), V))
    F = T.Eval(T.inner(V), (0, 1), (T.Eval(M, (0,), (x,)), x))  # ⟨Mx, x⟩
    rng = np.random.default_rng(0)
    feed0 = {f"x_{x.uid}": rng.standard_normal(3), f"M_{M.uid}": rng.standard_normal((3, 3))}
    case = Case(term=F, x=x, x0=feed0[f"x_{x.uid}"], feed0=feed0, x_name=f"x_{x.uid}", n=3)
    assert O.ad_vs_fd(case, rng).status == "bug", "planted 1.5x Jacobian went undetected — oracle is blind"


def test_reproducible_by_index():
    """A ``(seed, index)`` reproduces the same case: same structure, names, and feed values."""
    import grassmann as G

    c1 = generate(np.random.default_rng([0, 7]))
    c2 = generate(np.random.default_rng([0, 7]))
    assert G.unparse(c1.term) == G.unparse(c2.term)
    assert np.array_equal(c1.x0, c2.x0)
    # feed *keys* carry a process-global uid, so compare the values order-independently.
    def sig(feed):
        return sorted((v.shape, round(float(v.sum()), 9)) for v in feed.values())
    assert sig(c1.feed0) == sig(c2.feed0)
