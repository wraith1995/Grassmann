"""Batch driver: run the oracles over many seeded random cases, collect failures, shrink them.

Each trial ``i`` derives its own ``numpy`` ``Generator`` from ``(seed, i)`` so a reported
failing trial index reproduces exactly.  The runner is deterministic in ``(seed, n_trials,
methods, max_depth, dim)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import oracles as O
from .generate import generate
from .shrink import shrink

METHODS = ("ad_vs_fd", "rewrite", "identity")


@dataclass
class Failure:
    method: str
    trial: int
    seed: int
    detail: str
    term_repr: str
    shrunk_repr: str = ""


@dataclass
class Report:
    seed: int
    n_trials: int
    counts: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    @property
    def n_bugs(self) -> int:
        return len(self.failures)

    def summary(self) -> str:
        lines = [f"gfuzz report — seed={self.seed} trials={self.n_trials}"]
        for m in METHODS:
            c = self.counts.get(m, {})
            lines.append(f"  {m:10s}  ok={c.get('ok',0):4d}  skip={c.get('skip',0):4d}  BUG={c.get('bug',0):4d}")
        lines.append(f"  {'TOTAL BUGS':10s}  {self.n_bugs}")
        for f in self.failures:
            lines.append("")
            lines.append(f"  ✗ {f.method} trial#{f.trial} (reproduce: --seed {f.seed} --only {f.trial})")
            lines.append(f"    {f.detail}")
            if f.shrunk_repr:
                lines.append(f"    shrunk term: {f.shrunk_repr}")
            else:
                lines.append(f"    term: {f.term_repr}")
        return "\n".join(lines)


def _describe(term) -> str:
    try:
        import grassmann as G

        return G.unparse(term)
    except Exception:
        return repr(term)


def _trial_rng(seed: int, i: int) -> np.random.Generator:
    return np.random.default_rng([seed, i])


def _still_fails(method: str, case, rng_seed) -> bool:
    r = np.random.default_rng(rng_seed)
    if method == "ad_vs_fd":
        return O.ad_vs_fd(case, r).status == "bug"
    if method == "rewrite":
        return O.rewrite_invariant(case, r).status == "bug"
    return False  # identity method is operand-only, not case-based


def run(seed: int = 0, n_trials: int = 500, *, methods=METHODS, max_depth: int = 4,
        dim: int = 3, only: int | None = None, do_shrink: bool = True,
        quiet_warnings: bool = True) -> Report:
    rep = Report(seed=seed, n_trials=n_trials, counts={m: {"ok": 0, "bug": 0, "skip": 0} for m in methods})
    indices = [only] if only is not None else range(n_trials)
    ctx = warnings.catch_warnings() if quiet_warnings else _null()
    with ctx:
        if quiet_warnings:
            warnings.simplefilter("ignore", RuntimeWarning)
        for i in indices:
            r = _trial_rng(seed, i)
            case = generate(r, n=dim, max_depth=max_depth)
            if "ad_vs_fd" in methods:
                _record(rep, "ad_vs_fd", i, seed, O.ad_vs_fd(case, r), case, do_shrink,
                        lambda c: _still_fails("ad_vs_fd", c, [seed, i, 7]))
            if "rewrite" in methods:
                _record(rep, "rewrite", i, seed, O.rewrite_invariant(case, r), case, do_shrink,
                        lambda c: _still_fails("rewrite", c, [seed, i, 9]))
            if "identity" in methods:
                _record(rep, "identity", i, seed, O.identity_holds(r, n=dim), None, False, None)
    return rep


def _record(rep, method, i, seed, res, case, do_shrink, predicate):
    rep.counts[method][res.status] += 1
    if res.status != "bug":
        return
    term_repr = _describe(case.term) if case is not None else ""
    shrunk_repr = ""
    if do_shrink and case is not None and predicate is not None:
        try:
            small = shrink(case, predicate)
            shrunk_repr = _describe(small.term)
        except Exception:
            pass
    rep.failures.append(Failure(method=method, trial=i, seed=seed, detail=res.detail,
                                term_repr=term_repr, shrunk_repr=shrunk_repr))


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
