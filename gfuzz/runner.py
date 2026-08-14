"""Batch driver: run the oracles over many seeded random cases, collect failures, shrink them.

Each trial ``i`` derives its own ``numpy`` ``Generator`` from ``(seed, i)`` so a reported
failing trial index reproduces exactly.  Any *unexpected* exception from an oracle (not a
grassmann bail) is caught per-trial and recorded as a **crash bug** — one bad draw never
aborts the campaign, and a crash is exactly the kind of defect a broad scan is hunting.

The runner also aggregates **skip reasons** (keyed by the leading tag of each skip detail),
so ``--skips`` can show what inference rejected and where lowering has gaps — the "look at
the rejected ones" view for auditing the type checker.
"""

from __future__ import annotations

import re
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import oracles as O
from .generate import generate
from .shrink import shrink

METHODS = ("ad_vs_fd", "diff2", "rewrite", "typecheck", "shape", "hook", "identity")


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
    skip_reasons: dict = field(default_factory=dict)   # method -> Counter(tag -> count)
    skip_examples: dict = field(default_factory=dict)  # (method, tag) -> [unparse, ...]
    bug_counts: Counter = field(default_factory=Counter)  # signature -> total hits

    @property
    def n_bugs(self) -> int:
        """Number of *distinct* bug signatures found."""
        return len(self.failures)

    @property
    def n_bug_hits(self) -> int:
        """Total trials that hit any bug (before dedup)."""
        return sum(self.bug_counts.values())

    def summary(self, *, skips: bool = False) -> str:
        lines = [f"gfuzz report — seed={self.seed} trials={self.n_trials}"]
        for m in self.counts:
            c = self.counts[m]
            lines.append(f"  {m:10s}  ok={c.get('ok',0):5d}  skip={c.get('skip',0):5d}  BUG={c.get('bug',0):4d}")
        lines.append(f"  {'DISTINCT BUGS':13s}  {self.n_bugs}   (total hits: {self.n_bug_hits})")
        for f in self.failures:
            hits = self.bug_counts.get(_sig(f.method, f.detail), 1)
            lines.append("")
            lines.append(f"  ✗ {f.method} — {hits} hit(s); first at trial#{f.trial} "
                         f"(reproduce: --seed {f.seed} --only {f.trial})")
            lines.append(f"    {f.detail}")
            lines.append(f"    {'shrunk' if f.shrunk_repr else 'term'}: {f.shrunk_repr or f.term_repr}")
        if skips:
            lines.append("")
            lines.append("  --- skip breakdown (leading tag of each skip) ---")
            for m, ctr in self.skip_reasons.items():
                if not ctr:
                    continue
                lines.append(f"  {m}:")
                for tag, k in ctr.most_common():
                    lines.append(f"      {k:5d}  {tag}")
                    for ex in self.skip_examples.get((m, tag), [])[:2]:
                        lines.append(f"             e.g. {ex}")
        return "\n".join(lines)


def _sig(method: str, detail: str) -> str:
    """A dedup signature: method + the detail with numbers/addresses/quoted names stripped, so
    the same defect from different random draws collapses to one entry."""
    d = re.sub(r"0x[0-9a-fA-F]+", "", detail)
    d = re.sub(r"[-+]?\d[\d.eE+-]*", "#", d)
    d = re.sub(r"'[^']*'", "'…'", d)
    return f"{method}|{d[:120]}"


def _describe(term) -> str:
    try:
        import grassmann as G

        return G.unparse(term)
    except Exception:
        return repr(term)[:200]


def _trial_rng(seed: int, i: int) -> np.random.Generator:
    return np.random.default_rng([seed, i])


def _safe(method_name, fn, *args):
    """Run an oracle, converting any unexpected exception into a synthetic crash bug."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 — a crash IS the finding
        tb = traceback.format_exc().splitlines()[-3:]
        return O.Result(method_name, "bug", f"CRASH {type(e).__name__}: {e} | " + " ⏎ ".join(tb))


# A shrink predicate must preserve the *same* defect, not merely "some bug" — otherwise
# shrinking can drift to a different crash and print a misleading reproducer.  Each predicate
# re-runs its oracle and compares the failure signature to the original.

def _still_fails_ad(case, seed, i, sig):
    r = np.random.default_rng([seed, i, 7])
    res = _safe("ad_vs_fd", O.ad_vs_fd, case, r)
    return res.status == "bug" and _sig("ad_vs_fd", res.detail) == sig


def _still_fails_diff2(case, seed, i, sig):
    r = np.random.default_rng([seed, i, 17])
    res = _safe("diff2", O.diff_second, case, r)
    return res.status == "bug" and _sig("diff2", res.detail) == sig


def _still_fails_rewrite(case, seed, i, sig):
    r = np.random.default_rng([seed, i, 9])
    res = _safe("rewrite", O.rewrite_invariant, case, r)
    return res.status == "bug" and _sig("rewrite", res.detail) == sig


def _still_fails_typecheck(case, seed, i, sig):
    r = np.random.default_rng([seed, i, 11])
    res = _safe("typecheck", O.typecheck, case, r)
    return res.status == "bug" and _sig("typecheck", res.detail) == sig


def _still_fails_shape(case, seed, i, sig):
    r = np.random.default_rng([seed, i, 13])
    res = _safe("shape", O.shape_canonical, case, r)
    return res.status == "bug" and _sig("shape", res.detail) == sig


def _still_fails_hook(case, seed, i, sig):
    r = np.random.default_rng([seed, i, 15])
    res = _safe("hook", O.shape_hook, case, r)
    return res.status == "bug" and _sig("hook", res.detail) == sig


def run(seed: int = 0, n_trials: int = 500, *, methods=METHODS, max_depth: int = 4,
        dim: int = 3, only: int | None = None, do_shrink: bool = True,
        quiet_warnings: bool = True, keep_skip_examples: int = 3,
        shape_hook: bool = True, dynamic: bool = False) -> Report:
    if not shape_hook:
        methods = tuple(m for m in methods if m != "hook")  # drop the canonical-shape hook oracle
    rep = Report(seed=seed, n_trials=n_trials,
                 counts={m: {"ok": 0, "bug": 0, "skip": 0} for m in methods},
                 skip_reasons={m: Counter() for m in methods},
                 skip_examples={})
    indices = [only] if only is not None else range(n_trials)
    _run_trials(rep, indices, seed, dim, max_depth, methods, do_shrink,
                keep_skip_examples, quiet_warnings, dynamic)
    return rep


def _run_trials(rep, indices, seed, dim, max_depth, methods, do_shrink,
                keep_skip_examples, quiet_warnings, dynamic=False):
    with warnings.catch_warnings():
        if quiet_warnings:
            warnings.simplefilter("ignore")
        for i in indices:
            r = _trial_rng(seed, i)
            case = _safe_generate(r, dim, max_depth, dynamic)
            if case is None:
                continue
            if "ad_vs_fd" in methods:
                res = _safe("ad_vs_fd", O.ad_vs_fd, case, r)
                sig = _sig("ad_vs_fd", res.detail)
                _record(rep, "ad_vs_fd", i, seed, res, case, do_shrink,
                        lambda c, s=sig: _still_fails_ad(c, seed, i, s), keep_skip_examples)
            if "diff2" in methods:
                res = _safe("diff2", O.diff_second, case, r)
                sig = _sig("diff2", res.detail)
                _record(rep, "diff2", i, seed, res, case, do_shrink,
                        lambda c, s=sig: _still_fails_diff2(c, seed, i, s), keep_skip_examples)
            if "rewrite" in methods:
                res = _safe("rewrite", O.rewrite_invariant, case, r)
                sig = _sig("rewrite", res.detail)
                _record(rep, "rewrite", i, seed, res, case, do_shrink,
                        lambda c, s=sig: _still_fails_rewrite(c, seed, i, s), keep_skip_examples)
            if "typecheck" in methods:
                res = _safe("typecheck", O.typecheck, case, r)
                sig = _sig("typecheck", res.detail)
                _record(rep, "typecheck", i, seed, res, case, do_shrink,
                        lambda c, s=sig: _still_fails_typecheck(c, seed, i, s), keep_skip_examples)
            if "shape" in methods:
                res = _safe("shape", O.shape_canonical, case, r)
                sig = _sig("shape", res.detail)
                _record(rep, "shape", i, seed, res, case, do_shrink,
                        lambda c, s=sig: _still_fails_shape(c, seed, i, s), keep_skip_examples)
            if "hook" in methods:
                res = _safe("hook", O.shape_hook, case, r)
                sig = _sig("hook", res.detail)
                _record(rep, "hook", i, seed, res, case, do_shrink,
                        lambda c, s=sig: _still_fails_hook(c, seed, i, s), keep_skip_examples)
            if "identity" in methods:
                _record(rep, "identity", i, seed,
                        _safe("identity", lambda: O.identity_holds(r, n=dim)),
                        None, False, None, keep_skip_examples)
    return rep


def _safe_generate(rng, dim, max_depth, dynamic=False):
    try:
        return generate(rng, n=dim, max_depth=max_depth, dynamic=dynamic)
    except Exception:
        return None


def _record(rep, method, i, seed, res, case, do_shrink, predicate, keep_ex):
    rep.counts[method][res.status] += 1
    if res.status == "skip":
        tag = res.detail.split(":")[0].split(" ")[0] if res.detail else "?"
        rep.skip_reasons[method][tag] += 1
        if case is not None:
            key = (method, tag)
            exs = rep.skip_examples.setdefault(key, [])
            if len(exs) < keep_ex:
                exs.append(_describe(case.term))
        return
    if res.status != "bug":
        return
    sig = _sig(method, res.detail)
    first_time = sig not in rep.bug_counts
    rep.bug_counts[sig] += 1
    if not first_time:
        return  # a distinct-bug report already holds this signature
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
