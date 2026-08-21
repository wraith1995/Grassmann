"""Coverage instrument — what fraction of the grassmann surface does the generator exercise?

Walks a batch of generated terms and tallies, against the *live* closed sum types:

* every ``Term`` node kind that appears,
* every ``Space`` node kind (over each variable's space, plus the spaces named inside
  ``Zero``/``Identity``/subspace/``WithInner`` nodes),
* every ``MapProps`` flag actually asserted on a variable,
* every stdlib ``External`` op name.

and reports the **MISSING** sets — the direct answer to "which nodes / props do we never
generate?".  Run: ``../.venv/bin/python -m gfuzz.coverage --trials 3000``.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np

import grassmann.core.props as P
import grassmann.core.space as S
import grassmann.core.term as T

from .generate import generate


def _term_kinds() -> set[str]:
    return {c.__name__ for c in T.Term.__args__}


def _space_kinds() -> set[str]:
    return {c.__name__ for c in S.Space.__args__}


def _prop_flags() -> list[str]:
    return [f.name for f in dataclasses.fields(P.MapProps)]


def walk_term(node, term_ctr: Counter, space_ctr: Counter, prop_ctr: Counter,
              ext_ctr: Counter) -> None:
    """Recurse a term, tallying node kinds, referenced spaces, props, and External ops."""
    if T.is_term(node):
        term_ctr[type(node).__name__] += 1
        if isinstance(node, T.Var) and node.props is not None:
            for flag in _prop_flags():
                if getattr(node.props, flag) not in (None, False):
                    prop_ctr[flag] += 1
        if isinstance(node, T.External):
            name = getattr(node.data, "name", None) or type(node.data).__name__
            ext_ctr[str(name)] += 1
    for f in dataclasses.fields(node):
        val = getattr(node, f.name)
        for child in _iter(val):
            if T.is_term(child):
                walk_term(child, term_ctr, space_ctr, prop_ctr, ext_ctr)
            elif S.is_space(child):
                walk_space(child, space_ctr)
                # subspace / WithInner nodes carry Terms (defining maps, metrics) — recurse them
                for gc in _iter_space_terms(child):
                    walk_term(gc, term_ctr, space_ctr, prop_ctr, ext_ctr)


def walk_space(sp, space_ctr: Counter) -> None:
    if not S.is_space(sp):
        return
    space_ctr[type(sp).__name__] += 1
    for f in dataclasses.fields(sp):
        for child in _iter(getattr(sp, f.name)):
            if S.is_space(child):
                walk_space(child, space_ctr)


def _iter(val):
    if isinstance(val, (tuple, list)):
        yield from val
    else:
        yield val


def _iter_space_terms(sp):
    """The Term-valued fields of a space (an Img/Ker defining map, a WithInner metric)."""
    for f in dataclasses.fields(sp):
        for child in _iter(getattr(sp, f.name)):
            if T.is_term(child):
                yield child


def measure(seed: int = 0, trials: int = 3000, dim: int = 3, max_depth: int = 4,
            dynamic: bool = False) -> dict:
    term_ctr: Counter = Counter()
    space_ctr: Counter = Counter()
    prop_ctr: Counter = Counter()
    ext_ctr: Counter = Counter()
    for i in range(trials):
        rng = np.random.default_rng([seed, i])
        try:
            case = generate(rng, n=dim, max_depth=max_depth, dynamic=dynamic)  # same path as runner
        except Exception:
            continue
        walk_space(case.x.space, space_ctr)  # the diff var's space
        walk_term(case.term, term_ctr, space_ctr, prop_ctr, ext_ctr)
    return dict(term=term_ctr, space=space_ctr, prop=prop_ctr, ext=ext_ctr)


def report(res: dict) -> str:
    lines = []
    for label, live in (("Term", _term_kinds()), ("Space", _space_kinds())):
        ctr = res["term" if label == "Term" else "space"]
        hit = set(ctr) & live
        missing = sorted(live - set(ctr))
        lines.append(f"== {label} nodes: {len(hit)}/{len(live)} covered ==")
        for name, k in ctr.most_common():
            if name in live:
                lines.append(f"   {k:8d}  {name}")
        if missing:
            lines.append(f"   MISSING: {', '.join(missing)}")
        lines.append("")
    flags = _prop_flags()
    hitp = set(res["prop"])
    lines.append(f"== MapProps flags: {len(hitp & set(flags))}/{len(flags)} covered ==")
    for f in flags:
        lines.append(f"   {res['prop'].get(f, 0):8d}  {f}")
    lines.append("")
    lines.append("== stdlib External ops ==")
    for name, k in res["ext"].most_common():
        lines.append(f"   {k:8d}  {name}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="gfuzz.coverage")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=3000)
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--dynamic", action="store_true", help="measure the symbolic-dim (DimVar) mode")
    a = p.parse_args(argv)
    print(report(measure(a.seed, a.trials, a.dim, a.max_depth, a.dynamic)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
