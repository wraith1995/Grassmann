"""Structural shrinking of a failing :class:`~gfuzz.generate.Case`.

When a random term trips an oracle, the raw term is usually far larger than the defect
needs.  :func:`shrink` repeatedly replaces a subterm with a *value-carrying leaf of the
same space* (a fresh const var seeded from the subterm's own value at the sample point, so
the shrunk term still lowers to finite numbers) and keeps any replacement under which the
oracle still reports ``bug``.  The result is a minimal term that still exhibits the failure
— the thing you actually paste into a regression test.

The oracle is passed as a predicate ``still_fails(case) -> bool`` so shrinking is agnostic
to which method found the bug.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T

from .generate import Case


def _children(term: T.Term):
    """The immediate Term children of ``term`` and a rebuild function ``(new_children)->term``.

    This is a deliberately *partial* dispatch over the Term sum: it only knows the node kinds
    the generator emits.  The default arm (an unknown node) returns *no* children — i.e. the
    node is treated as an unshrinkable leaf.  That bail is always correct (it never produces a
    wrong term, only a less-thorough shrink), so no exhaustive ``assert_never`` is owed here —
    this is a dev-tool minimizer, not library dispatch.
    """
    if isinstance(term, T.Eval):
        args = list(term.args)
        return args, lambda ch: T.Eval(term.f, term.positions, tuple(ch))
    if isinstance(term, T.Sum):
        vs = list(term.vectors)
        return vs, lambda ch: T.sum_(*ch)
    if isinstance(term, T.Scale):
        return [term.vector], lambda ch: T.scale(term.scalar, ch[0])
    if isinstance(term, T.External):
        args = list(term.args)
        return args, lambda ch: T.External(term.space, term.data, tuple(ch))
    if isinstance(term, (T.Trace,)):
        return [term.f], lambda ch: T.Trace(ch[0], term.i, term.j)
    if isinstance(term, T.Det):
        return [term.f], lambda ch: T.Det(ch[0])
    if isinstance(term, T.LogDet):
        return [term.f], lambda ch: T.LogDet(ch[0])
    if isinstance(term, T.SolveApply):
        return [term.f, term.v], lambda ch: T.SolveApply(ch[0], ch[1])
    if isinstance(term, T.PinvApply):
        return [term.f, term.v], lambda ch: T.PinvApply(ch[0], ch[1])
    if isinstance(term, T.Dagger):
        return [term.f], lambda ch: T.dagger(ch[0])
    if isinstance(term, T.OProdV):
        vs = list(term.vectors)
        return vs, lambda ch: T.oprod_v(*ch)
    return [], lambda ch: term


def _leafize(case: Case, sub: T.Term) -> T.Term | None:
    """A fresh const leaf of ``sub``'s space, valued at ``sub``'s value under the case feed."""
    try:
        val = _value(case, sub)          # the subterm's value at the sample point
        sp = _synth_space(sub)           # its space
    except Exception:
        return None
    v = T.var(f"z{id(sub) & 0xffff}", sp)
    case.consts[v] = val
    return v


def _value(case: Case, sub: T.Term) -> np.ndarray:
    return np.asarray(G.compile(sub).run(case.feed())["result"], float)


def _synth_space(sub: T.Term) -> S.Space:
    from grassmann.infer.synth import synth_space

    return synth_space(sub)


def _walk_positions(term: T.Term):
    """Yield ``(path, subterm)`` for every Term node, path = list of child indices."""
    yield [], term
    ch, _ = _children(term)
    for i, c in enumerate(ch):
        for p, s in _walk_positions(c):
            yield [i] + p, s


def _replace(term: T.Term, path: list[int], new: T.Term) -> T.Term:
    if not path:
        return new
    ch, rebuild = _children(term)
    i = path[0]
    ch = list(ch)
    ch[i] = _replace(ch[i], path[1:], new)
    return rebuild(ch)


def shrink(case: Case, still_fails: Callable[[Case], bool], *, rounds: int = 40) -> Case:
    """Greedily replace subterms with leaves while ``still_fails`` holds.  Returns the smallest
    case found (never larger than the input)."""
    best = case
    for _ in range(rounds):
        progressed = False
        # try shrinking deeper (larger) subterms first
        nodes = sorted(_walk_positions(best.term), key=lambda ps: -len(ps[0]))
        for path, sub in nodes:
            if not path:
                continue  # never leafize the whole term
            if isinstance(sub, T.Var) or isinstance(sub, T.Lit):
                continue
            trial = Case(term=best.term, x=best.x, x0=best.x0,
                         consts=dict(best.consts), n=best.n)
            leaf = _leafize(trial, sub)
            if leaf is None:
                continue
            trial.term = _replace(best.term, path, leaf)
            try:
                if still_fails(trial):
                    best = trial
                    progressed = True
                    break
            except Exception:
                continue
        if not progressed:
            break
    return best
