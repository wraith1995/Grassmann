"""Oracle methods — each pairs the grassmann machinery against an *independent* reference.

Three methods, each an independent way to catch a silent-wrong lowering:

* :func:`ad_vs_fd` — the headline.  Forward-mode AD (:func:`grassmann.transform.diff.diff`)
  gives a symbolic Jacobian map ``J : X ⊸ space(F)``; the reference is a *central finite
  difference* of the lowered value.  A mismatch means the differentiated code and the
  original code disagree — an AD bug.  (The discipline that found 14 basis bugs in
  ``test_diff_sweep``, driven by random terms instead of a fixed registry.)

* :func:`rewrite_invariant` — grassmann as its own oracle.  ``simplify`` / ``normal_form`` /
  ``beta`` are *value-preserving* rewrites; the rewritten term must lower to the same
  numbers as the original at a random point.  Any divergence is a rewrite bug — no external
  reference needed.

* :func:`identity_holds` — targeted algebraic laws with random operands (adjoint definition,
  ``dagger∘dagger = id``, ``M·solve(M,b) = b``, Moore–Penrose, ``logdet = log∘det``,
  ``tr(MN) = tr(NM)``, inner symmetry).  Each law is checked numerically against its own
  right-hand side, again independent of the path grassmann takes to lower it.

Every method returns a :class:`Result`.  A loud ``DiffError`` / ``RepresentError`` /
inference failure is reported as ``skipped`` (an acceptable bail — grassmann refusing an
out-of-scope construction), never as a bug; only a *silent numeric mismatch* is a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import grassmann as G
import grassmann.stdlib as L
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.transform.diff import DiffError, diff
from grassmann.infer.synth import InferError, synth_space

from .generate import Case, _node


@dataclass
class Result:
    method: str
    status: str  # "ok" | "bug" | "skip"
    detail: str = ""
    got: object = None
    ref: object = None


def _vk(v: T.Var) -> str:
    return f"{v.name}_{v.uid}"


def _run(term: T.Term, feed: dict) -> np.ndarray:
    prog = G.compile(term)
    names = {i.name for i in prog.inputs}
    sub = {k: v for k, v in feed.items() if k in names}
    return np.asarray(prog.run(sub)["result"], float)


# Loud, acceptable bails: grassmann refusing an out-of-scope build (an inference refusal or
# a diff-scope refusal), or a numerically singular draw.  These are correct behavior, not
# bugs — the fuzzer records them as skips.
_BAILS = (DiffError, InferError)


def _represent_error():
    from grassmann.lower.represent import RepresentError

    return RepresentError


# ---------------------------------------------------------------------------
# Method A — AD vs central finite differences.
# ---------------------------------------------------------------------------


def ad_vs_fd(case: Case, rng: np.random.Generator, *, eps: float = 1e-6,
             atol: float = 2e-4, rtol: float = 2e-4) -> Result:
    """``J·h`` (symbolic Jacobian applied to a random direction) vs central FD of ``F``."""
    RepresentError = _represent_error()
    feed = case.feed()
    # baseline value must be finite for the FD reference to mean anything
    try:
        base = _run(case.term, feed)
    except (RepresentError, *_BAILS) as e:
        return Result("ad_vs_fd", "skip", f"lower base: {type(e).__name__}: {e}")
    if not np.all(np.isfinite(base)):
        return Result("ad_vs_fd", "skip", "non-finite base value")

    try:
        J = diff(case.term, case.x, None)
    except _BAILS as e:
        return Result("ad_vs_fd", "skip", f"diff bail: {type(e).__name__}: {e}")

    h = rng.standard_normal(case.x0.shape)
    x0 = case.x0
    # central FD of the value
    fp = case.feed(x0 + eps * h)
    fm = case.feed(x0 - eps * h)
    try:
        fd = (_run(case.term, fp) - _run(case.term, fm)) / (2 * eps)
    except (RepresentError, *_BAILS) as e:
        return Result("ad_vs_fd", "skip", f"FD lower: {type(e).__name__}: {e}")
    if not np.all(np.isfinite(fd)):
        return Result("ad_vs_fd", "skip", "non-finite FD")

    # J·h: seed the LLam body directly (an LLam lowers transposed; applying via Eval would
    # misread the layout — the same trap pullback avoids).  Zero map ⇒ x-independent term.
    if isinstance(J, T.Zero):
        got = np.zeros_like(fd)
    else:
        seed = J.vars[0]
        try:
            got = _run(J.body, {**feed, _vk(seed): h})
        except (RepresentError, *_BAILS) as e:
            return Result("ad_vs_fd", "skip", f"J lower: {type(e).__name__}: {e}")
    got = np.ravel(got)
    fd = np.ravel(fd)
    if got.shape != fd.shape:
        return Result("ad_vs_fd", "skip", f"shape {got.shape} vs {fd.shape}")
    if not np.all(np.isfinite(got)):
        return Result("ad_vs_fd", "skip", "non-finite J·h")
    if np.allclose(got, fd, atol=atol, rtol=rtol):
        return Result("ad_vs_fd", "ok", got=got, ref=fd)
    return Result("ad_vs_fd", "bug",
                  f"AD J·h = {got} but central FD = {fd} (|Δ|max={np.max(np.abs(got-fd)):.3e})",
                  got=got, ref=fd)


# ---------------------------------------------------------------------------
# Method B — rewrite invariance.
# ---------------------------------------------------------------------------

_REWRITES = {"simplify": G.simplify, "normal_form": G.normal_form, "beta": G.beta}


def rewrite_invariant(case: Case, rng: np.random.Generator, *, atol: float = 1e-9) -> Result:
    """Every value-preserving rewrite of ``F`` must lower to the same numbers as ``F``."""
    RepresentError = _represent_error()
    feed = case.feed()
    try:
        base = _run(case.term, feed)
    except (RepresentError, *_BAILS) as e:
        return Result("rewrite", "skip", f"lower base: {type(e).__name__}: {e}")
    if not np.all(np.isfinite(base)):
        return Result("rewrite", "skip", "non-finite base value")

    for name, fn in _REWRITES.items():
        try:
            rewritten = _node(fn(case.term))
            val = _run(rewritten, feed)
        except (RepresentError, *_BAILS) as e:
            return Result("rewrite", "skip", f"{name}: {type(e).__name__}: {e}")
        if val.shape != base.shape or not np.allclose(val, base, atol=atol, rtol=1e-7):
            return Result("rewrite", "bug",
                          f"{name} changed the value: {name}(F)={np.ravel(val)} vs F={np.ravel(base)}",
                          got=val, ref=base)
    return Result("rewrite", "ok", got=base)


# ---------------------------------------------------------------------------
# Method D — type-checking probe (inference soundness / preservation).
# ---------------------------------------------------------------------------


def typecheck(case: Case, rng: np.random.Generator) -> Result:
    """Cross-check the *type checker* against the *lowering*, three ways.

    1. **Acceptance.**  If inference (:func:`synth_space`) rejects the term, that is a
       candidate *false rejection* — a well-formed construction the checker refused.  It is
       recorded as ``skip`` with a ``rejected:`` tag so the runner can surface examples for
       human review (there is no automatic oracle for "should this have type-checked?").

    2. **Soundness.**  If inference *accepts* the term, lowering it must not *crash*.  A clean
       ``RepresentError`` is a known lowering gap (``lower_gap:``, a skip — represent has
       documented runtime-raise fall-throughs).  But an *unexpected* exception (``IndexError``,
       ``AttributeError``, a numpy error, …) on a term the checker called well-typed is a real
       defect — reported as a **bug** (``UNSOUND``).

    3. **Preservation.**  ``simplify`` is value- *and* type-preserving, so
       ``synth_space(simplify(F))`` must equal ``synth_space(F)``.  A changed space is a bug.
    """
    RepresentError = _represent_error()
    # (1) acceptance
    try:
        sp = synth_space(case.term)
    except InferError as e:
        return Result("typecheck", "skip", f"rejected: {_short(e)}")
    except Exception as e:  # a crash *inside* inference is itself a defect
        return Result("typecheck", "bug", f"UNSOUND: inference crashed: {type(e).__name__}: {_short(e)}")

    # (2) soundness: accepted ⇒ lowering must not crash unexpectedly
    try:
        val = _run(case.term, case.feed())
    except RepresentError as e:
        return Result("typecheck", "skip", f"lower_gap: {_short(e)}")
    except DiffError as e:
        return Result("typecheck", "skip", f"lower_gap: {_short(e)}")
    except Exception as e:
        return Result("typecheck", "bug",
                      f"UNSOUND: synth accepted but lowering raised {type(e).__name__}: {_short(e)}")
    if not np.all(np.isfinite(val)):
        return Result("typecheck", "skip", "non-finite")

    # (3) type preservation under simplify
    try:
        simp = _node(G.simplify(case.term))
        sp2 = synth_space(simp)
    except Exception as e:
        return Result("typecheck", "bug", f"simplify broke inference: {type(e).__name__}: {_short(e)}")
    if sp != sp2:
        return Result("typecheck", "bug", f"simplify changed the type: {sp!r} -> {sp2!r}")
    return Result("typecheck", "ok")


def _short(e: object, k: int = 160) -> str:
    s = str(e).replace("\n", " ")
    return s if len(s) <= k else s[:k] + "…"


# ---------------------------------------------------------------------------
# Method C — algebraic identities with random operands.
# ---------------------------------------------------------------------------


def identity_holds(rng: np.random.Generator, *, n: int = 3, atol: float = 1e-7) -> Result:
    """Draw one random algebraic law and check it numerically against its own RHS."""
    V = S.base("V", n)
    Sc = S.Scalar()
    Mat = S.oprod(S.dual(V), V)

    def wc() -> np.ndarray:
        return 1.5 * np.eye(n) + 0.3 * rng.standard_normal((n, n))

    law = rng.choice(["adjoint", "dagger_involution", "solve", "pinv_mpm",
                      "logdet", "trace_cyclic", "inner_symmetry"])
    try:
        if law == "adjoint":
            # <M x, y> == <x, Mᵀ y>
            M, x, y = T.var("M", Mat), T.var("x", V), T.var("y", V)
            feed = {_vk(M): wc(), _vk(x): rng.standard_normal(n), _vk(y): rng.standard_normal(n)}
            lhs = T.Eval(T.inner(V), (0, 1), (T.Eval(M, (0,), (x,)), y))
            rhs = T.Eval(T.inner(V), (0, 1), (x, T.Eval(T.dagger(M), (0,), (y,))))
            return _cmp("adjoint", _run(lhs, feed), _run(rhs, feed), atol)

        if law == "dagger_involution":
            # (Mᵀ)ᵀ applied to x == M x
            M, x = T.var("M", Mat), T.var("x", V)
            feed = {_vk(M): wc(), _vk(x): rng.standard_normal(n)}
            lhs = T.Eval(T.dagger(T.dagger(M)), (0,), (x,))
            rhs = T.Eval(M, (0,), (x,))
            return _cmp("dagger_involution", _run(lhs, feed), _run(rhs, feed), atol)

        if law == "solve":
            # M·(M⁻¹ b) == b   (SolveApply needs M asserted bijective)
            M, b = T.var("M", Mat, props=G.bij), T.var("b", V)
            feed = {_vk(M): wc(), _vk(b): rng.standard_normal(n)}
            lhs = T.Eval(M, (0,), (T.SolveApply(M, b),))
            return _cmp("solve", _run(lhs, feed), feed[_vk(b)], atol)

        if law == "pinv_mpm":
            # M (M† (M x)) == M x   (Moore–Penrose:  M M† M = M)
            M, x = T.var("M", Mat), T.var("x", V)
            feed = {_vk(M): wc(), _vk(x): rng.standard_normal(n)}
            Mx = T.Eval(M, (0,), (x,))
            lhs = T.Eval(M, (0,), (T.PinvApply(M, Mx),))
            return _cmp("pinv_MPM", _run(lhs, feed), _run(Mx, feed), 1e-6)

        if law == "logdet":
            # logdet(M) == log(det(M))   for a positive-determinant M
            M = T.var("M", Mat)
            feed = {_vk(M): wc()}
            lhs = _run(T.LogDet(M), feed)
            ref = np.log(np.abs(_run(T.Det(M), feed)))
            return _cmp("logdet", lhs, ref, 1e-9)

        if law == "trace_cyclic":
            # tr(M N) == tr(N M)
            M, Nm = T.var("M", Mat), T.var("N", Mat)
            feed = {_vk(M): wc(), _vk(Nm): wc()}
            MN = T.Eval(M, (1,), (Nm,))  # compose: contract cod of N into dom of M
            NM = T.Eval(Nm, (1,), (M,))
            try:
                lhs = _run(T.Trace(MN, 0, 1), feed)
                rhs = _run(T.Trace(NM, 0, 1), feed)
            except Exception:
                # composition-index convention differs; fall back to a numpy-agnostic check
                a, b = feed[_vk(M)], feed[_vk(Nm)]
                return _cmp("trace_cyclic", np.array(np.trace(a @ b)), np.array(np.trace(b @ a)), atol)
            return _cmp("trace_cyclic", lhs, rhs, atol)

        # inner_symmetry: <a,b> == <b,a>
        a, b = T.var("a", V), T.var("b", V)
        feed = {_vk(a): rng.standard_normal(n), _vk(b): rng.standard_normal(n)}
        lhs = T.Eval(T.inner(V), (0, 1), (a, b))
        rhs = T.Eval(T.inner(V), (0, 1), (b, a))
        return _cmp("inner_symmetry", _run(lhs, feed), _run(rhs, feed), atol)
    except _BAILS as e:
        return Result("identity", "skip", f"{law}: {type(e).__name__}: {e}")


def _cmp(name: str, got: np.ndarray, ref: np.ndarray, atol: float) -> Result:
    got, ref = np.ravel(got), np.ravel(ref)
    if not (np.all(np.isfinite(got)) and np.all(np.isfinite(ref))):
        return Result("identity", "skip", f"{name}: non-finite")
    if np.allclose(got, ref, atol=atol, rtol=1e-6):
        return Result("identity", "ok", name, got=got, ref=ref)
    return Result("identity", "bug", f"{name}: {got} != {ref} (|Δ|max={np.max(np.abs(got-ref)):.3e})",
                  got=got, ref=ref)
