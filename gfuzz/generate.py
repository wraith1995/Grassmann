"""Typed random generator of closed grassmann terms over a single value variable ``x``.

The generator threads a *space kind* (``VEC`` / ``SCALAR`` / ``MAT``) through a recursive
production table, so every subterm it emits is well typed by construction — no rejection
sampling on the type system.  Every leaf ``Var`` it mints is recorded together with a
concrete numeric value (well-conditioned matrices for the invertible ops), so a generated
:class:`Case` carries its own reproducible feed.

The distinguished variable ``x : V`` is the differentiation variable for the AD oracle; a
term may or may not actually depend on it (a term independent of ``x`` is a valid case —
its derivative is the zero map, which the oracle checks against a ~0 finite difference).

Domain-restricted scalar ops (``sqrt``/``log``/``reciprocal``/``div``) are emitted through
*guarded* arguments (``sqrt(t²+1)`` etc.) so the value and its finite differences stay
finite without rejecting the op — the point is to exercise the lowering, not to trip on a
branch cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import grassmann.stdlib as L
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.core.props import bij as _BIJ

# --- the fixed space palette ------------------------------------------------

VEC = "vec"
SCALAR = "scalar"
MAT = "mat"
KINDS = (VEC, SCALAR, MAT)


def _node(e: object) -> T.Term:
    """stdlib scalar ops return an embed ``Expr``; unwrap to the core term."""
    return e.node if hasattr(e, "node") else e  # type: ignore[return-value]


@dataclass
class Case:
    """A generated closed term ``F`` with its differentiation variable and a concrete feed."""

    term: T.Term
    x: T.Var
    x0: np.ndarray
    consts: dict[T.Var, np.ndarray] = field(default_factory=dict)
    n: int = 3

    def feed(self, x0: np.ndarray | None = None) -> dict:
        f = {f"{v.name}_{v.uid}": val for v, val in self.consts.items()}
        f[f"{self.x.name}_{self.x.uid}"] = self.x0 if x0 is None else x0
        return f


class Generator:
    """Draws well-typed random terms of a requested kind, recording leaf values as it goes."""

    def __init__(self, rng: np.random.Generator, n: int = 3, max_depth: int = 4):
        self.rng = rng
        self.n = n
        self.V = S.base("V", n)
        self.Sc = S.Scalar()
        self.Mat = S.oprod(S.dual(self.V), self.V)
        self.max_depth = max_depth
        self._uid = 0
        self.consts: dict[T.Var, np.ndarray] = {}
        self.x = T.var("x", self.V)
        self.x0 = rng.standard_normal(n)

    # -- leaves --------------------------------------------------------------

    def _fresh(self, base: str) -> str:
        self._uid += 1
        return f"{base}{self._uid}"

    def _const_vec(self) -> T.Term:
        v = T.var(self._fresh("c"), self.V)
        self.consts[v] = self.rng.standard_normal(self.n)
        return v

    def _const_scalar(self) -> T.Term:
        v = T.var(self._fresh("s"), self.Sc)
        self.consts[v] = np.asarray(self.rng.standard_normal())  # shape () scalar
        return v

    def _well_conditioned(self) -> np.ndarray:
        """An invertible, positive-determinant matrix (safe for solve/det/logdet/pinv)."""
        return 1.5 * np.eye(self.n) + 0.3 * self.rng.standard_normal((self.n, self.n))

    def _spd(self) -> np.ndarray:
        """A symmetric-positive-definite matrix (safe for the operator square root)."""
        a = self.rng.standard_normal((self.n, self.n))
        return a @ a.T + self.n * np.eye(self.n)

    def _const_mat(self, kind: str = "generic") -> T.Term:
        # `bij` matrices are invertible by construction (well-conditioned draw), so asserting
        # it is honest — and Solve requires the assertion.
        props = _BIJ if kind in ("generic", "bij", "spd") else None
        v = T.var(self._fresh("M"), self.Mat, props=props)
        self.consts[v] = self._spd() if kind == "spd" else self._well_conditioned()
        return v

    # -- guarded scalars (kept in a safe domain for FD) ----------------------

    def _positive(self, t: T.Term) -> T.Term:
        """``t² + 1`` — a strictly positive scalar for sqrt/log domains."""
        sq = _node(L.square(t))
        return T.sum_(sq, T.Lit(1.0))

    def _nonzero(self, t: T.Term) -> T.Term:
        """``t² + 1`` — a scalar bounded away from 0 for reciprocal/div denominators."""
        return self._positive(t)

    # -- productions ---------------------------------------------------------

    def gen(self, kind: str, depth: int) -> T.Term:
        if depth <= 0:
            return self._leaf(kind)
        prods = self._PRODUCTIONS[kind]
        # Bias toward leaves as depth shrinks; always allow the leaf fallback.
        builder = self.rng.choice(len(prods))
        try:
            return prods[builder](self, depth - 1)
        except _Regen:
            return self._leaf(kind)

    def _leaf(self, kind: str) -> T.Term:
        if kind == VEC:
            # bias so x actually appears
            return self.x if self.rng.random() < 0.5 else self._const_vec()
        if kind == SCALAR:
            return self._const_scalar()
        return self._const_mat()

    # VEC productions
    def _vec_matapply(self, d: int) -> T.Term:
        return T.Eval(self.gen(MAT, d), (0,), (self.gen(VEC, d),))

    def _vec_sum(self, d: int) -> T.Term:
        return T.sum_(self.gen(VEC, d), self.gen(VEC, d))

    def _vec_scale(self, d: int) -> T.Term:
        return T.scale(float(self.rng.uniform(-2, 2)), self.gen(VEC, d))

    def _vec_elementwise(self, d: int) -> T.Term:
        op = self.rng.choice(["tanh", "sin", "cos", "square", "sigmoid"])
        return _node(getattr(L, op)(self.gen(VEC, d)))

    def _vec_solve(self, d: int) -> T.Term:
        # solve M·y = v with a well-conditioned const M ⇒ y is x-dependent through v
        return T.SolveApply(self._const_mat(), self.gen(VEC, d))

    def _vec_pinvapply(self, d: int) -> T.Term:
        return T.PinvApply(self._const_mat(), self.gen(VEC, d))

    def _vec_daggerapply(self, d: int) -> T.Term:
        return T.Eval(T.dagger(self.gen(MAT, d)), (0,), (self.gen(VEC, d),))

    # SCALAR productions
    def _sca_inner(self, d: int) -> T.Term:
        return T.Eval(T.inner(self.V), (0, 1), (self.gen(VEC, d), self.gen(VEC, d)))

    def _sca_mul(self, d: int) -> T.Term:
        return T.Eval(T.inner(self.Sc), (0, 1), (self.gen(SCALAR, d), self.gen(SCALAR, d)))

    def _sca_sum(self, d: int) -> T.Term:
        return T.sum_(self.gen(SCALAR, d), self.gen(SCALAR, d))

    def _sca_nonlin(self, d: int) -> T.Term:
        op = self.rng.choice(["tanh", "sin", "cos", "exp", "square", "sigmoid", "softplus"])
        return _node(getattr(L, op)(self.gen(SCALAR, d)))

    def _sca_guarded(self, d: int) -> T.Term:
        op = self.rng.choice(["sqrt", "log", "reciprocal"])
        return _node(getattr(L, op)(self._positive(self.gen(SCALAR, d))))

    def _sca_div(self, d: int) -> T.Term:
        return _node(L.div(self.gen(SCALAR, d), self._nonzero(self.gen(SCALAR, d))))

    def _sca_trace(self, d: int) -> T.Term:
        return T.Trace(self.gen(MAT, d), 0, 1)

    def _sca_det(self, d: int) -> T.Term:
        # determinant of a *const* well-conditioned matrix (x-independent ⇒ derivative 0),
        # OR of an x-perturbed one built to stay invertible at x0
        return T.Det(self._xdep_invertible_mat(d))

    def _sca_logdet(self, d: int) -> T.Term:
        return T.LogDet(self._xdep_invertible_mat(d))

    # MAT productions
    def _mat_dagger(self, d: int) -> T.Term:
        return T.dagger(self.gen(MAT, d))

    def _mat_sum(self, d: int) -> T.Term:
        return T.sum_(self.gen(MAT, d), self.gen(MAT, d))

    def _mat_rank1(self, d: int) -> T.Term:
        """``u♭ ⊗ v`` — an x-dependent rank-1 matrix (covector ⊗ vector)."""
        u = self.gen(VEC, d)
        v = self.gen(VEC, d)
        return T.oprod_v(G_flat(u, self.V), v)

    def _xdep_invertible_mat(self, d: int) -> T.Term:
        """A well-conditioned const matrix plus a *small* x-dependent rank-1 bump, so det/logdet
        depend on x yet stay invertible and positive at the sample point."""
        base = self._const_mat()
        u = self.gen(VEC, max(d - 1, 0))
        v = self.gen(VEC, max(d - 1, 0))
        bump = T.scale(0.05, T.oprod_v(G_flat(u, self.V), v))
        return T.sum_(base, bump)

    _PRODUCTIONS = {
        VEC: [_vec_matapply, _vec_sum, _vec_scale, _vec_elementwise,
              _vec_solve, _vec_pinvapply, _vec_daggerapply],
        SCALAR: [_sca_inner, _sca_mul, _sca_sum, _sca_nonlin, _sca_guarded,
                 _sca_div, _sca_trace, _sca_det, _sca_logdet],
        MAT: [_mat_dagger, _mat_sum, _mat_rank1],
    }


class _Regen(Exception):
    """Raised by a production that cannot proceed; caught upstream to fall back to a leaf."""


def G_flat(v: T.Term, V: S.Space) -> T.Term:
    """``v♭`` — lower a vector to a covector in ``dual(V)`` via the musical flat map."""
    return T.Eval(T.flat(V), (0,), (v,))


def generate(rng: np.random.Generator, *, n: int = 3, max_depth: int = 4,
             out_kind: str | None = None) -> Case:
    """Draw one random :class:`Case`.  ``out_kind`` fixes the output kind (default: SCALAR or VEC)."""
    g = Generator(rng, n=n, max_depth=max_depth)
    kind = out_kind or rng.choice([SCALAR, VEC])
    term = g.gen(kind, max_depth)
    return Case(term=term, x=g.x, x0=g.x0, consts=dict(g.consts), n=n)
