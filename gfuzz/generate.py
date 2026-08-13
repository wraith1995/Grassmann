"""Typed, *space-aware* random generator of closed grassmann terms.

The generator draws a random **space** from the grassmann zoo — base Euclidean spaces of
several dimensions, ``Dual``, direct sums ``OPlus``, tensor products ``OProd``, replicated
``OPlusRep``, and linear-map spaces ``oprod(dual(A), B)`` (including *rectangular* maps
``A ⊸ B`` with ``A ≠ B``) — and then builds a well-typed value of that space by a recursive,
structure-directed production table.  Because every subterm is typed as it is built, no
rejection sampling on the type system is needed.

Feeds are not hand-shaped: a generated term is compiled once and its
:class:`polyarray.SymInput` shapes are read off, so a value of *any* composite space is fed
correctly (``OPlus`` → a flat vector, ``OProd`` → a matrix, a map → a ``(dom, cod)`` array).
Only the differentiation variable ``x`` and the matrices that must be invertible (``bij`` for
``solve``, SPD for the operator root) carry *controlled* values; everything else is random.

Domain-restricted scalar ops (``sqrt``/``log``/``div``/``reciprocal``) are emitted through
guarded arguments (``t²+1``) so values and their finite differences stay finite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import grassmann as G
import grassmann.stdlib as L
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.core.props import bij as _BIJ


def _node(e: object) -> T.Term:
    """stdlib scalar ops return an embed ``Expr``; unwrap to the core term."""
    return e.node if hasattr(e, "node") else e  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Space helpers.
# ---------------------------------------------------------------------------


def space_dim(sp: S.Space) -> int:
    """Total numeric dimension of a Euclidean space (product/sum over its structure)."""
    match sp:
        case S.Scalar():
            return 1
        case S.SVar():
            return 1
        case S.Dual(space=inner):
            return space_dim(inner)
        case S.OPlus(spaces=ss):
            return sum(space_dim(t) for t in ss)
        case S.OProd(spaces=ss):
            out = 1
            for t in ss:
                out *= space_dim(t)
            return out
        case S.OPlusRep(space=inner, count=k):
            return _dim_int(k) * space_dim(inner)
        case _:
            return space_dim(sp.space) if hasattr(sp, "space") else 1


def _dim_int(d: object) -> int:
    if isinstance(d, int):
        return d
    if isinstance(d, S.DimLit):
        return int(d.value)
    return int(getattr(d, "value", 1))


def as_map(sp: S.Space):
    """``(A, B)`` if ``sp`` is a linear-map space ``oprod(dual(A), B)`` = ``A ⊸ B``, else None."""
    if isinstance(sp, S.OProd) and len(sp.spaces) == 2 and isinstance(sp.spaces[0], S.Dual):
        return sp.spaces[0].space, sp.spaces[1]
    return None


# ---------------------------------------------------------------------------
# Case.
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """A generated closed term ``F`` with its differentiation variable and a complete feed.

    ``feed0`` maps every input name of ``term`` to a concrete value at the sample point
    (including ``x``).  ``bij_names`` / ``spd_names`` record the leaves whose values must be
    invertible; the runner keeps them controlled when it re-samples.
    """

    term: T.Term
    x: T.Var
    x0: np.ndarray
    feed0: dict[str, np.ndarray]
    x_name: str
    bij_names: set[str] = field(default_factory=set)
    spd_names: set[str] = field(default_factory=set)
    n: int = 3

    def feed(self, x0: np.ndarray | None = None) -> dict:
        if x0 is None:
            return dict(self.feed0)
        f = dict(self.feed0)
        f[self.x_name] = x0
        return f


class _Regen(Exception):
    """A production that cannot proceed; caught upstream to fall back to a leaf."""


# ---------------------------------------------------------------------------
# Generator.
# ---------------------------------------------------------------------------


class Generator:
    """Draws well-typed random terms of a requested space, recording leaf roles as it goes."""

    def __init__(self, rng: np.random.Generator, n: int = 3, max_depth: int = 4):
        self.rng = rng
        self.n = n
        self.max_depth = max_depth
        # a small palette of base Euclidean spaces (interned so equality holds)
        dims = sorted({2, 3, max(2, min(n, 4))})
        self.bases = [S.base(f"R{d}", d) for d in dims]
        self.Sc = S.Scalar()
        self._uid = 0
        self.bij_names: set[str] = set()
        self.spd_names: set[str] = set()
        # the differentiation variable lives in a base Euclidean space
        self.X = S.base("X", n)
        self.x = T.var("x", self.X)
        self.x0 = rng.standard_normal(n)

    # -- names / leaves ------------------------------------------------------

    def _fresh(self, base: str) -> str:
        self._uid += 1
        return f"{base}{self._uid}"

    def _leaf(self, sp: S.Space) -> T.Term:
        # bias so x actually appears where its space fits
        if sp == self.X and self.rng.random() < 0.5:
            return self.x
        return T.var(self._fresh("v"), sp)

    def _endo_space(self) -> S.Space:
        w = self._pick_base()
        return S.oprod(S.dual(w), w)

    def _pick_base(self) -> S.Space:
        # include X so x-dependence can thread through maps/inner
        pool = [*self.bases, self.X]
        return pool[self.rng.integers(len(pool))]

    def _bij_leaf(self, endo: S.Space) -> T.Term:
        v = T.var(self._fresh("M"), endo, props=_BIJ)
        self.bij_names.add(f"{v.name}_{v.uid}")
        return v

    def _spd_leaf(self, endo: S.Space) -> T.Term:
        v = T.var(self._fresh("S"), endo)
        self.spd_names.add(f"{v.name}_{v.uid}")
        return v

    # -- random space --------------------------------------------------------

    def random_space(self, depth: int, *, euclidean: bool = True, max_dim: int = 8) -> S.Space:
        if depth <= 0 or self.rng.random() < 0.35:
            return self.Sc if self.rng.random() < 0.25 else self._pick_base()
        choice = self.rng.choice(["dual", "oplus", "oprod", "rep", "map", "base"])
        try:
            if choice == "dual":
                return S.dual(self.random_space(depth - 1, max_dim=max_dim))
            if choice == "oplus":
                k = int(self.rng.integers(2, 4))
                parts = [self.random_space(depth - 1, max_dim=max_dim) for _ in range(k)]
                sp = S.oplus(*parts)
            elif choice == "oprod":
                parts = [self.random_space(depth - 1, max_dim=3) for _ in range(2)]
                sp = S.oprod(*parts)
            elif choice == "rep":
                k = int(self.rng.integers(2, 4))
                sp = S.oplus_rep(k, self._pick_base())
            elif choice == "map":
                a, b = self._pick_base(), self._pick_base()
                sp = S.oprod(S.dual(a), b)
            else:
                sp = self._pick_base()
            return sp if space_dim(sp) <= max_dim else self._pick_base()
        except Exception:
            return self._pick_base()

    # -- value of a given space ---------------------------------------------

    def gen(self, sp: S.Space, depth: int) -> T.Term:
        if depth <= 0:
            return self._leaf(sp)
        builders = self._applicable(sp)
        b = builders[self.rng.integers(len(builders))]
        try:
            return b(sp, depth - 1)
        except (_Regen, Exception):
            return self._leaf(sp)

    def _applicable(self, sp: S.Space):
        gen = [self._b_leaf, self._b_sum, self._b_scale, self._b_linapply]
        if _is_euclidean(sp):
            gen.append(self._b_elementwise)
        if isinstance(sp, S.Scalar):
            gen += [self._b_inner, self._b_mul, self._b_nonlin, self._b_guarded,
                    self._b_div, self._b_trace, self._b_det, self._b_logdet]
        if isinstance(sp, S.OPlus):
            gen.append(self._b_oplusv)
        if isinstance(sp, S.OProd):
            gen.append(self._b_oprodv)
        m = as_map(sp)
        if m is not None:
            gen.append(self._b_dagger)
            if m[0] == m[1]:
                gen.append(self._b_solve_map)
        return gen

    # generic
    def _b_leaf(self, sp, d):
        return self._leaf(sp)

    def _b_sum(self, sp, d):
        return T.sum_(self.gen(sp, d), self.gen(sp, d))

    def _b_scale(self, sp, d):
        return T.scale(float(self.rng.uniform(-2, 2)), self.gen(sp, d))

    def _b_linapply(self, sp, d):
        # A : T ⊸ sp  applied to a value of T  (rectangular maps included)
        t = self._pick_base()
        A = T.var(self._fresh("A"), S.oprod(S.dual(t), sp))
        return T.Eval(A, (0,), (self.gen(t, d),))

    def _b_elementwise(self, sp, d):
        op = self.rng.choice(["tanh", "sin", "cos", "square", "sigmoid"])
        return _node(getattr(L, op)(self.gen(sp, d)))

    # scalar
    def _b_inner(self, sp, d):
        w = self._pick_base()
        return T.Eval(T.inner(w), (0, 1), (self.gen(w, d), self.gen(w, d)))

    def _b_mul(self, sp, d):
        return T.Eval(T.inner(self.Sc), (0, 1), (self.gen(self.Sc, d), self.gen(self.Sc, d)))

    def _b_nonlin(self, sp, d):
        op = self.rng.choice(["tanh", "sin", "cos", "exp", "square", "sigmoid", "softplus"])
        return _node(getattr(L, op)(self.gen(self.Sc, d)))

    def _b_guarded(self, sp, d):
        op = self.rng.choice(["sqrt", "log", "reciprocal"])
        return _node(getattr(L, op)(self._positive(self.gen(self.Sc, d))))

    def _b_div(self, sp, d):
        return _node(L.div(self.gen(self.Sc, d), self._positive(self.gen(self.Sc, d))))

    def _b_trace(self, sp, d):
        return T.Trace(self._xdep_endo(d), 0, 1)

    def _b_det(self, sp, d):
        return T.Det(self._xdep_endo(d))

    def _b_logdet(self, sp, d):
        return T.LogDet(self._xdep_endo(d))

    # oplus / oprod
    def _b_oplusv(self, sp, d):
        return T.OPlusV(tuple(self.gen(part, d) for part in sp.spaces))

    def _b_oprodv(self, sp, d):
        return T.OProdV(tuple(self.gen(part, d) for part in sp.spaces))

    # maps
    def _b_dagger(self, sp, d):
        a, b = as_map(sp)
        return T.dagger(self.gen(S.oprod(S.dual(b), a), d))  # (B⊸A)† : A⊸B

    def _b_solve_map(self, sp, d):
        a, _ = as_map(sp)
        return T.solve(self._bij_leaf(S.oprod(S.dual(a), a)))

    # helpers
    def _positive(self, t):
        return T.sum_(_node(L.square(t)), T.Lit(1.0))

    def _xdep_endo(self, d):
        """A well-conditioned endomorphism plus a *small* x-dependent rank-1 bump, so det/trace
        depend on x yet stay well-behaved at the sample point."""
        w = self._pick_base()
        endo = S.oprod(S.dual(w), w)
        base = self._bij_leaf(endo)
        u = self.gen(w, max(d - 1, 0))
        v = self.gen(w, max(d - 1, 0))
        bump = T.scale(0.05, T.oprod_v(T.Eval(T.flat(w), (0,), (u,)), v))
        return T.sum_(base, bump)


def _is_euclidean(sp: S.Space) -> bool:
    match sp:
        case S.Scalar() | S.SVar():
            return True
        case S.Dual(space=inner) | S.OPlusRep(space=inner):
            return _is_euclidean(inner)
        case S.OPlus(spaces=ss) | S.OProd(spaces=ss):
            return all(_is_euclidean(t) for t in ss)
        case _:
            return False


# ---------------------------------------------------------------------------
# Feed construction (introspected shapes).
# ---------------------------------------------------------------------------


def _well_conditioned(n: int, rng) -> np.ndarray:
    return 1.5 * np.eye(n) + 0.3 * rng.standard_normal((n, n))


def _spd(n: int, rng) -> np.ndarray:
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _fill_feed(g: Generator, term: T.Term, rng: np.random.Generator) -> dict:
    """Read the compiled term's input shapes and draw a value for each — controlled for the
    diff var and the invertible/SPD leaves, random otherwise."""
    prog = G.compile(term)
    feed: dict[str, np.ndarray] = {}
    x_name = f"{g.x.name}_{g.x.uid}"
    for inp in prog.inputs:
        name, shape = inp.name, tuple(int(s) for s in inp.shape)
        if name == x_name:
            feed[name] = g.x0
        elif name in g.bij_names:
            # bij leaves are always square endomorphisms (shape (n, n))
            feed[name] = _well_conditioned(shape[0], rng)
        elif name in g.spd_names:
            feed[name] = _spd(shape[0], rng)
        else:
            feed[name] = rng.standard_normal(shape)
    return feed


def generate(rng: np.random.Generator, *, n: int = 3, max_depth: int = 4,
             out_space=None) -> Case:
    """Draw one random :class:`Case`.

    ``out_space`` fixes the output space (default: a random Euclidean space, biased toward
    Scalar/vector so the AD oracle has a value to finite-difference)."""
    g = Generator(rng, n=n, max_depth=max_depth)
    if out_space is None:
        r = rng.random()
        if r < 0.4:
            sp = g.Sc
        elif r < 0.7:
            sp = g._pick_base()
        else:
            sp = g.random_space(2, max_dim=8)
    else:
        sp = out_space
    term = g.gen(sp, max_depth)
    x_name = f"{g.x.name}_{g.x.uid}"
    feed0 = _fill_feed(g, term, rng)
    return Case(term=term, x=g.x, x0=g.x0, feed0=feed0, x_name=x_name,
                bij_names=set(g.bij_names), spd_names=set(g.spd_names), n=n)
