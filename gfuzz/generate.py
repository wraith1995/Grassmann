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
from grassmann.core import props as P
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.core.kinds import SpaceSort
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
    dims: dict[str, int] = field(default_factory=dict)  # DimVar name -> concrete size (dynamic mode)
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

    def __init__(self, rng: np.random.Generator, n: int = 3, max_depth: int = 4,
                 dynamic: bool = False):
        self.rng = rng
        self.n = n
        self.max_depth = max_depth
        self.dynamic = dynamic
        self.Sc = S.Scalar()
        self._uid = 0
        self.bij_names: set[str] = set()
        self.spd_names: set[str] = set()
        self.rank_specs: dict[str, tuple[int, int, int]] = {}  # name -> (dom, cod, rank)
        # `dims` maps each symbolic DimVar name to the CONCRETE size chosen for this case's feed
        # (empty in the static mode).  The compiled program is dimension-generic; the feed picks n.
        self.dims: dict[str, int] = {}
        if dynamic:
            # abstract ℝⁿ bases over distinct DimVars — the dimension-polymorphic (`DimVar`) lane.
            names = ["na", "nb"]
            self.dims = {nm: int(rng.integers(2, 5)) for nm in names}
            self.bases = [S.oplus_rep(S.DimVar(nm), self.Sc) for nm in names]
            self.X = self.bases[0]
            self.x = T.var("x", self.X)
            self.x0 = rng.standard_normal(self.dims["na"])
        else:
            dims = sorted({2, 3, max(2, min(n, 4))})
            self.bases = [S.base(f"R{d}", d) for d in dims]
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
        # "oplus" weighted up: direct sums were under-generated, and are where the OPlus
        # shape-realization bug lives.
        choice = self.rng.choice(["dual", "oplus", "oplus", "oprod", "rep", "map", "base",
                                  "subspace", "withinner"])
        try:
            if choice == "withinner":
                # a re-metricized base: WithInner(W, g) with g an SPD 2-covariant tensor (fed SPD)
                w = self._pick_base()
                g = T.var(self._fresh("g"), S.oprod(S.dual(w), S.dual(w)))
                self.spd_names.add(f"{g.name}_{g.uid}")
                return S.with_inner(w, g)
            if choice == "subspace":
                # a fundamental subspace of a fresh (honest-prop) defining map J : A ⊸ B
                a, b = self._pick_base(), self._pick_base()
                J = self._map_leaf(a, b, props=None)
                which = self.rng.choice(["img", "ker", "coimg", "coker"])
                sp = getattr(S, which)(J)
                return sp if 1 <= space_dim(sp) <= max_dim else self._pick_base()
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
        gen = [self._b_leaf, self._b_sum, self._b_scale, self._b_linapply, self._b_zero,
               self._b_neval, self._b_tuple_project]
        if _is_euclidean(sp):
            gen.append(self._b_elementwise)
        if isinstance(sp, S.Scalar):
            gen += [self._b_inner, self._b_mul, self._b_nonlin, self._b_guarded,
                    self._b_div, self._b_trace, self._b_det, self._b_logdet]
        if self._is_base(sp):
            # base Euclidean vector target: the endo/musical/subspace machinery lives here
            gen += [self._b_identity_apply, self._b_sharp, self._b_solveapply,
                    self._b_pinvapply, self._b_sqrt_apply, self._b_pinv_apply,
                    self._b_project_embed, self._b_assert, self._b_corestrict]
            if not self.dynamic:
                gen.append(self._b_lowrank_apply)  # rank/dim props need a concrete rank
        if isinstance(sp, S.WithInner):
            gen.append(self._b_remetric)
        if isinstance(sp, S.OPlus):
            gen += [self._b_oplusv, self._b_tuple_embed, self._b_tuple_permute]
        if isinstance(sp, S.OProd):
            gen += [self._b_oprodv, self._b_permute, self._b_invlift]
            if len(sp.spaces) == 2:
                gen.append(self._b_kron)
        if isinstance(sp, S.Dual) and isinstance(sp.space, S.OProd):
            gen.append(self._b_lift)
        m = as_map(sp)
        if m is not None:
            gen += [self._b_dagger, self._b_compose, self._b_llam]
            if m[0] == m[1]:
                gen.append(self._b_solve_map)
        return gen

    def _is_base(self, sp: S.Space) -> bool:
        return sp == self.X or sp in self.bases

    # generic
    def _b_leaf(self, sp, d):
        return self._leaf(sp)

    def _b_sum(self, sp, d):
        return T.sum_(self.gen(sp, d), self.gen(sp, d))

    def _b_scale(self, sp, d):
        return T.scale(float(self.rng.uniform(-2, 2)), self.gen(sp, d))

    def _b_linapply(self, sp, d):
        # A : T ⊸ sp  applied to a value of T  (rectangular maps included, with honest props)
        t = self._pick_base()
        A = self._map_leaf(t, sp)
        return T.Eval(A, (0,), (self.gen(t, d),))

    def _b_zero(self, sp, d):
        return T.zero(sp)

    def _b_neval(self, sp, d):
        """``(λ a:T. body(a)) applied to an arg`` — a nonlinear binder (``Lam`` + ``NEval``).
        The body genuinely uses ``a`` so β-reduction (checked by the rewrite oracle) is exercised."""
        t = self._pick_base()
        a = T.var(self._fresh("a"), t)
        if isinstance(sp, S.Scalar):
            body = T.Eval(T.inner(t), (0, 1), (a, self.gen(t, d)))  # ⟨a, w⟩
        else:
            body = T.Eval(self._map_leaf(t, sp), (0,), (a,))        # A·a : sp
            if _is_euclidean(sp):
                body = _node(L.tanh(body))
        lam = T.lam(((a.name, SpaceSort(t)),), body, (a,))
        return T.neval(lam, (0,), self.gen(t, d))

    def _b_tuple_project(self, sp, d):
        """Extract summand ``sp`` from an ``OPlus`` that contains it — the ``TupleProject`` map."""
        other = self._pick_base()
        i = int(self.rng.integers(2))
        parts = (sp, other) if i == 0 else (other, sp)
        osp = S.oplus(*parts)
        return T.Eval(T.TupleProject(osp, i), (0,), (self.gen(osp, d),))

    def _b_tuple_permute(self, sp, d):
        """Reverse the summands of an ``OPlus`` value (the ``TuplePermute`` node)."""
        parts = list(sp.spaces)
        k = len(parts)
        perm = tuple(range(k - 1, -1, -1))
        v = self.gen(S.oplus(*[parts[i] for i in perm]), d)
        return T.TuplePermute(v, perm)

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

    def _b_llam(self, sp, d):
        """``λ⊸ x:A. (B·x)`` — the LINEAR binder (a map ``A ⊸ B`` realized as its matrix).  The
        body uses ``x`` (so the LLam is well-formed), the same node ``diff`` itself emits."""
        a_sp, b_sp = as_map(sp)
        x = T.var(self._fresh("x"), a_sp)
        body = T.Eval(self._map_leaf(a_sp, b_sp), (0,), (x,))  # B·x, linear in x
        return T.LLam((x,), body)

    # -- new: base-target machinery (musical / solve / pinv / sqrt / identity / metric) ----

    def _map_leaf(self, dom, cod, *, props=None):
        """A fresh map var ``dom ⊸ cod`` with *honest* props: a random (introspected) value is
        generically full rank, so ``bij`` (square), ``inj`` (dom≤cod), ``surj`` (dom≥cod) all
        hold almost surely.  This exercises the props dimension without asserting a property the
        value does not have."""
        from grassmann.core.props import inj as _INJ, surj as _SURJ
        if props is None and self.dynamic:
            # symbolic dims have no comparable size — only a square (same-DimVar) map is honestly bij
            props = _BIJ if dom == cod and self.rng.random() < 0.6 else None
        elif props is None and self.rng.random() < 0.6:
            da, db = space_dim(dom), space_dim(cod)
            props = _BIJ if da == db else (_INJ if da < db else _SURJ)
        v = T.var(self._fresh("A"), S.oprod(S.dual(dom), cod), props=props)
        if props is _BIJ:
            self.bij_names.add(f"{v.name}_{v.uid}")
        return v

    def _rank_leaf(self, dom, cod):
        """A map ``dom ⊸ cod`` fed as an *exactly* rank-r product (r < min dim), asserting one of
        the rank-family props HONESTLY — so `rank`/`dim_img`/`dim_ker` inference is exercised."""
        da, db = space_dim(dom), space_dim(cod)
        r = int(self.rng.integers(1, min(da, db))) if min(da, db) > 1 else 1
        flavor = self.rng.choice(["rank", "dim_img", "dim_ker"])
        if flavor == "rank":
            props = P.rank(r)
        elif flavor == "dim_img":
            props = P.dim_img(r)
        else:
            props = P.dim_ker(da - r)
        v = T.var(self._fresh("K"), S.oprod(S.dual(dom), cod), props=props)
        self.rank_specs[f"{v.name}_{v.uid}"] = (da, db, r)
        return v

    def _b_lowrank_apply(self, sp, d):
        t = self._pick_base()
        return T.Eval(self._rank_leaf(t, sp), (0,), (self.gen(t, d),))

    def _b_assert(self, sp, d):
        # Assert(v, sp): re-assert a value already of space sp (coercion node, identity here)
        return T.assert_(self.gen(sp, d), sp)

    def _b_identity_apply(self, sp, d):
        return T.Eval(T.identity(sp), (0,), (self.gen(sp, d),))

    def _b_sharp(self, sp, d):
        # ♯ : sp* ⊸ sp, applied to a covector
        return T.Eval(T.sharp(sp), (0,), (self.gen(S.dual(sp), d),))

    def _b_solveapply(self, sp, d):
        return T.SolveApply(self._bij_leaf(S.oprod(S.dual(sp), sp)), self.gen(sp, d))

    def _b_pinvapply(self, sp, d):
        t = self._pick_base()
        return T.PinvApply(self._map_leaf(t, sp, props=None), self.gen(t, d))

    def _b_sqrt_apply(self, sp, d):
        return T.Eval(T.sqrt(self._spd_leaf(S.oprod(S.dual(sp), sp))), (0,), (self.gen(sp, d),))

    def _b_pinv_apply(self, sp, d):
        t = self._pick_base()
        return T.Eval(T.pinv(self._map_leaf(t, sp)), (0,), (self.gen(t, d),))

    def _b_remetric(self, sp, d):
        # sp is WithInner(inner, g): Remetric(sec, g) re-metricizes a section of `inner` into it,
        # reusing the SAME metric term the space carries (honest typing).
        inner, g = sp.space, sp.metric
        return T.Remetric(self.gen(inner, d), g)

    def _b_project_embed(self, sp, d):
        # project a value onto Img(J), then embed back to the ambient sp: a subspace round-trip
        t = self._pick_base()
        J = self._map_leaf(t, sp, props=None)
        z = self.gen(sp, d)
        return T.embed(T.project(z, S.img(J)), sp)

    def _b_permute(self, sp, d):
        # sp is OProd(parts): build a value of the reversed arrangement and permute it back
        parts = list(sp.spaces)
        k = len(parts)
        perm = tuple(range(k - 1, -1, -1))
        v = self.gen(S.oprod(*[parts[i] for i in perm]), d)
        return T.permute(v, perm)

    def _b_lift(self, sp, d):
        # sp = Dual(OProd(parts)): Lift fuses the OProd factors into one functional (curry).
        inner = sp.space
        v = self.gen(inner, d)
        return T.lift(v, tuple(range(len(inner.spaces))))

    def _b_invlift(self, sp, d):
        # InvLift(v, (0,), None) reshapes factor 0 back — space-preserving here (uncurry);
        # the oracles then check it also preserves the VALUE.
        return T.inv_lift(self.gen(sp, d), (0,), None)

    def _b_kron(self, sp, d):
        # sp = OProd(A, B): (MA ⊗ MB) applied to a tensor value in dom_A ⊗ dom_B
        a, b = sp.spaces
        ta, tb = self._pick_base(), self._pick_base()
        ma = self._map_leaf(ta, a)
        mb = self._map_leaf(tb, b)
        arg = T.OProdV((self.gen(ta, d), self.gen(tb, d)))
        return T.Eval(T.kron(ma, mb), (0,), (arg,))

    def _b_tuple_embed(self, sp, d):
        # embed a value of the i-th summand into the OPlus space sp
        i = int(self.rng.integers(len(sp.spaces)))
        return T.Eval(T.tuple_embed(sp, i), (0,), (self.gen(sp.spaces[i], d),))

    def _b_corestrict(self, sp, d):
        # Corestrict(J~) : t ⊸ Img(J) for an INJECTIVE J: t ⊸ sp (t smaller), embedded back to sp
        smaller = [s for s in (*self.bases, self.X) if space_dim(s) < space_dim(sp)]
        if not smaller:
            raise _Regen
        t = smaller[self.rng.integers(len(smaller))]
        from grassmann.core.props import inj as _INJ
        J = T.var(self._fresh("A"), S.oprod(S.dual(t), sp), props=_INJ)
        return T.embed(T.Eval(T.corestrict(J), (0,), (self.gen(t, d),)), sp)

    def _b_compose(self, sp, d):
        # sp is a map A ⊸ B: build (M2 ∘ M1) with M1: A⊸C, M2: C⊸B for a random middle C
        a, b = as_map(sp)
        c = self._pick_base()
        m1 = self._map_leaf(a, c)
        m2 = self._map_leaf(c, b)
        return T.compose(m2, (0,), m1)

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

    def _concrete(axis: object) -> int:
        # a DimAtom (symbolic ℝⁿ axis) resolves to this case's chosen concrete size; ints pass through
        if isinstance(axis, int):
            return axis
        nm = getattr(axis, "name", None)
        if nm in g.dims:
            return g.dims[nm]
        return int(axis)  # a concrete DimLit-like axis

    for inp in prog.inputs:
        name = inp.name
        shape = tuple(_concrete(s) for s in inp.shape)
        if name == x_name:
            feed[name] = g.x0
        elif name in g.bij_names:
            # bij leaves are always square endomorphisms (shape (n, n))
            feed[name] = _well_conditioned(shape[0], rng)
        elif name in g.spd_names:
            feed[name] = _spd(shape[0], rng)
        elif name in g.rank_specs:
            da, db, r = g.rank_specs[name]
            feed[name] = (rng.standard_normal((da, r)) @ rng.standard_normal((r, db))).reshape(shape)
        else:
            feed[name] = rng.standard_normal(shape)
    return feed


def generate(rng: np.random.Generator, *, n: int = 3, max_depth: int = 4,
             out_space=None, dynamic: bool = False) -> Case:
    """Draw one random :class:`Case`.

    ``out_space`` fixes the output space (default: a random Euclidean space, biased toward
    Scalar/vector so the AD oracle has a value to finite-difference).  ``dynamic=True`` builds
    over symbolic ``ℝⁿ`` bases (``DimVar``): the compiled program is dimension-generic and the
    feed picks a concrete size per ``DimVar`` — exercising grassmann's n-generic lowering."""
    g = Generator(rng, n=n, max_depth=max_depth, dynamic=dynamic)
    if out_space is None:
        r = rng.random()
        if r < 0.35:
            sp = g.Sc
        elif r < 0.6:
            sp = g._pick_base()
        elif r < 0.72:
            # a linear-map output — exercises the map-valued nodes (LLam / Dagger / Compose / Solve)
            sp = S.oprod(S.dual(g._pick_base()), g._pick_base())
        else:
            sp = g.random_space(2, max_dim=8)
    else:
        sp = out_space
    term = g.gen(sp, max_depth)
    x_name = f"{g.x.name}_{g.x.uid}"
    feed0 = _fill_feed(g, term, rng)
    return Case(term=term, x=g.x, x0=g.x0, feed0=feed0, x_name=x_name,
                bij_names=set(g.bij_names), spd_names=set(g.spd_names),
                dims=dict(g.dims), n=n)
