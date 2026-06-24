"""Finite-difference AD sweep — the permanent guard against SILENT-WRONG derivatives.

The AD analog of ``test_basis_honesty_sweep``.  For a registry of terms ``F(x)`` we assert the
*directional total derivative* the symbolic machinery produces equals a **central finite
difference**:

    diff(F, x, deps) =: J : X ⊸ space(F)          (the Jacobian map)
    J·h            ==  (F(x₀+εh) − F(x₀−εh)) / 2ε     for a random direction h

We compare via ``J·h`` (apply the Jacobian map to ``h`` through ``Eval``) rather than the raw
Jacobian tensor, so the check is **layout-robust** — the storage convention of the map never
enters.  The reference is a true finite difference of the lowered value (never a hand-derived
formula): this is the discipline that found 14 basis bugs, transplanted to ``diff``.

**Marked dependence.**  When ``F`` depends on ``x`` through *marked* value vars ``u`` (``deps =
{u.uid: Du}``, ``Du`` the supplied derivative map ``∂u/∂x``), the total derivative is
``Σₖ ∂F/∂uₖ · (Duₖ·h)``.  The finite difference must therefore perturb **x and every marked var
together**, each ``u`` along its own directional derivative ``Duₖ·h`` (computed via the
machinery, ``Eval(Du,(0,),h)`` — never by hand).  This is the key move that makes a plain FD a
correct reference for the *total* derivative.

**Axes** (each has caught a distinct basis bug for ``represent``; here they stress ``diff``):
terms × both ``GRASSMANN_LLAM_INDEX`` modes (vmap/eye, parametrized in-process) × the two
dependence regimes (direct-``x`` vs marked-dep) × operand-basis pinning (incl. subset pinning) ×
nesting × consumers × dims.  A loud ``DiffError``/``RepresentError`` is an ACCEPTABLE outcome;
only a *silent* numeric mismatch is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import grassmann as G
import grassmann.stdlib as L
from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.lower import represent
from grassmann.lower.represent import RepresentError, default_basis
from grassmann.transform.diff import DiffError, diff

_EPS = 1e-6
_ATOL = 2e-4
_SEED = 20260624


def _vk(v: T.Var) -> str:
    return f"{v.name}_{v.uid}"


def _run(term: T.Term, feed: dict, *, basis_env=None, target=None) -> np.ndarray:
    kw = {}
    if basis_env is not None:
        kw["basis_env"] = basis_env
    if target is not None:
        kw["target"] = target
    prog = represent(term, **kw)
    return np.asarray(prog.run(feed)["result"], float)


# ---------------------------------------------------------------------------
# Case description.
# ---------------------------------------------------------------------------


@dataclass
class Marked:
    """An x-dependent value var ``u`` with supplied derivative map ``Du`` (a Var of
    ``X ⊸ space(u)``).  ``u0`` is its base value; ``Du_val`` the derivative-map value fed to both
    the Jacobian program and (via the machinery) the finite-difference perturbation of ``u``."""

    u: T.Var
    u0: np.ndarray
    Du: T.Var
    Du_val: np.ndarray


@dataclass
class FDCase:
    """A differentiable term ``F`` of ``x`` with a base feed and (optional) marked deps."""

    term: T.Term
    x: T.Var
    x0: np.ndarray
    marked: list[Marked] = field(default_factory=list)
    consts: dict[T.Var, np.ndarray] = field(default_factory=dict)  # x-independent vars

    @property
    def deps(self) -> dict:
        return {m.u.uid: m.Du for m in self.marked}

    def pinnable(self) -> dict:
        """The operands a basis flavor can pin: x, marked-value vars, marked-derivative maps, and
        const vars — keyed Var -> (space, base value)."""
        out: dict = {self.x: (self.x.space, self.x0)}
        for m in self.marked:
            out[m.u] = (m.u.space, m.u0)
            out[m.Du] = (m.Du.space, m.Du_val)
        for c, val in self.consts.items():
            out[c] = (c.space, val)
        return out


def _base_feeds(c: FDCase) -> dict:
    feed = {_vk(c.x): c.x0}
    for m in c.marked:
        feed[_vk(m.u)] = m.u0
        feed[_vk(m.Du)] = m.Du_val
    for v, val in c.consts.items():
        feed[_vk(v)] = val
    return feed


# ---------------------------------------------------------------------------
# The two-route FD check.
# ---------------------------------------------------------------------------


def _directional_derivative(Du: T.Var, Du_val: np.ndarray, hdir: T.Var, hval, feeds) -> np.ndarray:
    """``Du·h`` via the machinery (never a hand contraction)."""
    return _run(T.Eval(Du, (0,), (hdir,)), {**feeds, _vk(hdir): hval, _vk(Du): Du_val})


def _fd_reference(c: FDCase, hdir: T.Var, hval: np.ndarray, eps: float) -> np.ndarray:
    """Central finite difference of the total derivative: perturb x (and every marked var along
    its own ``Du·h``) by ``±εh`` and difference the lowered term value."""
    base = _base_feeds(c)
    # per-marked directional derivatives Du·h (machinery)
    duh = {m.u: _directional_derivative(m.Du, m.Du_val, hdir, hval, base) for m in c.marked}

    def evF(sign: float) -> np.ndarray:
        feed = {_vk(v): val for v, val in c.consts.items()}
        feed[_vk(c.x)] = c.x0 + sign * eps * hval
        for m in c.marked:
            feed[_vk(m.u)] = m.u0 + sign * eps * duh[m.u]
        return _run(c.term, feed)

    return (evF(+1.0) - evF(-1.0)) / (2.0 * eps)


def _seed_of(J: T.Term) -> T.Var | None:
    """The internal seed binder of the Jacobian ``LLam`` (``None`` for the zero map)."""
    return None if isinstance(J, T.Zero) else J.vars[0]


def _apply_J(J: T.Term, hval: np.ndarray, feeds: dict, *, basis_env=None, target=None) -> np.ndarray | None:
    """``J·h`` — the symbolic directional derivative VALUE, lowered.

    We seed the ``LLam``'s *body* directly (the directional-derivative value, a plain value in
    ``space(e)``) rather than applying ``J`` through ``Eval``: an ``LLam`` map lowers transposed
    ``(cod, dom)`` and ``Eval`` would misread it (the layout trap ``pullback`` also avoids).  A
    ``Zero`` map signals an x-independent term (directional derivative ≡ 0)."""
    seed = _seed_of(J)
    if seed is None:
        return None  # x-independent ⇒ derivative is the zero value
    return _run(J.body, {**feeds, _vk(seed): hval}, basis_env=basis_env, target=target)


def _check_default(c: FDCase, rng: np.random.Generator) -> None:
    """Default-basis two-route: ``J·h == FD`` for a random direction h."""
    hdir = T.var("hdir", c.x.space)
    hval = rng.standard_normal(c.x0.shape)
    J = diff(c.term, c.x, c.deps)
    got = _apply_J(J, hval, _base_feeds(c))
    fd = _fd_reference(c, hdir, hval, _EPS)
    if got is None:  # x-independent map ⇒ FD must be ≈ 0
        assert np.allclose(fd.ravel(), 0.0, atol=_ATOL), f"diff gave the zero map but FD = {fd.ravel()}"
        return
    assert np.allclose(got.ravel(), fd.ravel(), atol=_ATOL), (
        f"SILENT-WRONG derivative: J·h = {got.ravel()} but central FD = {fd.ravel()}"
    )


# ---------------------------------------------------------------------------
# Basis-pinning axis (reuses the basis-honesty flavor machinery).
# ---------------------------------------------------------------------------


def _frame_matrix(n: int) -> np.ndarray:
    if n == 2:
        return np.array([[1.0, 1], [0, 1.0]])
    if n == 3:
        return np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    return np.eye(n) + np.tril(np.ones((n, n)), -1)


def _frame_feeds(frame_vars, M):
    return {_vk(v): M[:, i] for i, v in enumerate(frame_vars)}


def _factor_basis(factor, tag):
    n = represent.__globals__["_static_dim"](factor)
    M = _frame_matrix(n)
    if isinstance(factor, S.Dual):
        vs = [T.var(f"{tag}{i}", factor.space) for i in range(n)]
        return B.dual_basis(B.list_basis(*vs)), _frame_feeds(vs, M)
    vs = [T.var(f"{tag}{i}", factor) for i in range(n)]
    return B.list_basis(*vs), _frame_feeds(vs, M)


def _flavor(space, tag):
    """One non-orthonormal, non-default basis flavor for ``space`` (rank-1 or rank-2), or None."""
    factors = represent.__globals__["_factors"](space)
    if len(factors) == 1:
        n = represent.__globals__["_static_dim"](space)
        fv = [T.var(f"{tag}l{i}", space) for i in range(n)]
        return B.list_basis(*fv), _frame_feeds(fv, _frame_matrix(n))
    if len(factors) == 2:
        b0, f0 = _factor_basis(factors[0], f"{tag}0_")
        b1, f1 = _factor_basis(factors[1], f"{tag}1_")
        return B.basis_prod(b0, b1), {**f0, **f1}
    return None  # rank-≥3 operands: no flavor (covered at default)


def _machine_coords(v, space, basis, frame_feeds, std_value):
    """``std_value`` in ``basis`` — derived by the lowering's own cob (no hand-transform)."""
    return _run(v, {_vk(v): std_value, **frame_feeds}, basis_env={v.uid: default_basis(space)}, target=basis)


def _check_pinned(c: FDCase, pin_choice: dict, rng: np.random.Generator) -> None:
    """Operands in ``pin_choice`` (Var -> (basis, frame_feeds)) are pinned to a non-default basis,
    fed machine-derived coords; the rest stay default.  ``J·h`` at target=STD must still equal the
    basis-independent FD reference, OR the lowering raises (a loud bail is acceptable)."""
    hdir = T.var("hdir", c.x.space)
    hval = rng.standard_normal(c.x0.shape)
    try:
        J = diff(c.term, c.x, c.deps)
    except DiffError:
        return
    seed = _seed_of(J)
    if seed is None:  # x-independent map ⇒ nothing basis-dependent to probe
        return
    pinnable = c.pinnable()
    env, feeds = {}, {}
    for v, (sp, val) in pinnable.items():
        if v in pin_choice:
            basis, ff = pin_choice[v]
            try:
                coords = _machine_coords(v, sp, basis, ff, val)
            except RepresentError:
                return
            env[v.uid] = basis
            feeds[_vk(v)] = coords
            feeds.update(ff)
        else:
            env[v.uid] = default_basis(sp)
            feeds[_vk(v)] = val
    # the LLam's internal seed binder shares x's basis pin (it lives in X)
    if c.x in pin_choice:
        basis, ff = pin_choice[c.x]
        env[seed.uid] = basis
        feeds[_vk(seed)] = _machine_coords(seed, c.x.space, basis, ff, hval)
        feeds.update(ff)
    else:
        env[seed.uid] = default_basis(c.x.space)
        feeds[_vk(seed)] = hval
    try:
        got = _run(J.body, feeds, basis_env=env, target=B.STD)
    except RepresentError:
        return  # loud bail — acceptable
    fd = _fd_reference(c, hdir, hval, _EPS)
    assert np.allclose(got.ravel(), fd.ravel(), atol=_ATOL), (
        f"SILENT-WRONG (pinned) derivative: J·h(STD) = {got.ravel()} but FD = {fd.ravel()}"
    )


# ===========================================================================
# The case registry.  Each builder returns a fresh FDCase (new Vars each call).
# ===========================================================================

V = S.base("V", 3)
DV = S.dual(V)
MS = S.oprod(DV, V)  # V ⊸ V, stored (dom, cod)
V2 = S.base("V2", 2)
M2 = S.oprod(S.dual(V2), V2)
V4 = S.base("V4", 4)
M4 = S.oprod(S.dual(V4), V4)


def _rng_vals(rng, *shapes):
    return [rng.standard_normal(s) for s in shapes]


# --- direct-x dependence ---------------------------------------------------


def _b_identity():
    x = T.var("x", V)
    return FDCase(term=x, x=x, x0=np.array([1.0, 2, 3]))


def _b_inner_xx():
    x = T.var("x", V)
    return FDCase(term=T.Eval(T.inner(V), (0, 1), (x, x)), x=x, x0=np.array([1.0, -2, 0.5]))


def _b_sqrt_norm():
    x = T.var("x", V)
    e = L.sqrt(G.wrap(T.Eval(T.inner(V), (0, 1), (x, x)))).node
    return FDCase(term=e, x=x, x0=np.array([1.3, -0.7, 2.0]))


def _b_recip_norm():
    x = T.var("x", V)
    e = L.reciprocal(G.wrap(T.Eval(T.inner(V), (0, 1), (x, x)))).node
    return FDCase(term=e, x=x, x0=np.array([1.3, -0.7, 2.0]))


def _b_div():
    x = T.var("x", V)
    c = T.var("c", DV)
    num = G.wrap(T.Eval(T.inner(V), (0, 1), (x, x)))
    den = G.wrap(T.Eval(c, (0,), (x,)))
    e = L.div(num, den).node
    return FDCase(term=e, x=x, x0=np.array([1.0, 2, 3]), consts={c: np.array([1.0, 0.5, -0.3])})


def _b_abs():
    x = T.var("x", V)
    c = T.var("c", DV)
    e = L.absval(G.wrap(T.Eval(c, (0,), (x,)))).node
    return FDCase(term=e, x=x, x0=np.array([1.0, 2, 3]), consts={c: np.array([1.0, 0.5, -0.3])})


def _b_quad_form():
    x = T.var("x", V)
    A = T.var("A", MS)
    e = T.Eval(T.inner(V), (0, 1), (T.Eval(A, (0,), (x,)), x))  # ⟨Ax, x⟩
    Aval = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])
    return FDCase(term=e, x=x, x0=np.array([1.0, -1, 2]), consts={A: Aval})


def _b_map_apply():
    x = T.var("x", V)
    A = T.var("A", MS)
    Aval = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])
    return FDCase(term=T.Eval(A, (0,), (x,)), x=x, x0=np.array([1.0, 2, 3]), consts={A: Aval})


def _b_compose_apply():
    x = T.var("x", V)
    A, Bm = T.var("A", MS), T.var("Bm", MS)
    e = T.Eval(T.Compose(A, (0,), (Bm,)), (0,), (x,))
    Av = np.array([[2.0, 0, 1], [1, 1, 0], [0, 3, 1]])
    Bv = np.array([[1.0, 2, 0], [0, 1, 1], [1, 0, 2]])
    return FDCase(term=e, x=x, x0=np.array([1.0, 0, 2]), consts={A: Av, Bm: Bv})


def _b_self_oprod():
    x = T.var("x", V)
    return FDCase(term=T.OProdV((x, x)), x=x, x0=np.array([1.0, 2, 3]))


def _b_bilinear_xx():
    x = T.var("x", V)
    Bm = T.var("Bm", S.oprod(DV, DV))
    e = T.Eval(Bm, (0, 1), (x, x))
    Bv = np.array([[1.0, 0.2, 0], [0.1, 2, 0], [0, 0.3, 1.0]])
    return FDCase(term=e, x=x, x0=np.array([1.0, -1, 2]), consts={Bm: Bv})


def _b_solve_apply():
    x = T.var("x", V)
    A = T.var("A", MS, props=G.bij)
    Av = np.array([[3.0, 0, 1], [1, 4, 0], [0, 1, 3.0]])
    return FDCase(term=T.SolveApply(A, x), x=x, x0=np.array([1.0, 2, 3]), consts={A: Av})


def _b_pinv_apply():
    x = T.var("x", V)
    A = T.var("A", MS)
    Av = np.array([[3.0, 0, 1], [1, 4, 0], [0, 1, 3.0]])
    return FDCase(term=T.PinvApply(A, x), x=x, x0=np.array([1.0, 2, 3]), consts={A: Av})


def _b_scale_const():
    x = T.var("x", V)
    return FDCase(term=T.Scale(2.5, x), x=x, x0=np.array([1.0, 2, 3]))


def _b_scale_scalar():
    x = T.var("x", V)
    s = G.wrap(T.Eval(T.inner(V), (0, 1), (x, x)))  # ⟨x,x⟩·x — scalar·vector, both depend on x
    e = T.Scale(s.node, x)
    return FDCase(term=e, x=x, x0=np.array([1.0, -1, 2]))


def _b_sum_const():
    x = T.var("x", V)
    c = T.var("c", V)
    return FDCase(term=T.Sum((x, c)), x=x, x0=np.array([1.0, 2, 3]), consts={c: np.array([0.0, 1, 4])})


def _b_trace_outer():
    x = T.var("x", V)
    b = T.var("b", DV)
    e = T.Trace(T.OProdV((x, b)), 0, 1)  # ⟨b, x⟩
    return FDCase(term=e, x=x, x0=np.array([1.0, 2, 3]), consts={b: np.array([0.5, 1, -0.3])})


def _b_permute():
    x = T.var("x", V)
    e = T.Permute(T.OProdV((x, x)), (1, 0))
    return FDCase(term=e, x=x, x0=np.array([1.0, 2, 3]))


def _b_project():
    x = T.var("x", V)
    return FDCase(term=T.Project(x, V), x=x, x0=np.array([1.0, 2, 3]))


def _b_embed():
    x = T.var("x", V)
    return FDCase(term=T.Embed(x, V), x=x, x0=np.array([1.0, 2, 3]))


def _b_dagger_apply():
    x = T.var("x", V)
    A = T.var("A", MS)
    Av = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])
    e = T.Eval(T.Dagger(A), (0,), (x,))
    return FDCase(term=e, x=x, x0=np.array([1.0, 2, 3]), consts={A: Av})


def _b_nested_sqrt_recip():
    x = T.var("x", V)
    inner = G.wrap(T.Eval(T.inner(V), (0, 1), (x, x)))
    e = L.sqrt(L.reciprocal(inner)).node  # 1/|x|
    return FDCase(term=e, x=x, x0=np.array([1.0, -1, 2]))


def _b_dim2_inner():
    x = T.var("x", V2)
    return FDCase(term=T.Eval(T.inner(V2), (0, 1), (x, x)), x=x, x0=np.array([1.3, -0.7]))


def _b_dim4_map():
    x = T.var("x", V4)
    A = T.var("A", M4)
    Av = np.eye(4) + 0.3 * np.tril(np.ones((4, 4)), -1)
    return FDCase(term=T.Eval(A, (0,), (x,)), x=x, x0=np.array([1.0, 2, 3, 4]), consts={A: Av})


# --- marked-dep regime -----------------------------------------------------


def _mk_inner_uv(rng):
    x = T.var("x", V)
    u, v = T.var("u", V), T.var("v", V)
    Du, Dv = T.var("Du", MS), T.var("Dv", MS)
    e = T.Eval(T.inner(V), (0, 1), (u, v))
    u0, v0 = _rng_vals(rng, (3,), (3,))
    Duv, Dvv = _rng_vals(rng, (3, 3), (3, 3))
    return FDCase(term=e, x=x, x0=np.array([0.0, 0, 0]),
                  marked=[Marked(u, u0, Du, Duv), Marked(v, v0, Dv, Dvv)])


def _mk_map_marked(rng):
    x = T.var("x", V)
    A = T.var("A", MS)
    v = T.var("v", V)
    DA = T.var("DA", S.oprod(DV, DV, V))  # dA/dx : X ⊸ (V⊸V)
    e = T.Eval(A, (0,), (v,))
    Av, vv = _rng_vals(rng, (3, 3), (3,))
    DAv = rng.standard_normal((3, 3, 3))
    return FDCase(term=e, x=x, x0=np.array([0.0, 0, 0]), consts={v: vv},
                  marked=[Marked(A, Av, DA, DAv)])


def _mk_dagger_marked(rng):
    x = T.var("x", V)
    A = T.var("A", MS)
    v = T.var("v", V)
    DA = T.var("DA", S.oprod(DV, DV, V))
    e = T.Eval(T.Dagger(A), (0,), (v,))
    Av, vv = _rng_vals(rng, (3, 3), (3,))
    DAv = rng.standard_normal((3, 3, 3))
    return FDCase(term=e, x=x, x0=np.array([0.0, 0, 0]), consts={v: vv},
                  marked=[Marked(A, Av, DA, DAv)])


def _mk_compose_marked(rng):
    x = T.var("x", V)
    A, Bm = T.var("A", MS), T.var("Bm", MS)
    v = T.var("v", V)
    DA = T.var("DA", S.oprod(DV, DV, V))
    e = T.Eval(T.Compose(A, (0,), (Bm,)), (0,), (v,))
    Av, Bv, vv = _rng_vals(rng, (3, 3), (3, 3), (3,))
    DAv = rng.standard_normal((3, 3, 3))
    return FDCase(term=e, x=x, x0=np.array([0.0, 0, 0]), consts={Bm: Bv, v: vv},
                  marked=[Marked(A, Av, DA, DAv)])


def _mk_kron_marked(rng):
    x = T.var("x", V2)
    f, g = T.var("f", M2), T.var("g", M2)
    Df = T.var("Df", S.oprod(S.dual(V2), S.dual(V2), V2))
    e = T.Trace(T.Kron(f, g), 1, 0)  # tr(f⊗g) = tr(f)tr(g) — scalar
    fv, gv = _rng_vals(rng, (2, 2), (2, 2))
    Dfv = rng.standard_normal((2, 2, 2))
    return FDCase(term=e, x=x, x0=np.array([0.0, 0]), consts={g: gv},
                  marked=[Marked(f, fv, Df, Dfv)])


def _mk_remetric_flat(rng):
    # constant metric ⇒ Remetric is flat ⇒ plain total derivative (FD-checkable).
    x = T.var("x", V)
    sec = T.var("sec", V)
    g = T.var("g", MS)  # NOT in deps ⇒ constant ⇒ flat
    Dsec = T.var("Dsec", MS)
    e = T.Remetric(sec, g)
    secv = rng.standard_normal((3,))
    Dsecv = rng.standard_normal((3, 3))
    gv = np.array([[2.0, 0, 0], [0, 3, 0], [0, 0, 1.0]])
    return FDCase(term=e, x=x, x0=np.array([0.0, 0, 0]), consts={g: gv},
                  marked=[Marked(sec, secv, Dsec, Dsecv)])


def _mk_oprod_marked(rng):
    x = T.var("x", V)
    u, v = T.var("u", V), T.var("v", V)
    Du, Dv = T.var("Du", MS), T.var("Dv", MS)
    e = T.OProdV((u, v))
    u0, v0 = _rng_vals(rng, (3,), (3,))
    Duv, Dvv = _rng_vals(rng, (3, 3), (3, 3))
    return FDCase(term=e, x=x, x0=np.array([0.0, 0, 0]),
                  marked=[Marked(u, u0, Du, Duv), Marked(v, v0, Dv, Dvv)])


def _mk_solve_apply_marked_v(rng):
    # d(A⁻¹ v)/dx with v marked (seed through the vector), A constant.
    x = T.var("x", V)
    A = T.var("A", MS, props=G.bij)
    v = T.var("v", V)
    Dv = T.var("Dv", MS)
    e = T.SolveApply(A, v)
    Av = np.array([[3.0, 0, 1], [1, 4, 0], [0, 1, 3.0]])
    v0 = rng.standard_normal((3,))
    Dvv = rng.standard_normal((3, 3))
    return FDCase(term=e, x=x, x0=np.array([0.0, 0, 0]), consts={A: Av},
                  marked=[Marked(v, v0, Dv, Dvv)])


# direct-x cases take no rng; marked cases take rng.  Tag accordingly.
DIRECT_CASES = [
    ("identity", _b_identity), ("inner_xx", _b_inner_xx), ("sqrt_norm", _b_sqrt_norm),
    ("recip_norm", _b_recip_norm), ("div", _b_div), ("abs", _b_abs),
    ("quad_form", _b_quad_form), ("map_apply", _b_map_apply), ("compose_apply", _b_compose_apply),
    ("self_oprod", _b_self_oprod), ("bilinear_xx", _b_bilinear_xx), ("solve_apply", _b_solve_apply),
    ("pinv_apply", _b_pinv_apply), ("scale_const", _b_scale_const), ("scale_scalar", _b_scale_scalar),
    ("sum_const", _b_sum_const), ("trace_outer", _b_trace_outer), ("permute", _b_permute),
    ("project", _b_project), ("embed", _b_embed), ("dagger_apply", _b_dagger_apply),
    ("nested_sqrt_recip", _b_nested_sqrt_recip), ("dim2_inner", _b_dim2_inner),
    ("dim4_map", _b_dim4_map),
]

MARKED_CASES = [
    ("inner_uv", _mk_inner_uv), ("map_marked", _mk_map_marked), ("dagger_marked", _mk_dagger_marked),
    ("compose_marked", _mk_compose_marked), ("kron_marked", _mk_kron_marked),
    ("remetric_flat", _mk_remetric_flat), ("oprod_marked", _mk_oprod_marked),
    ("solve_apply_marked_v", _mk_solve_apply_marked_v),
]

ALL_CASES = (
    [(n, lambda b=b: b(), False) for n, b in DIRECT_CASES]
    + [(n, b, True) for n, b in MARKED_CASES]
)


def _build(builder, needs_rng, seed):
    rng = np.random.default_rng(seed)
    return (builder(rng) if needs_rng else builder()), rng


@pytest.fixture(params=["vmap", "eye"])
def llam_mode(request, monkeypatch):
    monkeypatch.setenv("GRASSMANN_LLAM_INDEX", request.param)
    return request.param


@pytest.mark.parametrize("name,builder,needs_rng", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_fd_default(name, builder, needs_rng, llam_mode):
    """``J·h == central FD`` at the default basis, for both LLam lowering modes."""
    case, rng = _build(builder, needs_rng, _SEED)
    _check_default(case, rng)


@pytest.mark.parametrize("name,builder,needs_rng", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_fd_pinned(name, builder, needs_rng, llam_mode):
    """Subset basis-pinning: each pinnable operand independently default or a non-default flavor;
    ``J·h`` at target=STD still equals the basis-independent FD reference (or a loud bail)."""
    import itertools

    case, _ = _build(builder, needs_rng, _SEED)
    pinnable = case.pinnable()
    # build a flavor per pinnable var (skip those with no flavor — rank-≥3)
    flav = {}
    for v, (sp, _val) in pinnable.items():
        f = _flavor(sp, f"{_vk(v)}_")
        if f is not None:
            flav[v] = f
    if not flav:
        pytest.skip("no pinnable operand has a basis flavor (rank-≥3 only)")
    vars_ = list(flav)
    # sweep each operand ∈ {default, flavor}; skip the all-default combo
    for combo in itertools.product(*([None, flav[v]] for v in vars_)):
        if all(c is None for c in combo):
            continue
        choice = {v: c for v, c in zip(vars_, combo) if c is not None}
        rng = np.random.default_rng(_SEED)
        _check_pinned(case, choice, rng)


# ---------------------------------------------------------------------------
# Rejection cases — a LOUD raise is the correct behavior (never silent-wrong).
# ---------------------------------------------------------------------------


def _r_neval():
    V_ = S.base("V", 2)
    x = T.var("x", V_)
    return lambda: diff(T.NEval(T.var("f", V_), (0,), (x,)), x)


def _r_bare_solve_marked():
    V_ = S.base("V", 2)
    x = T.var("x", V_)
    A = T.var("A", S.oprod(S.dual(V_), V_), props=G.bij)
    DA = T.var("DA", S.oprod(S.dual(V_), S.oprod(S.dual(V_), V_)))
    return lambda: diff(T.SolveApply(A, T.var("b", V_)), x, {A.uid: DA})


def _r_inverse_through_map():
    V_ = S.base("V", 2)
    x = T.var("x", V_)
    A = T.var("A", S.oprod(S.dual(V_), V_), props=G.bij)
    DA = T.var("DA", S.oprod(S.dual(V_), S.oprod(S.dual(V_), V_)))
    return lambda: diff(T.SolveApply(A, x), x, {A.uid: DA})


@pytest.mark.parametrize("name,builder", [
    ("neval_higher_order", _r_neval),
    ("bare_solve_marked", _r_bare_solve_marked),
    ("inverse_through_map", _r_inverse_through_map),
])
def test_diff_rejections(name, builder):
    """Unsupported differentiation raises ``DiffError`` (loud) — never a silent wrong map."""
    with pytest.raises(DiffError):
        builder()()
