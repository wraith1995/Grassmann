"""Tests for total covariant differentiation (:mod:`grassmann.transform.diff`).

``diff(e, x, deps)`` returns the total derivative of a closed term ``e`` w.r.t. a value
variable ``x`` as a coordinate-free map ``X ⊸ space(e)``. Cases are checked against
central-difference Jacobians of the value term, closed forms, and — for the covariant
case — a hand-computed Levi-Civita connection.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import grassmann as G
import grassmann.stdlib as L
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.infer.linearity import check_linearity
from grassmann.transform.diff import DiffError, diff

# Elementwise (Hadamard) derivatives over a VECTOR materialize the seed as a Jacobian; eye
# mode cannot thread that seed through the Hadamard external (a (n,)⊙(n,n) shape clash) — the
# same documented seed-threading gap as the SolveOp/PinvApply eye-mode cases. The feature is
# correct in the default vmap mode, so these checks are vmap-only.
_vmap_only = pytest.mark.skipif(
    os.environ.get("GRASSMANN_LLAM_INDEX", "vmap") == "eye",
    reason="elementwise vector AD threads the seed through a Hadamard external; eye mode "
    "cannot (use default vmap) — same class as the SolveOp eye-mode gap",
)


def _xkey(x: T.Var) -> str:
    return f"{x.name}_{x.uid}"


def _run(term: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(G.compile(term).run(feed)["result"])


def _fd_jacobian(value: T.Term, x: T.Var, base_feed: dict, n: int, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian d(value)/dx (columns = ∂/∂xᵢ)."""
    key = _xkey(x)
    cols = []
    x0 = np.asarray(base_feed[key], dtype=float)
    for i in range(n):
        fp, fm = dict(base_feed), dict(base_feed)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        fp[key], fm[key] = xp, xm
        cols.append((_run(value, fp) - _run(value, fm)) / (2 * h))
    return np.stack(cols, axis=-1)


# ---------------------------------------------------------------------------
# Leaves / linearity.
# ---------------------------------------------------------------------------


def test_identity_derivative() -> None:
    V = S.base("V", 3)
    x = T.var("x", V)
    J = diff(x, x)  # d x / d x = Identity
    out = _run(J, {_xkey(x): np.zeros(3)})
    np.testing.assert_allclose(out, np.eye(3))


def test_constant_is_zero_map() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    c = T.var("c", V)  # x-independent
    J = diff(c, x)
    assert isinstance(J, T.Zero)


def test_xindependent_direct_sum_is_zero_map() -> None:
    """An x-independent ``OPlusV`` differentiates to the zero map, not an unused-seed ``LLam``.

    The ``OPlusV`` arm builds ``⊕ᵢ dvᵢ``; when every component is x-independent that is
    ``⊕ᵢ 0``, which must collapse to ``Zero`` (as the ``Sum`` / ``OProdV`` arms do).  A
    regression guard: it previously produced ``OPlusV(Zero, Zero)`` and the diff wrapper then
    wrapped it in ``LLam((h,), …)`` over an unused seed, which ``LLam`` rejects
    (``linear variable(s) do not occur in the body``).
    """
    A = S.base("A", 2)
    x = T.var("x", S.base("X", 2))
    c1, c2 = T.var("c1", A), T.var("c2", A)
    assert isinstance(diff(T.OPlusV((c1, c2)), x), T.Zero)
    # also through a nonlinearity on a constant component
    assert isinstance(diff(T.OPlusV((L.tanh(c1).node, c2)), x), T.Zero)


def test_mixed_direct_sum_derivative_matches_fd() -> None:
    """A direct sum with one x-dependent component differentiates correctly (the x-independent
    block reads zero), confirming the all-zero collapse did not disturb the mixed case."""
    A = S.base("A", 2)
    X = S.base("X", 2)
    x = T.var("x", X)
    M = T.var("M", S.oprod(S.dual(X), A))
    c = T.var("c", A)  # x-independent block
    F = T.OPlusV((T.Eval(M, (0,), (x,)), c))  # (M x) ⊕ c
    J = diff(F, x)
    assert isinstance(J, T.LLam)
    Mval, cval, xval = np.random.randn(2, 2), np.random.randn(2), np.random.randn(2)
    feed = {f"M_{M.uid}": Mval, f"c_{c.uid}": cval, _xkey(x): xval}
    h = np.random.randn(2)
    seed = J.vars[0]
    got = _run(J.body, {**feed, f"{seed.name}_{seed.uid}": h}).ravel()
    fd = _fd_jacobian(F, x, feed, 2) @ h
    np.testing.assert_allclose(got, fd, atol=1e-5)
    # the map lowers as xᵀM (value shape (dom, cod)), so d(M x)·h = Mᵀh in the first block;
    # the constant block is 0
    np.testing.assert_allclose(got, np.concatenate([Mval.T @ h, np.zeros(2)]), atol=1e-6)


# ---------------------------------------------------------------------------
# Product / chain rule vs finite differences.
# ---------------------------------------------------------------------------


def test_quadratic_form_matches_closed_form() -> None:
    V = S.base("V", 3)
    x = T.var("x", V)
    A = T.var("A", S.oprod(S.dual(V), V))
    e = T.Eval(T.inner(V), (0, 1), (T.Eval(A, (0,), (x,)), x))  # ⟨Ax, x⟩
    J = diff(e, x)
    Aval, xval = np.random.randn(3, 3), np.random.randn(3)
    feed = {f"A_{A.uid}": Aval, _xkey(x): xval}
    Jv = _run(J, feed).ravel()
    np.testing.assert_allclose(Jv, xval @ (Aval + Aval.T), atol=1e-6)
    np.testing.assert_allclose(Jv, _fd_jacobian(e, x, feed, 3).ravel(), atol=1e-4)


def test_bilinear_form_over_marked_vars() -> None:
    # The primary case: a bilinear form over x-dependent marked vars.
    V = S.base("V", 3)
    x = T.var("x", V)
    u, v = T.var("u", V), T.var("v", V)
    Du = T.var("Du", S.oprod(S.dual(V), V))
    Dv = T.var("Dv", S.oprod(S.dual(V), V))
    e = T.Eval(T.inner(V), (0, 1), (u, v))  # ⟨u, v⟩
    J = diff(e, x, deps={u.uid: Du, v.uid: Dv})
    check_linearity(J)
    Duv, Dvv = np.random.randn(3, 3), np.random.randn(3, 3)
    uval, vval = np.random.randn(3), np.random.randn(3)
    feed = {f"Du_{Du.uid}": Duv, f"Dv_{Dv.uid}": Dvv, f"u_{u.uid}": uval, f"v_{v.uid}": vval}
    Jv = _run(J, feed).ravel()
    # d⟨u,v⟩·h = ⟨Du h, v⟩ + ⟨u, Dv h⟩ ⇒ J = vᵀDuᵀ + uᵀDvᵀ.
    np.testing.assert_allclose(Jv, vval @ Duv.T + uval @ Dvv.T, atol=1e-6)


def test_marked_none_dep_becomes_input() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    u = T.var("u", V)
    e = T.Eval(T.inner(V), (0, 1), (u, u))
    J = diff(e, x, deps={u.uid: None})  # fresh symbolic ∂u/∂x
    # a fresh map var appears among the program inputs.
    prog = G.compile(J)
    names = {getattr(i, "name", "") for i in prog.inputs}
    assert any(n.startswith("d_u_") for n in names)


# ---------------------------------------------------------------------------
# Nonlinear scalar ops.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        L.sqrt,
        L.reciprocal,
        L.exp,
        L.log,
        L.square,
        L.sin,
        L.cos,
        L.tan,
        L.sinh,
        L.cosh,
        L.tanh,
    ],
)
def test_scalar_chain_rules_vs_fd(op) -> None:
    # op applied to ⟨x,x⟩ (a positive scalar — in-domain for sqrt/log) then FD-checked.
    V = S.base("V", 2)
    x = T.var("x", V)
    dotxx = T.Eval(T.inner(V), (0, 1), (x, x))
    e = op(G.wrap(dotxx)).node
    J = diff(e, x)
    xval = np.array([1.3, -0.7])
    feed = {_xkey(x): xval}
    np.testing.assert_allclose(_run(J, feed).ravel(), _fd_jacobian(e, x, feed, 2).ravel(), atol=1e-4)


def _sig(u):
    return 1.0 / (1.0 + np.exp(-u))


@pytest.mark.parametrize(
    ("op", "fprime"),
    [
        (L.exp, np.exp),
        (L.sin, np.cos),
        (L.cos, lambda u: -np.sin(u)),
        (L.tan, lambda u: 1.0 / np.cos(u) ** 2),
        (L.sinh, np.cosh),
        (L.tanh, lambda u: 1.0 / np.cosh(u) ** 2),
        (L.square, lambda u: 2.0 * u),
        # xv below has |components| < 1, in-domain for asin/acos
        (L.atan, lambda u: 1.0 / (1.0 + u**2)),
        (L.asin, lambda u: 1.0 / np.sqrt(1.0 - u**2)),
        (L.acos, lambda u: -1.0 / np.sqrt(1.0 - u**2)),
        (L.sigmoid, lambda u: _sig(u) * (1.0 - _sig(u))),
        (L.softplus, _sig),
    ],
)
@_vmap_only
def test_elementwise_vector_derivative_vs_fd(op, fprime) -> None:
    """diff of an elementwise op over a VECTOR is the Jacobian ``diag(f'(u))`` — via the
    Hadamard pointwise op. Checked vs the closed diagonal and central FD."""
    V = S.base("V", 3)
    x = T.var("x", V)
    e = op(G.wrap(x)).node
    J = diff(e, x)
    xv = np.array([0.2, -0.4, 0.5])
    feed = {_xkey(x): xv}
    Jv = _run(J, feed)
    np.testing.assert_allclose(Jv, np.diag(fprime(xv)), atol=1e-6)
    np.testing.assert_allclose(Jv, _fd_jacobian(e, x, feed, 3), atol=1e-4)


@pytest.mark.parametrize("op", [L.atan, L.asin, L.acos, L.sigmoid, L.softplus])
def test_batch2_scalar_derivative_vs_fd(op) -> None:
    """Scalar derivatives (built with the unit-constant ``_ones``) vs central FD.
    Operand ⟨x,x⟩ is small (≈0.13) so asin/acos stay in domain."""
    V = S.base("V", 2)
    x = T.var("x", V)
    dotxx = T.Eval(T.inner(V), (0, 1), (x, x))
    e = op(G.wrap(dotxx)).node
    J = diff(e, x)
    xval = np.array([0.3, 0.2])
    feed = {_xkey(x): xval}
    np.testing.assert_allclose(_run(J, feed).ravel(), _fd_jacobian(e, x, feed, 2).ravel(), atol=1e-4)


@_vmap_only
def test_hadamard_bilinear_derivative_vs_fd() -> None:
    """The Hadamard product is bilinear: d(a⊙b)=da⊙b+a⊙db. With a=x and b fixed, d(x⊙b)/dx
    = diag(b). FD-checked."""
    V = S.base("V", 3)
    x = T.var("x", V)
    b = T.var("b", V)  # x-independent
    e = L.hadamard(G.wrap(x), G.wrap(b)).node  # x ⊙ b
    J = diff(e, x)
    xv, bv = np.array([1.0, -2.0, 0.5]), np.array([3.0, 0.7, -1.0])
    feed = {_xkey(x): xv, f"b_{b.uid}": bv}
    np.testing.assert_allclose(_run(J, feed), np.diag(bv), atol=1e-6)
    np.testing.assert_allclose(_run(J, feed), _fd_jacobian(e, x, feed, 3), atol=1e-4)


# NOTE: second-order AD through these ops (differentiating an already-differentiated
# LLam map) trips a higher-order `represent` limitation (Scalar≅ℝ¹ cob / a free-var
# 'b' KeyError), independent of the derivative arms — that is arbitrary/higher-
# order AD territory, not op registration. The first-order chain rule above covers all
# nine ops; each arm composes from primitives, so it is higher-order-safe by construction
# once the engine supports it.


# ---------------------------------------------------------------------------
# Inverses.
# ---------------------------------------------------------------------------


def test_inverse_apply_through_vector() -> None:
    import os

    from grassmann.lower.represent import RepresentError

    V = S.base("V", 2)
    x = T.var("x", V)
    A = T.var("A", S.oprod(S.dual(V), V), props=G.bij)
    e = T.SolveApply(A, x)  # A⁻¹ x  (seed through the vector)
    J = diff(e, x)
    Aval = np.random.randn(2, 2) + 3 * np.eye(2)
    xval = np.random.randn(2)
    feed = {f"A_{A.uid}": Aval, _xkey(x): xval}
    # The derivative is the *applied* inverse SolveApply(A, h) = A⁻¹h (the same operator as the
    # primal), NOT the transpose A⁻ᵀh that routing through Eval(Solve(A), ·) would give.  Verify the
    # directional derivative directly (seed the body — layout-clean) against the central FD.
    hv = np.random.randn(2)
    seed = J.vars[0]
    if os.environ.get("GRASSMANN_LLAM_INDEX", "vmap") == "eye":
        # eye mode cannot thread the free seed through the bulk SolveOp — a loud, clean raise.
        with pytest.raises(RepresentError):
            _run(J, feed)
        return
    jh = _run(J.body, {**feed, f"{seed.name}_{seed.uid}": hv})
    fdh = (
        np.asarray(G.compile(e).run({**feed, _xkey(x): xval + 1e-6 * hv})["result"])
        - np.asarray(G.compile(e).run({**feed, _xkey(x): xval - 1e-6 * hv})["result"])
    ) / 2e-6
    np.testing.assert_allclose(np.asarray(jh).ravel(), fdh.ravel(), atol=1e-4)


def test_diff_through_marked_map_applied() -> None:
    # d(A(x)·v)/dx with A a marked map: exercises a seed-carrying map applied via Eval.
    M = S.base("M", 3)
    x = T.var("x", M)
    A = T.var("A", S.oprod(S.dual(M), M))
    v = T.var("v", M)
    DA = T.var("DA", S.oprod(S.dual(M), S.dual(M), M))  # dA/dx : M ⊸ (M ⊸ M), flat
    J = diff(T.Eval(A, (0,), (v,)), x, deps={A.uid: DA})
    DAv, vval, Aval = np.random.randn(3, 3, 3), np.random.randn(3), np.random.randn(3, 3)
    feed = {f"A_{A.uid}": Aval, f"v_{v.uid}": vval, f"DA_{DA.uid}": DAv}
    Jv = _run(J, feed)
    # Jacobian now stored (dom, cod) — output indices (i, k).
    np.testing.assert_allclose(Jv, np.einsum("imk,m->ik", DAv, vval), atol=1e-6)


def test_inverse_through_map_is_rejected() -> None:
    # The APPLIED inverse SolveApply(A, b) with A marked still routes the seed through the bulk op
    # ⇒ rejected.  (The BARE form Eval(Solve(A), b) IS supported — see below.)
    V = S.base("V", 2)
    x = T.var("x", V)
    A = T.var("A", S.oprod(S.dual(V), V), props=G.bij)
    b = T.var("b", V)
    DA = T.var("DA", S.oprod(S.dual(V), S.oprod(S.dual(V), V)))
    with pytest.raises(DiffError):
        diff(T.SolveApply(A, b), x, deps={A.uid: DA})


def _fd_marked_map(term, x, mapvar, Dvar, base_feed, mapval, Dval, hv, atol=1e-4):
    """FD-check diff(term, x, {mapvar: Dvar}) where term depends on x only through the marked map:
    perturb mapval along ∂map·h (machinery) and central-difference."""
    J = diff(term, x, deps={mapvar.uid: Dvar})
    seed = J.vars[0]
    feed = {**base_feed, f"{mapvar.name}_{mapvar.uid}": mapval, f"{Dvar.name}_{Dvar.uid}": Dval}
    jh = _run(J.body, {**feed, f"{seed.name}_{seed.uid}": hv})
    hd = T.var("hd", x.space)
    dmh = _run(T.Eval(Dvar, (0,), (hd,)), {**feed, f"hd_{hd.uid}": hv})
    fp = _run(term, {**feed, f"{mapvar.name}_{mapvar.uid}": mapval + 1e-6 * dmh})
    fm = _run(term, {**feed, f"{mapvar.name}_{mapvar.uid}": mapval - 1e-6 * dmh})
    np.testing.assert_allclose(np.asarray(jh).ravel(), ((fp - fm) / 2e-6).ravel(), atol=atol)


def test_bare_solve_map_derivative() -> None:
    # d(f⁻¹)/dx = −f⁻¹(df)f⁻¹, the BARE inverse-map derivative (un-gated, term→term).  Probed as
    # ⟨w, Eval(Solve(f), z)⟩ with f marked.  The seed threads through the einsum Composes, not the
    # bulk SolveOp.
    rng = np.random.default_rng(1)
    V = S.base("V", 3)
    x = T.var("x", V)
    f = T.var("f", S.oprod(S.dual(V), V), props=G.bij)
    Df = T.var("Df", S.oprod(S.dual(V), S.dual(V), V))
    w, z = T.var("w", S.dual(V)), T.var("z", V)
    e = T.Eval(w, (0,), (T.Eval(T.Solve(f), (0,), (z,)),))
    fv = rng.standard_normal((3, 3)) + 3 * np.eye(3)
    base = {f"w_{w.uid}": rng.standard_normal(3), f"z_{z.uid}": rng.standard_normal(3)}
    _fd_marked_map(e, x, f, Df, base, fv, rng.standard_normal((3, 3, 3)), rng.standard_normal(3))


@pytest.mark.parametrize("ambient", [2, 4])  # square (=Solve) and rectangular (=Golub–Pereyra rank term)
def test_bare_pinv_map_derivative(ambient) -> None:
    # d(J⁺)/dx — the BARE pseudoinverse-map derivative.  ⟨w, Eval(Pinv(J), z)⟩ with J marked.
    # This is exactly `project` through a moving Img(J): project ≡ Eval(Pinv(J), ·), term→term.
    rng = np.random.default_rng(2 + ambient)
    Rt, Re = S.base("Rt", 2), S.base("Re", ambient)
    x = T.var("x", Rt)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.bij if ambient == 2 else G.inj)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    w, z = T.var("w", S.dual(Rt)), T.var("z", Re)
    e = T.Eval(w, (0,), (T.Eval(T.Pinv(J), (0,), (z,)),))
    Jv = rng.standard_normal((2, ambient))
    base = {f"w_{w.uid}": rng.standard_normal(2), f"z_{z.uid}": rng.standard_normal(ambient)}
    _fd_marked_map(e, x, J, DJ, base, Jv, rng.standard_normal((2, 2, ambient)), rng.standard_normal(2))


def test_sqrt_spd_map_derivative() -> None:
    # d(√f) — the SPD operator-square-root derivative.  dS solves the Sylvester  S·dS + dS·S = df
    # (S = √f), built term→term as the inverse of the Sylvester operator on the MAP space
    # (V⊸V)⊸(V⊸V) — Kron of the dual(V) and V factors, NO basis fuse, result lands as a genuine
    # V⊸V map.  Probed as ⟨w, √(JᵀJ)·u⟩ with J marked, so the immersed metric G = JᵀJ (hence df) is
    # automatically symmetric.  √G is the metric-whitening factor whose derivative makes the
    # moving-subspace embed/project AD basis-honest in any frame.
    rng = np.random.default_rng(7)
    Rt, Re = S.base("Rt", 3), S.base("Re", 3)
    x = T.var("x", Rt)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    w, u = T.var("w", S.dual(Rt)), T.var("u", Rt)
    Gm = T.Compose(T.Dagger(J), (0,), (J,))  # JᵀJ : Rt ⊸ Rt, SPD
    e = T.Eval(w, (0,), (T.Eval(T.Sqrt(Gm), (0,), (u,)),))
    Jv = rng.standard_normal((3, 3))
    base = {f"w_{w.uid}": rng.standard_normal(3), f"u_{u.uid}": rng.standard_normal(3)}
    _fd_marked_map(e, x, J, DJ, base, Jv, rng.standard_normal((3, 3, 3)), rng.standard_normal(3))


def test_polar_gauge_engages_and_is_correct() -> None:
    # The gauge mechanism: the subspace gauge is read by `diff` only (`represent` untouched).
    # (1) the default `qr` gauge ⇒ realization has no Sqrt (the opaque QR frame is a fed var);
    #     registering `polar` for J ⇒ the realization IS Sqrt-bearing (Q = J·G^{-1/2}).
    # (2) the polar realization formula dQ·(Q⁺·embed v) is FD-correct — checked through the explicit
    #     polar frame, since `represent` has no polar frame for the structural Embed yet.
    from grassmann.transform.diff import clear_subspace_gauges, set_subspace_gauge

    rng = np.random.default_rng(11)
    Rt, Re = S.base("Rt", 3), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    x = T.var("x", Rt)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    clear_subspace_gauges()
    try:
        assert "Sqrt" not in repr(diff(e, x, deps={v.uid: Dv, J.uid: DJ}))  # immersion default
        set_subspace_gauge(J, "polar")
        assert "Sqrt" in repr(diff(e, x, deps={v.uid: Dv, J.uid: DJ}))  # polar engaged
    finally:
        clear_subspace_gauges()
    # (2) the polar frame Q = J·G^{-1/2}, differentiated, FD-matches (explicit form):
    u = T.var("u", Rt)
    gram = T.Compose(T.Dagger(J), (0,), (J,))
    Q = T.Compose(J, (0,), (T.Pinv(T.Sqrt(gram)),))
    ep = T.Eval(w, (0,), (T.Eval(Q, (0,), (u,)),))
    jac = diff(ep, x, deps={J.uid: DJ})
    seed = jac.vars[0]
    Jv, DJv = rng.standard_normal((3, 3)), rng.standard_normal((3, 3, 3))
    wv, uv, hv = rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(3)
    base = {f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"w_{w.uid}": wv, f"u_{u.uid}": uv}
    jh = _run(jac.body, {**base, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**base, f"x_{x.uid}": hv})
    fp = _run(ep, {**base, f"J_{J.uid}": Jv + 1e-6 * dJh})
    fm = _run(ep, {**base, f"J_{J.uid}": Jv - 1e-6 * dJh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / 2e-6).ravel(), atol=1e-4)


def test_polar_gauge_project_fd() -> None:
    # 1st-order PROJECT analog of the polar embed formula check.  project into Img(J) in the polar
    # frame Q = J·G^{-1/2} is Q⁺·z (= the polar coordinates of z's tangential part).  Differentiated
    # term→term (polar is closed-form ⇒ chain rule, NO opaque feed), FD-matches.  Built through the
    # EXPLICIT polar frame, since `represent` has no polar basis for the structural Project.
    rng = np.random.default_rng(12)
    Rt, Re = S.base("Rt", 3), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    z = T.var("z", Re)  # ambient value to project into Img(J)
    wc = T.var("wc", S.dual(Rt))  # covector in the polar (Rt) coordinate frame
    gram = T.Compose(T.Dagger(J), (0,), (J,))
    Q = T.Compose(J, (0,), (T.Pinv(T.Sqrt(gram)),))  # the explicit polar frame Q : Rt ⊸ Re
    ep = T.Eval(wc, (0,), (T.Eval(T.Pinv(Q), (0,), (z,)),))  # ⟨wc, Q⁺·z⟩
    jac = diff(ep, x, deps={J.uid: DJ})
    seed = jac.vars[0]
    Jv, DJv = rng.standard_normal((3, 3)), rng.standard_normal((3, 3, 3))
    wcv, zv, hv = rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(3)
    base = {f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"wc_{wc.uid}": wcv, f"z_{z.uid}": zv}
    jh = _run(jac.body, {**base, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**base, f"x_{x.uid}": hv})
    fp = _run(ep, {**base, f"J_{J.uid}": Jv + 1e-6 * dJh})
    fm = _run(ep, {**base, f"J_{J.uid}": Jv - 1e-6 * dJh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / 2e-6).ravel(), atol=1e-4)


def test_polar_gauge_second_order_embed_fd() -> None:
    # 2nd-order polar embed: diff(diff(⟨w, Q·v⟩)) through the EXPLICIT polar frame Q = J·G^{-1/2}.
    # Polar is closed-form, so the chain rule gives ALL orders with NO opaque feed (unlike the qr/SVD
    # towers).  FD-matches a central SECOND difference.
    rng = np.random.default_rng(13)
    Rt, Re = S.base("Rt", 3), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    D2J = T.var("D2J", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), Re))
    v = T.var("v", Rt)  # coordinates in the polar frame
    w = T.var("w", S.dual(Re))
    gram = T.Compose(T.Dagger(J), (0,), (J,))
    Q = T.Compose(J, (0,), (T.Pinv(T.Sqrt(gram)),))
    e = T.Eval(w, (0,), (T.Eval(Q, (0,), (v,)),))  # ⟨w, Q·v⟩
    Jv = rng.standard_normal((3, 3))
    DJv = _sym(rng.standard_normal((3, 3, 3)), 1)
    D2Jv = _sym(rng.standard_normal((3, 3, 3, 3)), 2)
    vv, wv = rng.standard_normal(3), rng.standard_normal(3)
    dval = rng.standard_normal(3)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv,
        f"v_{v.uid}": vv, f"w_{w.uid}": wv,
    }
    deps1 = {J.uid: DJ}
    deps2 = {J.uid: DJ, DJ.uid: D2J}
    H = diff(diff(e, x, deps1), x, deps2)
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    got = _run(body, {**feed, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    def perturb(s):
        dJ = s * np.einsum("akn,a->kn", DJv, dval) + 0.5 * s * s * np.einsum(
            "abkn,a,b->kn", D2Jv, dval, dval)
        return {**feed, f"J_{J.uid}": Jv + dJ}

    fd2 = _second_central(_run, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


def test_qr_gauge_opaque_matches_projectedbasis_fd() -> None:
    # The OPAQUE qr gauge: supply Q (the QR frame) and ∂Q/∂x as program-backed variables; `diff`
    # forms dQ = (∂Q/∂x)·h exactly like any marked variable.  Then diff(embed) realized in the
    # DEFAULT ProjectedBasis (QR) FD-matches — i.e. carrying the *matching* (opaque) QR connection
    # FIXES the basis the immersion gauge gets wrong (the 0.286-vs-1.900 case).  `represent` is
    # untouched: the opaque dQ is a fed variable.
    from grassmann.transform.diff import clear_subspace_gauges, set_subspace_gauge

    def qr_pos(M):
        q, r = np.linalg.qr(M)
        s = np.sign(np.diag(r))
        s[s == 0] = 1.0
        return q * s

    rng = np.random.default_rng(7)
    Rt, Re = S.base("Rt", 3), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    x = T.var("x", Rt)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    w = T.var("w", S.dual(Re))
    Qv = T.var("Qfr", S.oprod(S.dual(Rt), Re), props=G.bij)  # opaque QR frame Q : Rt ⊸ Re
    DQv = T.var("DQfr", S.oprod(S.dual(Rt), S.dual(Rt), Re))  # ∂Q/∂x : Rt ⊸ (Rt ⊸ Re)
    e = T.Eval(w, (0,), (T.embed(v, Re),))

    Jv = rng.standard_normal((3, 3))
    vv, Dvv = rng.standard_normal(3), rng.standard_normal((3, 3))
    DJv = rng.standard_normal((3, 3, 3))
    wv, hv = rng.standard_normal(3), rng.standard_normal(3)
    jm = Jv.T  # immersion columns in ambient
    q_amb = qr_pos(jm)  # (Re, Rt) ambient orthonormal frame; grassmann embed(v∘) = q_amb·v∘
    qval = q_amb.T  # grassmann map (Rt, Re):  Eval(Qv, v∘) = q_amb·v∘
    dqval = np.zeros((3, 3, 3))  # ∂Q/∂x[a] via FD of qr along DJ[a]; stored (a, Rt, Re)
    for a in range(3):
        dja = DJv[a].T
        dq_amb = (qr_pos(jm + 1e-6 * dja) - qr_pos(jm - 1e-6 * dja)) / 2e-6
        dqval[a] = dq_amb.T

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr", frame=Qv, frame_deriv=DQv)
    try:
        jac = diff(e, x, deps={v.uid: Dv, J.uid: DJ})
    finally:
        clear_subspace_gauges()
    seed = jac.vars[0]
    base = {
        f"J_{J.uid}": Jv, f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"DJ_{DJ.uid}": DJv,
        f"w_{w.uid}": wv, f"Qfr_{Qv.uid}": qval, f"DQfr_{DQv.uid}": dqval,
    }
    # sanity: the supplied frame Q matches grassmann's ProjectedBasis — Q⁺·embed(v) recovers v's
    # coordinates vv (so dQ·(Q⁺·embed v) is the matching QR connection).
    np.testing.assert_allclose(
        _run(T.Eval(T.Pinv(Qv), (0,), (T.embed(v, Re),)), base), vv, atol=1e-9,
    )
    jh = _run(jac.body, {**base, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**base, f"x_{x.uid}": hv})
    dvh = _run(T.Eval(Dv, (0,), (x,)), {**base, f"x_{x.uid}": hv})
    fp = _run(e, {**base, f"J_{J.uid}": Jv + 1e-6 * dJh, f"v_{v.uid}": vv + 1e-6 * dvh})
    fm = _run(e, {**base, f"J_{J.uid}": Jv - 1e-6 * dJh, f"v_{v.uid}": vv - 1e-6 * dvh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / 2e-6).ravel(), atol=1e-4)


# ---------------------------------------------------------------------------
# Gauge × realized-basis FD matrix.  The AD gauge frame ``Q`` must match the value frame
# `represent` lowers the moving subspace in: ``(immersion, ImageBasis)`` and the DEFAULT
# ``(qr, ProjectedBasis)``.  Each consistent pair must FD-match; the MISMATCHED pair must NOT.
# Both embed and project through a moving ``Img(J)`` are covered, in both LLam modes (suite-level).
# ---------------------------------------------------------------------------


def _moving_img_world(rng):
    """Shared moving-``Img(J)`` setup: the chart ``J`` (inj), its motion ``DJ``, and base values."""
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    Jv = rng.standard_normal((2, 3))
    DJv = rng.standard_normal((2, 2, 3))
    hv = rng.standard_normal(2)
    base = {f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv}
    return Rt, Re, J, imgJ, x, DJ, Jv, DJv, hv, base


@pytest.mark.parametrize("gauge", ["immersion", "qr"])
def test_gauge_matches_realized_basis_embed_fd(gauge) -> None:
    # embed through a moving Img(J): the gauge frame must match the value frame.  immersion ↔
    # ImageBasis (pinned via basis_env); qr ↔ the DEFAULT ProjectedBasis (opaque dQ fed via
    # qr_gauge_feed).  Both FD-match.
    from grassmann.core import basis as B
    from grassmann.lower.represent import default_basis, represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        qr_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(101)
    Rt, Re, J, imgJ, x, DJ, Jv, DJv, hv, base = _moving_img_world(rng)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    vv, Dvv, wv = rng.standard_normal(2), rng.standard_normal((2, 2)), rng.standard_normal(3)
    feed = {**base, f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"w_{w.uid}": wv}

    env, extra = None, {}
    if gauge == "immersion":
        ib = B.ImageBasis(J)  # the value frame matching the immersion gauge
        env = {v.uid: ib, Dv.uid: B.basis_prod(B.dual_basis(default_basis(Rt)), ib)}

    def runp(t, ff):
        kw = {"basis_env": env} if env else {}
        return np.asarray(represent(t, **kw).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, gauge)
    try:
        jd = diff(e, x, deps={v.uid: Dv, J.uid: DJ})
        if gauge == "qr":  # supply the opaque ProjectedBasis frame Q / ∂Q/∂x
            extra = qr_gauge_feed(J, DJ, feed)
    finally:
        clear_subspace_gauges()
    seed = jd.vars[0]
    allfeed = {**feed, **extra}
    jh = runp(jd.body, {**allfeed, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    dvh = runp(T.Eval(Dv, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    fp = runp(e, {**feed, f"J_{J.uid}": Jv + 1e-6 * dJh, f"v_{v.uid}": vv + 1e-6 * dvh})
    fm = runp(e, {**feed, f"J_{J.uid}": Jv - 1e-6 * dJh, f"v_{v.uid}": vv - 1e-6 * dvh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / 2e-6).ravel(), atol=1e-4)


@pytest.mark.parametrize("gauge", ["immersion", "qr"])
def test_gauge_matches_realized_basis_project_fd(gauge) -> None:
    # project through a moving Img(J) = di⁺ of the gauge frame.  Same matched-pair discipline as
    # embed: immersion ↔ ImageBasis, qr ↔ the DEFAULT ProjectedBasis.  Both FD-match.
    from grassmann.core import basis as B
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        qr_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(202)
    Rt, Re, J, imgJ, x, DJ, Jv, DJv, hv, base = _moving_img_world(rng)
    z = T.var("z", Re)  # ambient value to project into Img(J)
    Dz = T.var("Dz", S.oprod(S.dual(Rt), Re))
    wc = T.var("wc", S.dual(imgJ))  # covector in the subspace's coordinate frame
    e = T.Eval(wc, (0,), (T.project(z, imgJ),))
    zv, Dzv, wcv = rng.standard_normal(3), rng.standard_normal((2, 3)), rng.standard_normal(2)
    feed = {**base, f"z_{z.uid}": zv, f"Dz_{Dz.uid}": Dzv, f"wc_{wc.uid}": wcv}

    env, extra = None, {}
    if gauge == "immersion":
        env = {wc.uid: B.dual_basis(B.ImageBasis(J))}  # covector in the ImageBasis (matching frame)

    def runp(t, ff):
        kw = {"basis_env": env} if env else {}
        return np.asarray(represent(t, **kw).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, gauge)
    try:
        jd = diff(e, x, deps={z.uid: Dz, J.uid: DJ})
        if gauge == "qr":
            extra = qr_gauge_feed(J, DJ, feed)
    finally:
        clear_subspace_gauges()
    seed = jd.vars[0]
    allfeed = {**feed, **extra}
    jh = runp(jd.body, {**allfeed, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    dzh = _run(T.Eval(Dz, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    fp = runp(e, {**feed, f"J_{J.uid}": Jv + 1e-6 * dJh, f"z_{z.uid}": zv + 1e-6 * dzh})
    fm = runp(e, {**feed, f"J_{J.uid}": Jv - 1e-6 * dJh, f"z_{z.uid}": zv - 1e-6 * dzh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / 2e-6).ravel(), atol=1e-4)


def test_mismatched_gauge_is_off_by_the_qr_connection() -> None:
    # EXACT gap (positive assertion): the immersion gauge (Q = J) realized in the orthonormal DEFAULT
    # ProjectedBasis (Q = qr_pos(Jᵀ)) is wrong by EXACTLY the dropped QR connection.  For
    # embed(v) = Q_amb·v∘ with Q_amb = J_m·R⁻¹ (J_m = Jᵀ, the ambient immersion), the immersion gauge
    # captures ∂J_m·R⁻¹·v∘ but DROPS the frame-rotation term J_m·Ṙ⁻¹·v∘.  So
    #     fd − jvp  ==  wᵀ·J_m·Ṙ⁻¹·v∘ ,
    # with R⁻¹ = J⁺·Q (== the inverse of qr_pos's R) and Ṙ⁻¹ from the analytic QR differential.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        _qr_dQ,
        _qr_pos,
        clear_subspace_gauges,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(303)
    Rt, Re, J, imgJ, x, DJ, Jv, DJv, hv, base = _moving_img_world(rng)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    vv, Dvv, wv = rng.standard_normal(2), rng.standard_normal((2, 2)), rng.standard_normal(3)
    feed = {**base, f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"w_{w.uid}": wv}

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)  # DEFAULT ProjectedBasis value

    clear_subspace_gauges()
    set_subspace_gauge(J, "immersion")  # gauge ≠ the realized ProjectedBasis ⇒ off by the connection
    try:
        jd = diff(e, x, deps={v.uid: Dv, J.uid: DJ})
    finally:
        clear_subspace_gauges()
    seed = jd.vars[0]
    jh = runp(jd.body, {**feed, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": hv})  # ∂J·h, grassmann (k, n)
    dvh = runp(T.Eval(Dv, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    fp = runp(e, {**feed, f"J_{J.uid}": Jv + 1e-6 * dJh, f"v_{v.uid}": vv + 1e-6 * dvh})
    fm = runp(e, {**feed, f"J_{J.uid}": Jv - 1e-6 * dJh, f"v_{v.uid}": vv - 1e-6 * dvh})
    fd = (fp - fm) / 2e-6

    # The dropped QR connection wᵀ·J_m·Ṙ⁻¹·v∘, from qr_pos / the analytic QR differential.
    Jm = Jv.T                      # ambient immersion columns (n, k)
    dJm = dJh.T                    # ∂J_m·h, ambient (n, k)
    vcirc = vv                     # v's ProjectedBasis coords (v is a Var in imgJ ⇒ value IS coords)
    Q_amb, R = _qr_pos(Jm)
    Rinv = np.linalg.inv(R)        # R⁻¹ = J⁺·Q
    dQ = _qr_dQ(Q_amb, R, dJm)     # frame motion (analytic)
    dR = Q_amb.T @ dJm - (Q_amb.T @ dQ) @ R   # from J_m = Q_amb·R ⇒ dR = Qᵀ(dJ_m − dQ·R)
    dRinv = -Rinv @ dR @ Rinv      # Ṙ⁻¹ = −R⁻¹·Ṙ·R⁻¹
    connection = wv @ (Jm @ (dRinv @ vcirc))

    np.testing.assert_allclose(fd - jh, connection, atol=1e-4)


# ---------------------------------------------------------------------------
# Second-order FD: ∂²(embed/project) via diff(diff(·)) vs a central SECOND difference.
# The immersion gauge is closed-form ⇒ correct at all orders.  The opaque qr gauge's 2nd order needs
# ∂²Q/∂x² (a further opaque variable) — the single (Q, ∂Q/∂x) pair cannot supply it, so the opaque
# 2nd order is a documented SKIP (see test_qr_gauge_second_order_needs_d2Q below).
# ---------------------------------------------------------------------------


def _second_central(runp, e, perturb, eps=1e-4):
    """``(f(+d) − 2 f(0) + f(−d)) / ε²`` — central second difference of the lowered term."""
    f0 = runp(e, perturb(0.0))
    fp = runp(e, perturb(+eps))
    fm = runp(e, perturb(-eps))
    return (fp - 2.0 * f0 + fm) / (eps * eps)


def test_immersion_second_order_embed_fd() -> None:
    from grassmann.core import basis as B
    from grassmann.lower.represent import default_basis, represent
    from grassmann.transform.diff import clear_subspace_gauges, set_subspace_gauge

    rng = np.random.default_rng(404)
    Rt, Re, J, imgJ, x, DJ, Jv, DJv, hv, base = _moving_img_world(rng)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    vv, Dvv, wv = rng.standard_normal(2), rng.standard_normal((2, 2)), rng.standard_normal(3)
    dval = rng.standard_normal(2)
    feed = {**base, f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"w_{w.uid}": wv}
    ib = B.ImageBasis(J)
    env = {v.uid: ib, Dv.uid: B.basis_prod(B.dual_basis(default_basis(Rt)), ib)}

    def runp(t, ff):
        return np.asarray(represent(t, basis_env=env).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "immersion")
    try:
        deps = {v.uid: Dv, J.uid: DJ}
        H = diff(diff(e, x, deps), x, deps)  # ∂²: LLam((h2,), LLam((h,), d²body))
    finally:
        clear_subspace_gauges()
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    got = runp(body, {**feed, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    dJd = runp(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": dval})
    dvd = runp(T.Eval(Dv, (0,), (x,)), {**feed, f"x_{x.uid}": dval})

    def perturb(s):
        return {**feed, f"J_{J.uid}": Jv + s * dJd, f"v_{v.uid}": vv + s * dvd}

    fd2 = _second_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


def test_immersion_second_order_project_fd() -> None:
    from grassmann.core import basis as B
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import clear_subspace_gauges, set_subspace_gauge

    rng = np.random.default_rng(505)
    Rt, Re, J, imgJ, x, DJ, Jv, DJv, hv, base = _moving_img_world(rng)
    z = T.var("z", Re)
    Dz = T.var("Dz", S.oprod(S.dual(Rt), Re))
    wc = T.var("wc", S.dual(imgJ))
    e = T.Eval(wc, (0,), (T.project(z, imgJ),))
    zv, Dzv, wcv = rng.standard_normal(3), rng.standard_normal((2, 3)), rng.standard_normal(2)
    dval = rng.standard_normal(2)
    feed = {**base, f"z_{z.uid}": zv, f"Dz_{Dz.uid}": Dzv, f"wc_{wc.uid}": wcv}
    env = {wc.uid: B.dual_basis(B.ImageBasis(J))}

    def runp(t, ff):
        return np.asarray(represent(t, basis_env=env).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "immersion")
    try:
        deps = {z.uid: Dz, J.uid: DJ}
        H = diff(diff(e, x, deps), x, deps)
    finally:
        clear_subspace_gauges()
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    got = runp(body, {**feed, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    dJd = runp(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": dval})
    dzd = runp(T.Eval(Dz, (0,), (x,)), {**feed, f"x_{x.uid}": dval})

    def perturb(s):
        return {**feed, f"J_{J.uid}": Jv + s * dJd, f"z_{z.uid}": zv + s * dzd}

    fd2 = _second_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


def _third_central(runp, e, perturb, eps=1e-2):
    """Central THIRD difference with one **Richardson** step ⇒ ``O(ε⁴)`` error.

    The bare 5-point third difference ``D(h) = [f(2h) − 2f(h) + 2f(−h) − f(−2h)] / (2h³)`` has an
    ``O(ε²)`` truncation error; combining it at ``h`` and ``h/2`` as ``(4·D(h/2) − D(h))/3`` cancels
    that leading term, leaving ``O(ε⁴)`` — sharp enough to *verify* the 3rd-order AD rather than
    smoke-test it."""
    def D(h):
        fp2 = runp(e, perturb(+2.0 * h))
        fp1 = runp(e, perturb(+h))
        fm1 = runp(e, perturb(-h))
        fm2 = runp(e, perturb(-2.0 * h))
        return (fp2 - 2.0 * fp1 + 2.0 * fm1 - fm2) / (2.0 * h * h * h)

    return (4.0 * D(eps / 2.0) - D(eps)) / 3.0


def _sym(a: np.ndarray, naxes: int) -> np.ndarray:
    """Symmetrize ``a`` over its first ``naxes`` axes (a true high-order partial is symmetric)."""
    import itertools as _it

    rest = list(range(naxes, a.ndim))
    perms = list(_it.permutations(range(naxes)))
    return sum(a.transpose(list(p) + rest) for p in perms) / len(perms)


def _qr_world(rng, Rt_dim, Re_dim, order):
    """A moving-``Img(J)`` world with J's derivative tower ``[DJ, …, D^order J]`` (props.inj chart),
    plus base values (J & all D^mJ symmetric in the differentiation axes)."""
    Rt, Re = S.base("Rt", Rt_dim), S.base("Re", Re_dim)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    x = T.var("x", Rt)
    # D^m J : (Dual(Rt))^m [diff axes] ⊗ J's own space [Dual(Rt) ⊗ Re]  — m+1 Dual(Rt) factors then
    # the Re codomain; value shape (|X|,)^m + (k, n) = (Rt_dim,)^(m+1) + (Re_dim,).
    names = ["DJ"] + [f"D{m}J" for m in range(2, order + 1)]
    DJtower = [
        T.var(names[m - 1], S.oprod(*([S.dual(Rt)] * (m + 1)), Re)) for m in range(1, order + 1)
    ]
    Jv = rng.standard_normal((Rt_dim, Re_dim))
    # Symmetric only in the m DIFFERENTIATION axes (the leading m); the J-domain axis & Re are not.
    DJvtower = [
        _sym(rng.standard_normal((*([Rt_dim] * (m + 1)), Re_dim)), m) for m in range(1, order + 1)
    ]
    return Rt, Re, J, imgJ, x, DJtower, Jv, DJvtower


def test_qr_gauge_second_order_embed_fd() -> None:
    # Arbitrary-order QR AD: diff(diff(embed)) through a MOVING Img(J), value realized
    # in the DEFAULT ProjectedBasis (gauge qr), with the whole Q-tower fed via qr_gauge_feed — must
    # FD-match a central SECOND difference.  The second order needs ∂²Q, which the
    # self-differentiating tower supplies (the single (Q,∂Q/∂x) pair cannot).
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        qr_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(404)
    Rt, Re, J, imgJ, x, (DJ, D2J), Jv, (DJv, D2Jv) = _qr_world(rng, 2, 3, 2)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    D2v = T.var("D2v", S.oprod(S.dual(Rt), S.dual(Rt), imgJ))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    vv = rng.standard_normal(2)
    Dvv = rng.standard_normal((2, 2))
    D2vv = _sym(rng.standard_normal((2, 2, 2)), 2)
    wv = rng.standard_normal(3)
    dval = rng.standard_normal(2)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv,
        f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"D2v_{D2v.uid}": D2vv, f"w_{w.uid}": wv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)  # DEFAULT ProjectedBasis

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    # The nested diffs need each derivative var's OWN derivative marked (J→DJ→D²J, v→Dv→D²v).
    deps1 = {v.uid: Dv, J.uid: DJ}
    deps2 = {v.uid: Dv, J.uid: DJ, Dv.uid: D2v, DJ.uid: D2J}
    try:
        H = diff(diff(e, x, deps1), x, deps2)
        extra = qr_gauge_feed(J, [DJ, D2J], feed)
    finally:
        clear_subspace_gauges()
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    allfeed = {**feed, **extra}
    got = runp(body, {**allfeed, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    def perturb(s):
        dJ = s * np.einsum("akn,a->kn", DJv, dval) + 0.5 * s * s * np.einsum(
            "abkn,a,b->kn", D2Jv, dval, dval)
        dv = s * np.einsum("ak,a->k", Dvv, dval) + 0.5 * s * s * np.einsum(
            "abk,a,b->k", D2vv, dval, dval)
        return {**feed, f"J_{J.uid}": Jv + dJ, f"v_{v.uid}": vv + dv}

    fd2 = _second_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


def test_qr_gauge_second_order_project_fd() -> None:
    # Same arbitrary-order QR tower for PROJECT (di⁺ of the moving frame), value in the DEFAULT
    # ProjectedBasis, FD-matched to a central SECOND difference.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        qr_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(505)
    Rt, Re, J, imgJ, x, (DJ, D2J), Jv, (DJv, D2Jv) = _qr_world(rng, 2, 3, 2)
    z = T.var("z", Re)
    Dz = T.var("Dz", S.oprod(S.dual(Rt), Re))
    D2z = T.var("D2z", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    wc = T.var("wc", S.dual(imgJ))
    e = T.Eval(wc, (0,), (T.project(z, imgJ),))
    zv = rng.standard_normal(3)
    Dzv = rng.standard_normal((2, 3))
    D2zv = _sym(rng.standard_normal((2, 2, 3)), 2)
    wcv = rng.standard_normal(2)
    dval = rng.standard_normal(2)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv,
        f"z_{z.uid}": zv, f"Dz_{Dz.uid}": Dzv, f"D2z_{D2z.uid}": D2zv, f"wc_{wc.uid}": wcv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    deps1 = {z.uid: Dz, J.uid: DJ}
    deps2 = {z.uid: Dz, J.uid: DJ, Dz.uid: D2z, DJ.uid: D2J}
    try:
        H = diff(diff(e, x, deps1), x, deps2)
        extra = qr_gauge_feed(J, [DJ, D2J], feed)
    finally:
        clear_subspace_gauges()
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    allfeed = {**feed, **extra}
    got = runp(body, {**allfeed, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    def perturb(s):
        dJ = s * np.einsum("akn,a->kn", DJv, dval) + 0.5 * s * s * np.einsum(
            "abkn,a,b->kn", D2Jv, dval, dval)
        dz = s * np.einsum("ak,a->k", Dzv, dval) + 0.5 * s * s * np.einsum(
            "abk,a,b->k", D2zv, dval, dval)
        return {**feed, f"J_{J.uid}": Jv + dJ, f"z_{z.uid}": zv + dz}

    fd2 = _second_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


def test_qr_gauge_third_order_embed_fd() -> None:
    # THIRD-order QR AD — proves the tower is truly arbitrary-order.  Small subspace (Rt=Re=2) to
    # keep the 3rd-derivative tensors cheap.  diff³(embed) in the DEFAULT ProjectedBasis vs a central
    # THIRD difference (the dominant error is the FD truncation, not the AD tower).
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        qr_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(606)
    Rt, Re, J, imgJ, x, (DJ, D2J, D3J), Jv, (DJv, D2Jv, D3Jv) = _qr_world(rng, 2, 2, 3)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    D2v = T.var("D2v", S.oprod(S.dual(Rt), S.dual(Rt), imgJ))
    D3v = T.var("D3v", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), imgJ))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    vv = rng.standard_normal(2)
    Dvv = rng.standard_normal((2, 2))
    D2vv = _sym(rng.standard_normal((2, 2, 2)), 2)
    D3vv = _sym(rng.standard_normal((2, 2, 2, 2)), 3)
    wv = rng.standard_normal(2)
    dval = rng.standard_normal(2)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv, f"D3J_{D3J.uid}": D3Jv,
        f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"D2v_{D2v.uid}": D2vv, f"D3v_{D3v.uid}": D3vv,
        f"w_{w.uid}": wv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    deps1 = {v.uid: Dv, J.uid: DJ}
    deps2 = {**deps1, Dv.uid: D2v, DJ.uid: D2J}
    deps3 = {**deps2, D2v.uid: D3v, D2J.uid: D3J}
    try:
        H = diff(diff(diff(e, x, deps1), x, deps2), x, deps3)
        extra = qr_gauge_feed(J, [DJ, D2J, D3J], feed)
    finally:
        clear_subspace_gauges()
    h3, h2, h1 = H.vars[0], H.body.vars[0], H.body.body.vars[0]
    body = H.body.body.body
    allfeed = {**feed, **extra}
    got = runp(body, {
        **allfeed,
        f"{h3.name}_{h3.uid}": dval, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval,
    })

    def perturb(s):
        dJ = (
            s * np.einsum("akn,a->kn", DJv, dval)
            + 0.5 * s * s * np.einsum("abkn,a,b->kn", D2Jv, dval, dval)
            + (s ** 3 / 6.0) * np.einsum("abckn,a,b,c->kn", D3Jv, dval, dval, dval)
        )
        dv = (
            s * np.einsum("ak,a->k", Dvv, dval)
            + 0.5 * s * s * np.einsum("abk,a,b->k", D2vv, dval, dval)
            + (s ** 3 / 6.0) * np.einsum("abck,a,b,c->k", D3vv, dval, dval, dval)
        )
        return {**feed, f"J_{J.uid}": Jv + dJ, f"v_{v.uid}": vv + dv}

    fd3 = _third_central(runp, e, perturb)
    # Richardson-extrapolated on both sides (the fed ∂³Q tower and this reference) ⇒ residual ~8e-6,
    # so this is a SHARP 3rd-order check (was a loose ~4e-2 smoke test before Richardson).
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd3), rtol=1e-4, atol=1e-4)


def test_gauss_decomposition_of_embed() -> None:
    # embed IS di (the pushforward of the embedding i).  Differentiating it term→term gives the
    # second covariant derivative ∇di, which obeys the Gauss formula
    #     ∇di(X,Y) = di(∇̄_X Y − ∇^M_X Y) + II(X,Y),   ḡ = i*h = di*h,
    # the SAME object as our covariant connection (tangential) + second fundamental form (normal).
    # Here ∇^M = 0 (flat parameter), h = I (standard ambient).  Every piece is checked against FD /
    # geometry, all from J and Pinv(J) — no orthonormal basis is ever differentiated.
    rng = np.random.default_rng(3)
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    Y = T.var("Y", Rt)
    DY = T.var("DY", S.oprod(S.dual(Rt), Rt))
    Jp = T.Pinv(J)
    di_Y = T.Eval(J, (0,), (Y,))  # di(Y) = embed(Y) = J·Y, linear in J (iso/ImageBasis frame)
    Jv = rng.standard_normal((2, 3))
    D = rng.standard_normal((2, 2, 3))
    DJv = 0.5 * (D + D.transpose(1, 0, 2))  # ∂J symmetric in dir↔tangent ⇒ a true Hessian ∂²i
    Yv, DYv, hv = rng.standard_normal(2), rng.standard_normal((2, 2)), rng.standard_normal(2)
    feed = {f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"Y_{Y.uid}": Yv, f"DY_{DY.uid}": DYv}

    # (1) the AD second derivative ∇di(h, Y) == central FD (J and Y both moving)
    Jd = diff(di_Y, x, deps={J.uid: DJ, Y.uid: DY})
    seed = Jd.vars[0]
    nabla = _run(Jd.body, {**feed, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    dYh = _run(T.Eval(DY, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    fp = _run(di_Y, {**feed, f"J_{J.uid}": Jv + 1e-6 * dJh, f"Y_{Y.uid}": Yv + 1e-6 * dYh})
    fm = _run(di_Y, {**feed, f"J_{J.uid}": Jv - 1e-6 * dJh, f"Y_{Y.uid}": Yv - 1e-6 * dYh})
    np.testing.assert_allclose(nabla, (fp - fm) / 2e-6, atol=1e-4)

    # P = di∘di⁺ = embed∘project, the orthogonal projector onto the tangent space Img(J)
    zc = T.var("zc", Re)
    p_term = T.Eval(J, (0,), (T.Eval(Jp, (0,), (zc,)),))

    def P_of(zval):
        return _run(p_term, {**feed, f"zc_{zc.uid}": zval})

    # (2) tangential part == di(∇̄_h Y), the induced connection: ∇̄_h Y = DY·h + Pinv(J)·((∂_h J)·Y)
    gamma_Y = T.Eval(Jp, (0,), (T.Eval(T.Eval(DJ, (0,), (x,)), (0,), (Y,)),))  # Γ(h)·Y
    barnabla = T.Sum((T.Eval(DY, (0,), (x,)), gamma_Y))                        # ∇̄_h Y
    di_barnabla = _run(T.Eval(J, (0,), (barnabla,)), {**feed, f"x_{x.uid}": hv})
    np.testing.assert_allclose(P_of(nabla), di_barnabla, atol=1e-5)

    # (3) normal part == II : purely normal, and symmetric (the second fundamental form)
    np.testing.assert_allclose(P_of(nabla - di_barnabla), 0.0, atol=1e-6)  # P·II = 0

    def II_of(direction, vec):  # (∂_direction J)·vec, normal part — II(direction, vec)
        jd = diff(T.Eval(J, (0,), (Y,)), x, deps={J.uid: DJ})  # only J marked ⇒ pure (∂J)·Y
        s = jd.vars[0]
        nb = _run(jd.body, {**feed, f"Y_{Y.uid}": vec, f"{s.name}_{s.uid}": direction})
        return nb - P_of(nb)

    np.testing.assert_allclose(II_of(hv, Yv), II_of(Yv, hv), atol=1e-6)  # II symmetric (∂²i is a Hessian)


def test_moving_subspace_ad_term_to_term() -> None:
    # Term→term AD through a moving subspace, "everywhere it appears", all built from the defining
    # map J and Pinv(J) (no orthonormal basis differentiated): the FEM tangent (immersion), the
    # cotangent metric (JᵀJ)⁻¹ that gradients use, and the projectors that mediate Img/CoImg/Ker/
    # CoKer.  Every case is FD-verified against a moving J.
    rng = np.random.default_rng(9)
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    x = T.var("x", Rt)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    w1, w2 = T.var("w1", Rt), T.var("w2", Rt)
    z, z2 = T.var("z", Re), T.var("z2", S.dual(Re))
    Jv = rng.standard_normal((2, 3))
    base = {f"w1_{w1.uid}": rng.standard_normal(2), f"w2_{w2.uid}": rng.standard_normal(2),
            f"z_{z.uid}": rng.standard_normal(3), f"z2_{z2.uid}": rng.standard_normal(3)}
    DJv, hv = rng.standard_normal((2, 2, 3)), rng.standard_normal(2)
    Jp = T.Pinv(J)
    gram = T.Compose(T.Dagger(J), (0,), (J,))  # JᵀJ : the immersed (tangent) metric
    cases = {
        # immersed tangent metric ⟨J w1, J w2⟩ (the FEM stiffness/mass integrand structure)
        "immersed_metric": T.Eval(T.inner(Re), (0, 1), (T.Eval(J, (0,), (w1,)), T.Eval(J, (0,), (w2,)))),
        # cotangent metric (JᵀJ)⁻¹ — what gradient/cotangent values pair with
        "cotangent_metric": T.Eval(T.inner(Rt), (0, 1), (T.Eval(T.Pinv(gram), (0,), (w1,)), w2)),
        # ambient projector onto Img(J): JJ⁺  (the Img / CoImg realization)
        "img_projector": T.Eval(z2, (0,), (T.Eval(J, (0,), (T.Eval(Jp, (0,), (z,)),)),)),
        # complementary projector I − JJ⁺ onto Img(J)^⊥ (the CoKer side)
        "coker_projector": T.Eval(z2, (0,),
                                  (T.Sum((z, T.Scale(-1.0, T.Eval(J, (0,), (T.Eval(Jp, (0,), (z,)),))))),)),
    }
    for name, e in cases.items():
        _fd_marked_map(e, x, J, DJ, base, Jv, DJv, hv)


# ---------------------------------------------------------------------------
# Covariant connection (tangent space with a non-constant metric).
# ---------------------------------------------------------------------------


def test_flat_metric_recovers_total_derivative() -> None:
    # A constant (unmarked) metric ⇒ not a covariant section ⇒ the plain total derivative.
    from grassmann.transform.diff import _covariant_section, _DCtx

    V = S.base("V", 2)
    x = T.var("x", V)
    g = T.var("g", S.oprod(S.dual(V), S.dual(V)))
    e = T.Remetric(T.var("sec", V), g)
    ctx = _DCtx(x=x, h=T.var("dx", V), deps={})  # g absent ⇒ constant
    assert _covariant_section(e, ctx) is None


def test_covariant_derivative_matches_christoffel() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    g = T.var("g", S.oprod(S.dual(V), S.dual(V)))
    sec = T.var("sec", V)
    Dsec = T.var("Dsec", S.oprod(S.dual(V), V))
    Dg = T.var("Dg", S.oprod(S.dual(V), S.dual(V), S.dual(V)))
    e = T.Remetric(sec, g)  # the section in the tangent space WithInner(V, g)
    J = diff(e, x, deps={sec.uid: Dsec, g.uid: Dg})

    g0 = np.array([[2.0, 0.3], [0.3, 1.5]])
    Dgv = np.random.randn(2, 2, 2)
    Dgv = 0.5 * (Dgv + Dgv.transpose(0, 2, 1))  # ∂_a g_bc symmetric in (b, c)
    secv, Dsecv = np.random.randn(2), np.random.randn(2, 2)
    feed = {f"g_{g.uid}": g0, f"sec_{sec.uid}": secv, f"Dg_{Dg.uid}": Dgv, f"Dsec_{Dsec.uid}": Dsecv}
    Jv = _run(J, feed)

    ginv = np.linalg.inv(g0)
    gam_lower = np.zeros((2, 2, 2))  # Γ_{kij} = ½(∂_i g_jk + ∂_j g_ik − ∂_k g_ij)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gam_lower[k, i, j] = 0.5 * (Dgv[i, j, k] + Dgv[j, i, k] - Dgv[k, i, j])
    hand = Dsecv.T + np.einsum("kl,lij,j->ki", ginv, gam_lower, secv)
    np.testing.assert_allclose(Jv, hand.T, atol=1e-6)  # Jv now (dom,cod)


def test_cotangent_connection_matches_christoffel() -> None:
    # ∇_h ω = dω − Γ(h)ᵀ·ω for a 1-form section.
    M = S.base("M", 2)
    x = T.var("x", M)
    g = T.var("g", S.oprod(S.dual(M), S.dual(M)))
    om = T.var("om", S.dual(M))
    Dom = T.var("Dom", S.oprod(S.dual(M), S.dual(M)))  # dω/dx : M → M*
    Dg = T.var("Dg", S.oprod(S.dual(M), S.dual(M), S.dual(M)))
    J = diff(T.Remetric(om, g), x, deps={om.uid: Dom, g.uid: Dg})

    g0 = np.array([[2.0, 0.3], [0.3, 1.5]])
    ginv = np.linalg.inv(g0)
    Dgv = np.random.randn(2, 2, 2)
    Dgv = 0.5 * (Dgv + Dgv.transpose(0, 2, 1))
    omv, Domv = np.random.randn(2), np.random.randn(2, 2)
    feed = {f"g_{g.uid}": g0, f"om_{om.uid}": omv, f"Dg_{Dg.uid}": Dgv, f"Dom_{Dom.uid}": Domv}
    Jv = _run(J, feed)

    gam_low = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gam_low[k, i, j] = 0.5 * (Dgv[i, j, k] + Dgv[j, i, k] - Dgv[k, i, j])
    gam2 = np.einsum("kl,lmi->kmi", ginv, gam_low)  # Γ^k_{mi}
    hand = Domv.T - np.einsum("kmi,k->mi", gam2, omv)
    np.testing.assert_allclose(Jv, hand.T, atol=1e-6)  # Jv now (dom,cod)


def test_connection_map_applied_to_vector() -> None:
    # The (1,1) connection map Γ(h):M⊸M (built via Permute + the M⊸M* decode + Eval
    # f-free-labels) applied to a vector matches the Christoffel contraction.
    from grassmann.transform.diff import _connection_map

    M = S.base("M", 2)
    h = T.var("dx", M)
    g = T.var("g", S.oprod(S.dual(M), S.dual(M)))
    Dg = T.var("Dg", S.oprod(S.dual(M), S.dual(M), S.dual(M)))
    v = T.var("v", M)
    conn = _connection_map(M, g, Dg, h)  # Γ(h) : M ⊸ M (seed-carrying, bare)
    applied = T.Eval(conn, (0,), (v,))

    g0 = np.array([[2.0, 0.3], [0.3, 1.5]])
    ginv = np.linalg.inv(g0)
    Dgv = np.random.randn(2, 2, 2)
    Dgv = 0.5 * (Dgv + Dgv.transpose(0, 2, 1))
    hv, vv = np.random.randn(2), np.random.randn(2)
    feed = {f"g_{g.uid}": g0, f"Dg_{Dg.uid}": Dgv, f"dx_{h.uid}": hv, f"v_{v.uid}": vv}
    out = _run(applied, feed)
    gl = np.zeros(2)
    for a in range(2):
        for i in range(2):
            for j in range(2):
                gl[a] += 0.5 * (Dgv[i, j, a] + Dgv[j, i, a] - Dgv[a, i, j]) * hv[i] * vv[j]
    np.testing.assert_allclose(out, ginv @ gl, atol=1e-6)


def test_covariant_identity_tensor_is_parallel() -> None:
    # ∇I = 0 : the identity (1,1) tensor is parallel under any metric connection.
    M = S.base("M", 2)
    x = T.var("x", M)
    g = T.var("g", S.oprod(S.dual(M), S.dual(M)))
    Dg = T.var("Dg", S.oprod(S.dual(M), S.dual(M), S.dual(M)))
    J = diff(T.Remetric(T.Identity(M), g), x, deps={g.uid: Dg})
    g0 = np.array([[2.0, 0.3], [0.3, 1.5]])
    Dgv = np.random.randn(2, 2, 2)
    Dgv = 0.5 * (Dgv + Dgv.transpose(0, 2, 1))
    Jv = _run(J, {f"g_{g.uid}": g0, f"Dg_{Dg.uid}": Dgv})
    np.testing.assert_allclose(Jv, 0.0, atol=1e-10)


def test_covariant_endomorphism_tensor() -> None:
    # ∇_h T = dT + Γ(h)T − TΓ(h) for a (1,1) tensor field T : M ⊸ M.
    M = S.base("M", 2)
    x = T.var("x", M)
    g = T.var("g", S.oprod(S.dual(M), S.dual(M)))
    Dg = T.var("Dg", S.oprod(S.dual(M), S.dual(M), S.dual(M)))
    Tt = T.var("Tt", S.oprod(S.dual(M), M))
    DT = T.var("DT", S.oprod(S.dual(M), S.dual(M), M))  # dT/dx, flat
    J = diff(T.Remetric(Tt, g), x, deps={Tt.uid: DT, g.uid: Dg})

    g0 = np.array([[2.0, 0.3], [0.3, 1.5]])
    ginv = np.linalg.inv(g0)
    Dgv = np.random.randn(2, 2, 2)
    Dgv = 0.5 * (Dgv + Dgv.transpose(0, 2, 1))
    Tval, DTv = np.random.randn(2, 2), np.random.randn(2, 2, 2)
    feed = {f"g_{g.uid}": g0, f"Dg_{Dg.uid}": Dgv, f"Tt_{Tt.uid}": Tval, f"DT_{DT.uid}": DTv}
    Jv = _run(J, feed)  # stored (X, dom, cod) — the binder X leads

    def conn_op(i):  # Γ(e_i) as the operator matrix [k, m]
        out = np.zeros((2, 2))
        for k in range(2):
            for m in range(2):
                for a in range(2):
                    out[k, m] += ginv[k, a] * 0.5 * (Dgv[m, i, a] + Dgv[i, m, a] - Dgv[a, i, m])
        return out

    Top = Tval.T  # T as an operator (stored dom,cod ⇒ operator is the transpose)
    for i in range(2):
        Ci = conn_op(i)
        corr = Ci @ Top - Top @ Ci  # ΓT − TΓ, operator [k, l]
        dT_op = DTv[i, :, :].T  # the i-th x-slice of dT, as an operator
        # Jv now (X, dom, cod): slice the leading X axis, then transpose the (dom,cod)
        # block to operator form to match dT_op + corr.
        np.testing.assert_allclose(Jv[i, :, :].T, dT_op + corr, atol=1e-6)


def test_metric_is_compatible_nabla_g_zero() -> None:
    # ∇g = 0 — metric compatibility, the defining property of the Levi-Civita connection.
    # The metric is differentiated as a (0,2) section of its own connection.
    M = S.base("M", 2)
    x = T.var("x", M)
    g = T.var("g", S.oprod(S.dual(M), S.dual(M)))
    Dg = T.var("Dg", S.oprod(S.dual(M), S.dual(M), S.dual(M)))
    J = diff(T.Remetric(g, g), x, deps={g.uid: Dg})
    g0 = np.array([[2.0, 0.3], [0.3, 1.5]])
    Dgv = np.random.randn(2, 2, 2)
    Dgv = 0.5 * (Dgv + Dgv.transpose(0, 2, 1))  # ∂_a g_bc symmetric in (b, c)
    Jv = _run(J, {f"g_{g.uid}": g0, f"Dg_{Dg.uid}": Dgv})
    np.testing.assert_allclose(Jv, 0.0, atol=1e-10)


def test_metric_pairing_is_parallel() -> None:
    # ⟨∇ω, v⟩ + ⟨ω, ∇v⟩ = d⟨ω, v⟩ — the connection's effect cancels in the pairing.
    M = S.base("M", 2)
    x = T.var("x", M)
    g = T.var("g", S.oprod(S.dual(M), S.dual(M)))
    Dg = T.var("Dg", S.oprod(S.dual(M), S.dual(M), S.dual(M)))
    sec = T.var("sec", M)
    Dsec = T.var("Dsec", S.oprod(S.dual(M), M))
    om = T.var("om", S.dual(M))
    Dom = T.var("Dom", S.oprod(S.dual(M), S.dual(M)))
    Jt = diff(T.Remetric(sec, g), x, deps={sec.uid: Dsec, g.uid: Dg})
    Jc = diff(T.Remetric(om, g), x, deps={om.uid: Dom, g.uid: Dg})

    g0 = np.array([[2.0, 0.3], [0.3, 1.5]])
    Dgv = np.random.randn(2, 2, 2)
    Dgv = 0.5 * (Dgv + Dgv.transpose(0, 2, 1))
    secv, Dsecv = np.random.randn(2), np.random.randn(2, 2)
    omv, Domv = np.random.randn(2), np.random.randn(2, 2)
    feed = {
        f"g_{g.uid}": g0, f"Dg_{Dg.uid}": Dgv,
        f"sec_{sec.uid}": secv, f"Dsec_{Dsec.uid}": Dsecv,
        f"om_{om.uid}": omv, f"Dom_{Dom.uid}": Domv,
    }
    Jtv = _run(Jt, feed)  # (∇v)^k_i
    Jcv = _run(Jc, feed)  # (∇ω)_m_i
    # Jacobians now stored (dom, cod): contract the COD axis (axis 1) with the pairing.
    lhs = Jcv @ secv + Jtv @ omv  # ⟨∇ω,v⟩ + ⟨ω,∇v⟩, indexed by direction i
    rhs = Domv @ secv + Dsecv @ omv  # d⟨ω,v⟩ = (∂ω)v + ω(∂v)
    np.testing.assert_allclose(lhs, rhs, atol=1e-6)


# ---------------------------------------------------------------------------
# Frame / chart-Jacobian connection: ∇ from a moving frame J(x), the FEM case.
# The induced (pullback) metric is g = JᵀJ; the connection Γ(h) = J⁺·(∂J·h) needs no rank-3
# metric gradient.  Cross-checked against the trusted marked-(g, Dg) metric connection.
# ---------------------------------------------------------------------------


def _jtj_and_grad(Jv: np.ndarray, DJv: np.ndarray):
    """The pullback metric ``g = JᵀJ`` and its gradient ``Dg = ∂(JᵀJ)`` for a chart Jacobian
    ``J`` (stored ``[tangent i, ambient p]``) with derivative ``DJv[a, i, p] = ∂ₐ J[i, p]``."""
    g = Jv @ Jv.T  # g_ij = Σ_p J[i,p] J[j,p]
    Dg = np.einsum("aip,jp->aij", DJv, Jv) + np.einsum("ip,ajp->aij", Jv, DJv)
    return g, Dg


def _frame_vs_metric(sec, ssp, M, A):
    """Build the frame section ``Remetric(sec, JᵀJ)`` and the equivalent marked-metric section
    ``Remetric(sec, g)``; return ``(diff_frame, diff_metric, vars)`` ready to feed consistently."""
    x = T.var("x", M)
    DM = S.dual(M)
    J = T.var("J", S.oprod(DM, A), props=G.bij if M == A else G.inj)
    DJ = T.var("DJ", S.oprod(DM, DM, A))
    ssp_factors = ssp.spaces if isinstance(ssp, S.OProd) else (ssp,)
    Dsec = T.var("Dsec", S.oprod(DM, *ssp_factors))  # X ⊸ ssp, FLAT (not nested-oprod)
    gV = T.var("g", S.oprod(DM, DM))
    DgV = T.var("Dg", S.oprod(DM, DM, DM))
    frame = diff(T.Remetric(sec, T.Compose(T.Dagger(J), (0,), (J,))), x,
                 deps={sec.uid: Dsec, J.uid: DJ})
    metric = diff(T.Remetric(sec, gV), x, deps={sec.uid: Dsec, gV.uid: DgV})
    return frame, metric, (x, J, DJ, Dsec, gV, DgV)


@pytest.mark.parametrize("ambient_dim", [2, 3])  # square chart (=change of coords) and codim-1 immersion
@pytest.mark.parametrize("kind", ["vector", "covector", "endo"])
def test_frame_connection_matches_metric_connection(kind, ambient_dim) -> None:
    rng = np.random.default_rng(20 + ambient_dim + len(kind))
    M = S.base("M", 2)
    A = S.base("A", ambient_dim)
    DM = S.dual(M)
    if kind == "vector":
        sec, ssp = T.var("sec", M), M
        secv = rng.standard_normal(2)
    elif kind == "covector":
        sec, ssp = T.var("sec", DM), DM
        secv = rng.standard_normal(2)
    else:  # (1,1) endomorphism T : M ⊸ M
        ssp = S.oprod(DM, M)
        sec = T.var("sec", ssp)
        secv = rng.standard_normal((2, 2))

    frame, metric, (x, J, DJ, Dsec, gV, DgV) = _frame_vs_metric(sec, ssp, M, A)
    Jv = rng.standard_normal((2, ambient_dim))
    D = rng.standard_normal((2, 2, ambient_dim))
    DJv = 0.5 * (D + D.transpose(1, 0, 2))  # a valid chart Hessian (∂J symmetric in dir↔tangent)
    g0, Dg0 = _jtj_and_grad(Jv, DJv)
    Dsecv = rng.standard_normal((2, *np.shape(secv)))  # X ⊸ ssp, flat axes
    hv = rng.standard_normal(2)

    base = {f"sec_{sec.uid}": secv, f"Dsec_{Dsec.uid}": Dsecv}
    sf = _seed_key(frame)
    sm = _seed_key(metric)
    got_frame = _run(frame.body, {**base, f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, sf[0]: hv})
    got_metric = _run(metric.body, {**base, f"g_{gV.uid}": g0, f"Dg_{DgV.uid}": Dg0, sm[0]: hv})
    np.testing.assert_allclose(got_frame, got_metric, atol=1e-6)


def _seed_key(J):
    s = J.vars[0]
    return (f"{s.name}_{s.uid}",)


def test_constant_chart_is_flat() -> None:
    # J not marked (x-independent chart) ⇒ Remetric(sec, JᵀJ) is flat ⇒ plain total derivative.
    from grassmann.transform.diff import _covariant_section, _DCtx

    M = S.base("M", 2)
    A = S.base("A", 3)
    J = T.var("J", S.oprod(S.dual(M), A), props=G.inj)
    sec = T.var("sec", M)
    e = T.Remetric(sec, T.Compose(T.Dagger(J), (0,), (J,)))
    ctx = _DCtx(x=T.var("x", M), h=T.var("dx", M), deps={})  # J absent ⇒ constant chart
    assert _covariant_section(e, ctx) is None


# ---------------------------------------------------------------------------
# AD through a term-dependent immersion, PARAMETER frame (the frame-free, term→term way).
# A tangent vector w : dom(J) immersed via Eval(J, w) differentiates by the ordinary product rule
# even when the chart J is x-dependent — no orthonormal ImageBasis, no triangular gauge term.  The
# covariant / normal split falls out via the gauge-invariant projector P = embed∘project, and the
# covariant part equals Eval(J, dw + Γ·w) with Γ = Pinv(J)∘∂J (the task-3 frame connection).
# ---------------------------------------------------------------------------


def test_parameter_frame_immersion_ad_and_split() -> None:
    rng = np.random.default_rng(7)
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    X = Rt
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    x, w, h = T.var("x", X), T.var("w", Rt), T.var("h", X)
    Dw = T.var("Dw", S.oprod(S.dual(X), Rt))
    DJ = T.var("DJ", S.oprod(S.dual(X), S.dual(Rt), Re))
    Jv = rng.standard_normal((2, 3))
    wv, Dwv = rng.standard_normal(2), rng.standard_normal((2, 2))
    DJv, hv = rng.standard_normal((2, 2, 3)), rng.standard_normal(2)
    feed = {f"J_{J.uid}": Jv, f"w_{w.uid}": wv, f"Dw_{Dw.uid}": Dwv, f"DJ_{DJ.uid}": DJv, f"h_{h.uid}": hv}

    # (1) a scalar pairing ⟨wamb, J·w⟩ with BOTH w and J marked — differentiates term→term vs FD.
    wamb = T.var("wamb", S.dual(Re))
    e = T.Eval(wamb, (0,), (T.Eval(J, (0,), (w,)),))
    wav = rng.standard_normal(3)
    Jd = diff(e, x, deps={w.uid: Dw, J.uid: DJ})
    seed = Jd.vars[0]
    feed_s = {**feed, f"wamb_{wamb.uid}": wav}
    jh = _run(Jd.body, {**feed_s, f"{seed.name}_{seed.uid}": hv})
    dwh = _run(T.Eval(Dw, (0,), (h,)), feed)
    dJh = _run(T.Eval(DJ, (0,), (h,)), feed)
    fp = _run(e, {**feed_s, f"J_{J.uid}": Jv + 1e-6 * dJh, f"w_{w.uid}": wv + 1e-6 * dwh})
    fm = _run(e, {**feed_s, f"J_{J.uid}": Jv - 1e-6 * dJh, f"w_{w.uid}": wv - 1e-6 * dwh})
    np.testing.assert_allclose(np.asarray(jh).ravel(), ((fp - fm) / 2e-6).ravel(), atol=1e-4)

    # (2) covariant / normal split of the immersion's directional derivative.
    full = T.Sum((T.Eval(T.Eval(DJ, (0,), (h,)), (0,), (w,)), T.Eval(J, (0,), (T.Eval(Dw, (0,), (h,)),))))
    fullv = _run(full, feed)
    fc = T.var("fc", Re)
    cov = _run(T.embed(T.project(fc, imgJ), Re), {**feed, f"fc_{fc.uid}": fullv})  # P = embed∘project
    # covariant == Eval(J, dw + Γ·w) with Γ the task-3 frame connection
    Gam = T.Compose(T.Pinv(J), (0,), (T.Eval(DJ, (0,), (h,)),))
    cov_conn = _run(T.Eval(J, (0,), (T.Sum((T.Eval(Dw, (0,), (h,)), T.Eval(Gam, (0,), (w,)))),)), feed)
    np.testing.assert_allclose(cov, cov_conn, atol=1e-5)             # covariant = frame connection
    np.testing.assert_allclose(cov + (fullv - cov), fullv, atol=1e-9)  # full = covariant + normal


# ---------------------------------------------------------------------------
# Failure modes.
# ---------------------------------------------------------------------------


def test_higher_order_is_rejected() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    # a Frame (basis binder) on the path ⇒ DiffError.
    with pytest.raises(DiffError):
        diff(T.NEval(T.var("f", V), (0,), (x,)), x)


def test_project_through_moving_img_differentiates() -> None:
    # project IS di⁺.  Differentiating project(z, Img(f)) w.r.t. the chart f (which defines the
    # space) COMPUTES the realization derivative ∂(f⁺)·z (the un-gated pseudoinverse derivative) —
    # a genuine Jacobian map, not a Zero or a raise.
    W, V = S.base("W", 2), S.base("V", 4)
    f = T.var("f", S.oprod(S.dual(W), V), props=G.inj)
    z = T.var("z", V)
    J = diff(T.Project(z, S.img(f)), f)  # d/df of the projection w.r.t. its defining map
    assert not isinstance(J, T.Zero)  # the ∂f contribution is present, not silently dropped
    assert isinstance(J, T.LLam)      # a genuine Jacobian map


def test_section_in_img_space_differentiates_with_constant_chart() -> None:
    # A section LIVING in a term-dependent space Img(J) (the FEM tangent model): with the chart J
    # held constant (the common per-cell case) the total derivative is correct — diff supports
    # values in term-dependent spaces.
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    x = T.var("x", Rt)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))  # ⟨w, embed_J(v)⟩
    Jv = diff(e, x, deps={v.uid: Dv})  # J constant ⇒ embedding constant ⇒ correct
    seed = Jv.vars[0]
    rng = np.random.default_rng(5)
    Jval = np.array([[2.0, 0.3, 1], [0.5, 1.5, 0]])
    vv, Dvv, wv, hv = (rng.standard_normal(2), rng.standard_normal((2, 2)),
                       rng.standard_normal(3), rng.standard_normal(2))
    feed = {f"J_{J.uid}": Jval, f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"w_{w.uid}": wv}
    jh = _run(Jv.body, {**feed, f"{seed.name}_{seed.uid}": hv})
    hd = T.var("hd", Rt)
    duh = _run(T.Eval(Dv, (0,), (hd,)), {**feed, f"hd_{hd.uid}": hv})
    fdp = _run(e, {**feed, f"v_{v.uid}": vv + 1e-6 * duh})
    fdm = _run(e, {**feed, f"v_{v.uid}": vv - 1e-6 * duh})
    np.testing.assert_allclose(np.asarray(jh).ravel(), ((fdp - fdm) / 2e-6).ravel(), atol=1e-4)


def test_embed_through_moving_img_differentiates() -> None:
    # The curved-element case: embed IS di, so embed(v, Img(J)) with a MARKED (moving) chart J now
    # differentiates term→term — d(embed) = embed(dv) + ∂(di)·v — instead of silently dropping the
    # ∂J part.  Verified vs FD in a CONSISTENT ImageBasis (embed and Pinv use the same immersion;
    # "bases just need to be consistent").
    from grassmann.core import basis as B
    from grassmann.lower.represent import default_basis, represent
    from grassmann.transform.diff import clear_subspace_gauges, set_subspace_gauge
    rng = np.random.default_rng(7)
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    x = T.var("x", Rt)
    v = T.var("v", imgJ)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    Jv, vv = rng.standard_normal((2, 3)), rng.standard_normal(2)
    Dvv, DJv = rng.standard_normal((2, 2)), rng.standard_normal((2, 2, 3))
    wv, hv = rng.standard_normal(3), rng.standard_normal(2)
    feed = {f"J_{J.uid}": Jv, f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"DJ_{DJ.uid}": DJv, f"w_{w.uid}": wv}
    ib = B.ImageBasis(J)  # the immersion (iso) basis — consistent with Pinv(J)
    env = {v.uid: ib, Dv.uid: B.basis_prod(B.dual_basis(default_basis(Rt)), ib)}

    def runp(t, ff):
        return np.asarray(represent(t, basis_env=env).run(ff)["result"], float)

    # The value frame here is the (non-orthonormal) ImageBasis — pin the AD gauge to ``immersion``
    # so the differential's frame matches it (the default gauge is the orthonormal QR ProjectedBasis).
    clear_subspace_gauges()
    set_subspace_gauge(J, "immersion")
    try:
        jd = diff(e, x, deps={v.uid: Dv, J.uid: DJ})
    finally:
        clear_subspace_gauges()
    seed = jd.vars[0]
    jh = runp(jd.body, {**feed, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    dvh = runp(T.Eval(Dv, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    fp = runp(e, {**feed, f"J_{J.uid}": Jv + 1e-6 * dJh, f"v_{v.uid}": vv + 1e-6 * dvh})
    fm = runp(e, {**feed, f"J_{J.uid}": Jv - 1e-6 * dJh, f"v_{v.uid}": vv - 1e-6 * dvh})
    np.testing.assert_allclose(jh, (fp - fm) / 2e-6, atol=1e-4)


def test_inner_with_xdep_metric_raises_else_zero() -> None:
    # Inner/Flat/Sharp of a WithInner space whose metric depends on x has a NONZERO operator
    # derivative (d⟨a,b⟩_g = aᵀ(∂g)b); the covariant path only intercepts Remetric, so a bare
    # Inner here must be a loud DiffError, not a SILENT Zero.  A CONSTANT metric is
    # genuinely x-independent ⇒ the zero map (no raise).
    V = S.base("V", 4)
    GS = S.oprod(S.dual(V), S.dual(V))
    g = T.var("g", GS)
    a, b = T.var("a", V), T.var("b", V)
    x = T.var("x", S.base("X", 2))
    Dg = T.var("Dg", S.oprod(S.dual(x.space), S.dual(V), S.dual(V)))
    pairing = T.Eval(T.inner(S.WithInner(V, g)), (0, 1), (a, b))
    with pytest.raises(DiffError):
        diff(pairing, x, deps={g.uid: Dg})  # x-dependent metric ⇒ loud bail
    assert isinstance(diff(pairing, x), T.Zero)  # constant metric (g ∉ deps) ⇒ zero map


def test_surface_diff_reachable() -> None:
    V = G.space("V", 3)
    x = G.var("x", V)
    e = G.inner(V)(x, x)  # ⟨x, x⟩
    J = G.diff(e, x)
    xval = np.random.randn(3)
    out = np.asarray(G.run(J, {_xkey(x.node): xval})["result"])
    np.testing.assert_allclose(out.ravel(), 2 * xval, atol=1e-6)


# ---------------------------------------------------------------------------
# FFS gauge × realized-basis FD matrix (ffs-frame).  The gauge-aware moving-subspace AD extends
# beyond ``Img(J)`` to the other three fundamental subspaces — ``CoImg(J)`` (row space, range frame
# SVD V[:,:δ]), ``Ker(J)`` (null space, complement V[:,δ:]) and ``CoKer(J)`` (left null, complement
# U[:,δ:]).  ``embed``/``project`` through a moving one realize in the SAME (DEFAULT) frame
# ``represent`` lowers, with the opaque self-differentiating frame tower fed via ``ffs_gauge_feed``.
# Each must FD-match: CoImg/Ker unify to their own coordinates and CoKer uses its own (left-null)
# frame, not the Img frame.
# Pick generic J with well-separated singular values so the SVD frame is stable across the FD steps.
# ---------------------------------------------------------------------------


# kind -> (ctor, Rt_dim, Re_dim, props, ambient="dom"/"cod", δ).  Non-degenerate per the FFS table:
# CoImg/Ker need a TALL surjective J (e.g. 4↦2); CoKer needs a WIDE injective J (2↦4).
_FFS_TEST_CASES = {
    "coimg": (S.coimg, 4, 2, G.surj, "dom", 2),
    "ker": (S.ker, 4, 2, G.surj, "dom", 2),
    "coker": (S.coker, 2, 4, G.inj, "cod", 2),
}


def _ffs_world(rng, kind):
    """A moving FFS subspace world (chart J + motion DJ + base values) for the given FFS ``kind``."""
    ctor, Rtd, Red, props, amb_side, delta = _FFS_TEST_CASES[kind]
    Rt, Re = S.base("Rt", Rtd), S.base("Re", Red)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=props)
    sub = ctor(J)
    amb = Rt if amb_side == "dom" else Re
    amb_dim = Rtd if amb_side == "dom" else Red
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    Jv = rng.standard_normal((Rtd, Red))
    DJv = rng.standard_normal((Rtd, Rtd, Red))
    hv = rng.standard_normal(Rtd)
    return Rt, Re, J, sub, amb, amb_dim, delta, x, DJ, Jv, DJv, hv, Rtd


@pytest.mark.parametrize("kind", list(_FFS_TEST_CASES))
def test_ffs_gauge_embed_fd(kind) -> None:
    # embed through a moving CoImg/Ker/CoKer(J): value in the DEFAULT (SVD) ProjectedBasis the
    # represent lowers, gauge = default qr, opaque frame tower fed via ffs_gauge_feed.  FD-matches.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(700 + len(kind))
    Rt, Re, J, sub, amb, amb_dim, delta, x, DJ, Jv, DJv, hv, Rtd = _ffs_world(rng, kind)
    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    w = T.var("w", S.dual(amb))
    e = T.Eval(w, (0,), (T.embed(v, amb),))
    vv = rng.standard_normal(delta)
    Dvv = rng.standard_normal((Rtd, delta))
    wv = rng.standard_normal(amb_dim)
    feed = {f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv,
            f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"w_{w.uid}": wv}

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)  # DEFAULT (SVD) ProjectedBasis

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    try:
        jd = diff(e, x, deps={v.uid: Dv, J.uid: DJ})
        extra = ffs_gauge_feed(sub, DJ, feed)  # FFS-aware tower feed (pass the subspace node)
    finally:
        clear_subspace_gauges()
    seed = jd.vars[0]
    jh = runp(jd.body, {**feed, **extra, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    dvh = runp(T.Eval(Dv, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    fp = runp(e, {**feed, f"J_{J.uid}": Jv + 1e-6 * dJh, f"v_{v.uid}": vv + 1e-6 * dvh})
    fm = runp(e, {**feed, f"J_{J.uid}": Jv - 1e-6 * dJh, f"v_{v.uid}": vv - 1e-6 * dvh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / 2e-6).ravel(), atol=1e-4)


@pytest.mark.parametrize("kind", list(_FFS_TEST_CASES))
def test_ffs_gauge_project_fd(kind) -> None:
    # project through a moving CoImg/Ker/CoKer(J) = di⁺ of the SVD frame, value in the DEFAULT
    # ProjectedBasis, opaque tower fed via ffs_gauge_feed.  FD-matches.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(800 + len(kind))
    Rt, Re, J, sub, amb, amb_dim, delta, x, DJ, Jv, DJv, hv, Rtd = _ffs_world(rng, kind)
    z = T.var("z", amb)  # ambient value to project into the subspace
    Dz = T.var("Dz", S.oprod(S.dual(Rt), amb))
    wc = T.var("wc", S.dual(sub))  # covector in the subspace's coordinate frame
    e = T.Eval(wc, (0,), (T.project(z, sub),))
    zv = rng.standard_normal(amb_dim)
    Dzv = rng.standard_normal((Rtd, amb_dim))
    wcv = rng.standard_normal(delta)
    feed = {f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv,
            f"z_{z.uid}": zv, f"Dz_{Dz.uid}": Dzv, f"wc_{wc.uid}": wcv}

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    try:
        jd = diff(e, x, deps={z.uid: Dz, J.uid: DJ})
        extra = ffs_gauge_feed(sub, DJ, feed)
    finally:
        clear_subspace_gauges()
    seed = jd.vars[0]
    jh = runp(jd.body, {**feed, **extra, f"{seed.name}_{seed.uid}": hv})
    dJh = _run(T.Eval(DJ, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    dzh = _run(T.Eval(Dz, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    fp = runp(e, {**feed, f"J_{J.uid}": Jv + 1e-6 * dJh, f"z_{z.uid}": zv + 1e-6 * dzh})
    fm = runp(e, {**feed, f"J_{J.uid}": Jv - 1e-6 * dJh, f"z_{z.uid}": zv - 1e-6 * dzh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / 2e-6).ravel(), atol=1e-4)


def test_ffs_gauge_ker_second_order_embed_fd() -> None:
    # Arbitrary-order FFS AD: diff(diff(embed)) through a MOVING Ker(J), value in the DEFAULT (SVD
    # complement) ProjectedBasis, with the whole Q-tower fed via ffs_gauge_feed — FD-matches a central
    # SECOND difference.  Shows the self-differentiating frame tower extends past the QR (Img) frame to
    # the SVD frames at arbitrary order.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(909)
    Rtd, Red, delta = 4, 2, 2
    Rt, Re = S.base("Rt", Rtd), S.base("Re", Red)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.surj)
    sub = S.ker(J)
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    D2J = T.var("D2J", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), Re))
    Jv = rng.standard_normal((Rtd, Red))
    DJv = _sym(rng.standard_normal((Rtd, Rtd, Red)), 1)
    D2Jv = _sym(rng.standard_normal((Rtd, Rtd, Rtd, Red)), 2)

    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    D2v = T.var("D2v", S.oprod(S.dual(Rt), S.dual(Rt), sub))
    w = T.var("w", S.dual(Rt))
    e = T.Eval(w, (0,), (T.embed(v, Rt),))
    vv = rng.standard_normal(delta)
    Dvv = rng.standard_normal((Rtd, delta))
    D2vv = _sym(rng.standard_normal((Rtd, Rtd, delta)), 2)
    wv = rng.standard_normal(Rtd)
    dval = rng.standard_normal(Rtd)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv,
        f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"D2v_{D2v.uid}": D2vv, f"w_{w.uid}": wv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    deps1 = {v.uid: Dv, J.uid: DJ}
    deps2 = {v.uid: Dv, J.uid: DJ, Dv.uid: D2v, DJ.uid: D2J}
    try:
        H = diff(diff(e, x, deps1), x, deps2)
        extra = ffs_gauge_feed(sub, [DJ, D2J], feed)
    finally:
        clear_subspace_gauges()
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    got = runp(body, {**feed, **extra, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    def perturb(s):
        dJ = s * np.einsum("akn,a->kn", DJv, dval) + 0.5 * s * s * np.einsum(
            "abkn,a,b->kn", D2Jv, dval, dval)
        dv = s * np.einsum("ak,a->k", Dvv, dval) + 0.5 * s * s * np.einsum(
            "abk,a,b->k", D2vv, dval, dval)
        return {**feed, f"J_{J.uid}": Jv + dJ, f"v_{v.uid}": vv + dv}

    fd2 = _second_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


# kind -> (ctor, Rt_dim, Re_dim, props, δ) for the higher-order FFS tower tests.  Full-rank J (the
# SVD frame is smooth, NO rank drift): TALL surjective for coimg/ker (4↦2, δ=2), WIDE injective for
# coker (2↦4, δ=cod−rank=2).  Pick generic J ⇒ well-separated singular values (stable SVD ordering).
_FFS_TOWER_CASES = {
    "coimg": (S.coimg, 4, 2, G.surj, 2),
    "ker": (S.ker, 4, 2, G.surj, 2),
    "coker": (S.coker, 2, 4, G.inj, 2),
}


def _ffs_tower_world(rng, kind, order):
    """A moving FFS subspace world with J's derivative tower ``[DJ, …, D^order J]`` (FULL-rank chart
    so the SVD frame is smooth), plus base values (J & each D^mJ symmetric in the differentiation
    axes).  Mirrors :func:`_qr_world` but for the SVD-framed CoImg/Ker/CoKer kinds."""
    ctor, Rtd, Red, props, delta = _FFS_TOWER_CASES[kind]
    Rt, Re = S.base("Rt", Rtd), S.base("Re", Red)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=props)
    sub = ctor(J)
    x = T.var("x", Rt)
    names = ["DJ"] + [f"D{m}J" for m in range(2, order + 1)]
    DJtower = [
        T.var(names[m - 1], S.oprod(*([S.dual(Rt)] * (m + 1)), Re)) for m in range(1, order + 1)
    ]
    Jv = rng.standard_normal((Rtd, Red))
    DJvtower = [
        _sym(rng.standard_normal((*([Rtd] * (m + 1)), Red)), m) for m in range(1, order + 1)
    ]
    return Rt, Re, J, sub, x, DJtower, Jv, DJvtower, Rtd, Red, delta


@pytest.mark.parametrize("kind", list(_FFS_TOWER_CASES))
def test_ffs_gauge_second_order_embed_fd(kind) -> None:
    # 2nd-order FFS AD: diff(diff(embed)) through a MOVING CoImg/Ker/CoKer(J), value in the DEFAULT
    # (SVD) ProjectedBasis, the self-differentiating frame tower [DJ, D2J] fed via ffs_gauge_feed —
    # FD-matches a central SECOND difference.  Generalizes the ker-only embed case to all three SVD
    # frames and both orders, proving the tower extends past the QR (Img) frame.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(909 + len(kind))
    Rt, Re, J, sub, x, (DJ, D2J), Jv, (DJv, D2Jv), Rtd, Red, delta = _ffs_tower_world(rng, kind, 2)
    amb = Rt if kind in ("coimg", "ker") else Re
    amb_dim = Rtd if kind in ("coimg", "ker") else Red
    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    D2v = T.var("D2v", S.oprod(S.dual(Rt), S.dual(Rt), sub))
    w = T.var("w", S.dual(amb))
    e = T.Eval(w, (0,), (T.embed(v, amb),))
    vv = rng.standard_normal(delta)
    Dvv = rng.standard_normal((Rtd, delta))
    D2vv = _sym(rng.standard_normal((Rtd, Rtd, delta)), 2)
    wv = rng.standard_normal(amb_dim)
    dval = rng.standard_normal(Rtd)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv,
        f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"D2v_{D2v.uid}": D2vv, f"w_{w.uid}": wv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    deps1 = {v.uid: Dv, J.uid: DJ}
    deps2 = {v.uid: Dv, J.uid: DJ, Dv.uid: D2v, DJ.uid: D2J}
    try:
        H = diff(diff(e, x, deps1), x, deps2)
        extra = ffs_gauge_feed(sub, [DJ, D2J], feed)
    finally:
        clear_subspace_gauges()
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    got = runp(body, {**feed, **extra, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    def perturb(s):
        dJ = s * np.einsum("akn,a->kn", DJv, dval) + 0.5 * s * s * np.einsum(
            "abkn,a,b->kn", D2Jv, dval, dval)
        dv = s * np.einsum("ak,a->k", Dvv, dval) + 0.5 * s * s * np.einsum(
            "abk,a,b->k", D2vv, dval, dval)
        return {**feed, f"J_{J.uid}": Jv + dJ, f"v_{v.uid}": vv + dv}

    fd2 = _second_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


@pytest.mark.parametrize("kind", list(_FFS_TOWER_CASES))
def test_ffs_gauge_second_order_project_fd(kind) -> None:
    # 2nd-order FFS AD for PROJECT (di⁺ of the moving SVD frame) through a MOVING CoImg/Ker/CoKer(J),
    # value in the DEFAULT ProjectedBasis, tower [DJ, D2J] fed via ffs_gauge_feed — FD-matches a
    # central SECOND difference.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(1009 + len(kind))
    Rt, Re, J, sub, x, (DJ, D2J), Jv, (DJv, D2Jv), Rtd, Red, delta = _ffs_tower_world(rng, kind, 2)
    amb = Rt if kind in ("coimg", "ker") else Re
    amb_dim = Rtd if kind in ("coimg", "ker") else Red
    z = T.var("z", amb)  # ambient value to project into the subspace
    Dz = T.var("Dz", S.oprod(S.dual(Rt), amb))
    D2z = T.var("D2z", S.oprod(S.dual(Rt), S.dual(Rt), amb))
    wc = T.var("wc", S.dual(sub))
    e = T.Eval(wc, (0,), (T.project(z, sub),))
    zv = rng.standard_normal(amb_dim)
    Dzv = rng.standard_normal((Rtd, amb_dim))
    D2zv = _sym(rng.standard_normal((Rtd, Rtd, amb_dim)), 2)
    wcv = rng.standard_normal(delta)
    dval = rng.standard_normal(Rtd)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv,
        f"z_{z.uid}": zv, f"Dz_{Dz.uid}": Dzv, f"D2z_{D2z.uid}": D2zv, f"wc_{wc.uid}": wcv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    deps1 = {z.uid: Dz, J.uid: DJ}
    deps2 = {z.uid: Dz, J.uid: DJ, Dz.uid: D2z, DJ.uid: D2J}
    try:
        H = diff(diff(e, x, deps1), x, deps2)
        extra = ffs_gauge_feed(sub, [DJ, D2J], feed)
    finally:
        clear_subspace_gauges()
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    got = runp(body, {**feed, **extra, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    def perturb(s):
        dJ = s * np.einsum("akn,a->kn", DJv, dval) + 0.5 * s * s * np.einsum(
            "abkn,a,b->kn", D2Jv, dval, dval)
        dz = s * np.einsum("ak,a->k", Dzv, dval) + 0.5 * s * s * np.einsum(
            "abk,a,b->k", D2zv, dval, dval)
        return {**feed, f"J_{J.uid}": Jv + dJ, f"z_{z.uid}": zv + dz}

    fd2 = _second_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


@pytest.mark.parametrize("kind", ["coimg"])
def test_ffs_gauge_third_order_embed_fd(kind) -> None:
    # THIRD-order SVD-framed FFS AD — proves the tower extends past QR to the SVD frames at 3rd order.
    # Mirrors test_qr_gauge_third_order_embed_fd exactly (Richardson _third_central, tight rtol/atol
    # 1e-4).  Small subspace (surj 3↦2, δ=2) to keep the 3rd-derivative tensors cheap.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(1111)
    Rtd, Red, delta = 3, 2, 2
    Rt, Re = S.base("Rt", Rtd), S.base("Re", Red)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.surj)
    sub = S.coimg(J) if kind == "coimg" else S.ker(J)
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    D2J = T.var("D2J", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), Re))
    D3J = T.var("D3J", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), S.dual(Rt), Re))
    Jv = rng.standard_normal((Rtd, Red))
    DJv = _sym(rng.standard_normal((Rtd, Rtd, Red)), 1)
    D2Jv = _sym(rng.standard_normal((Rtd, Rtd, Rtd, Red)), 2)
    D3Jv = _sym(rng.standard_normal((Rtd, Rtd, Rtd, Rtd, Red)), 3)

    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    D2v = T.var("D2v", S.oprod(S.dual(Rt), S.dual(Rt), sub))
    D3v = T.var("D3v", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), sub))
    w = T.var("w", S.dual(Rt))
    e = T.Eval(w, (0,), (T.embed(v, Rt),))
    vv = rng.standard_normal(delta)
    Dvv = rng.standard_normal((Rtd, delta))
    D2vv = _sym(rng.standard_normal((Rtd, Rtd, delta)), 2)
    D3vv = _sym(rng.standard_normal((Rtd, Rtd, Rtd, delta)), 3)
    wv = rng.standard_normal(Rtd)
    dval = rng.standard_normal(Rtd)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv, f"D3J_{D3J.uid}": D3Jv,
        f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"D2v_{D2v.uid}": D2vv, f"D3v_{D3v.uid}": D3vv,
        f"w_{w.uid}": wv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    deps1 = {v.uid: Dv, J.uid: DJ}
    deps2 = {**deps1, Dv.uid: D2v, DJ.uid: D2J}
    deps3 = {**deps2, D2v.uid: D3v, D2J.uid: D3J}
    try:
        H = diff(diff(diff(e, x, deps1), x, deps2), x, deps3)
        extra = ffs_gauge_feed(sub, [DJ, D2J, D3J], feed)
    finally:
        clear_subspace_gauges()
    h3, h2, h1 = H.vars[0], H.body.vars[0], H.body.body.vars[0]
    body = H.body.body.body
    got = runp(body, {
        **feed, **extra,
        f"{h3.name}_{h3.uid}": dval, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval,
    })

    def perturb(s):
        dJ = (
            s * np.einsum("akn,a->kn", DJv, dval)
            + 0.5 * s * s * np.einsum("abkn,a,b->kn", D2Jv, dval, dval)
            + (s ** 3 / 6.0) * np.einsum("abckn,a,b,c->kn", D3Jv, dval, dval, dval)
        )
        dv = (
            s * np.einsum("ak,a->k", Dvv, dval)
            + 0.5 * s * s * np.einsum("abk,a,b->k", D2vv, dval, dval)
            + (s ** 3 / 6.0) * np.einsum("abck,a,b,c->k", D3vv, dval, dval, dval)
        )
        return {**feed, f"J_{J.uid}": Jv + dJ, f"v_{v.uid}": vv + dv}

    fd3 = _third_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd3), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Non-injective ``Img(J)`` gauge × realized-basis FD (ffs-frame, the rank-deficient Img case).
# For an INJECTIVE J the range frame is qr_pos(Jᵀ) (dim = dom); for a NON-injective J (``props=rank(r)``,
# r < dom) ``represent`` realizes ``Img(J)`` in the SVD LEFT-singular frame ``U[:,:r]`` (dim = r), NOT
# qr_pos.  So the non-inj Img must route through the same subspace-coord SVD path as
# CoImg/Ker/CoKer (matching the QR/dom-dim gauge to the SVD/rank-dim value would be SILENTLY WRONG),
# with the slice width PINNED at the declared rank so the Taylor-reconstructed frame
# does not drift off the rank-r manifold.  Both embed and project
# FD-match to machine precision (residuals ~1e-11 at eps=1e-6).
#
# Build an EXACTLY-rank-r moving subspace ``Img(J(x))`` via ``Jm(x) = M(x)·N(x)`` (M cod×r, N r×dom,
# both smooth in x) so ``Img`` genuinely MOVES while staying exactly rank r, and feed ``diff`` the
# matching first-derivative ``DJ`` from the product rule.
# ---------------------------------------------------------------------------


def _noninj_img_world(rng):
    """An exactly-rank-r moving ``Img(J)`` world: chart ``J:Rt(3)→Re(4)`` of declared ``rank(2)`` with
    ``Jm(x)=M(x)·N(x)``, its motion ``DJ`` (product rule), and an ``x → J(x₀+x)`` immersion builder for
    the FD perturbation (so the perturbed J stays exactly rank 2)."""
    Rtd, Red, r = 3, 4, 2
    Rt, Re = S.base("Rt", Rtd), S.base("Re", Red)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.rank(2))
    sub = S.img(J)
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    M0 = rng.standard_normal((Red, r))
    N0 = rng.standard_normal((r, Rtd))
    Md = rng.standard_normal((Rtd, Red, r))  # dM/dx_a
    Nd = rng.standard_normal((Rtd, r, Rtd))  # dN/dx_a

    def Jval_of(delta):  # grassmann (k=dom, n=cod) of the exactly-rank-r Jm(x₀+delta)
        M = M0 + np.tensordot(delta, Md, axes=([0], [0]))
        N = N0 + np.tensordot(delta, Nd, axes=([0], [0]))
        return (M @ N).T

    Jv = Jval_of(np.zeros(Rtd))
    DJv = np.stack([(Md[a] @ N0 + M0 @ Nd[a]).T for a in range(Rtd)], axis=0)  # (a, k, n)
    # M,N LINEAR in x ⇒ J = M·N is degree-2 ⇒ the tower is FINITE & EXACT: D²J is the symmetric
    # product-rule second derivative ∂²(MN)[a,b] = Md[a]·Nd[b] + Md[b]·Nd[a] (M''=N''=0), and D³J≡0.
    # So the feed's Taylor reconstruction is EXACTLY rank-r at every step (no frame drift).
    D2Jv = np.stack(
        [np.stack([(Md[a] @ Nd[b] + Md[b] @ Nd[a]).T for b in range(Rtd)], axis=0) for a in range(Rtd)],
        axis=0,
    )  # (a, b, k, n), symmetric in a↔b
    hv = rng.standard_normal(Rtd)
    return Rt, Re, J, sub, x, DJ, Jv, DJv, D2Jv, Jval_of, hv, Rtd, Red, r


def test_noninj_img_frame_is_svd() -> None:
    # Sanity: represent realizes a non-injective Img(J) in the SVD left-singular frame U[:,:r], which is
    # exactly what _ffs_frame_value("img", ·) now returns (column span; signs are gauge).
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import _ffs_frame_value

    rng = np.random.default_rng(515)
    _Rt, Re, J, sub, _x, _DJ, Jv, _DJv, _D2Jv, _Jof, _hv, _Rtd, _Red, r = _noninj_img_world(rng)
    v = T.var("v", sub)
    cols = []
    for i in range(r):
        vv = np.zeros(r)
        vv[i] = 1.0
        out = np.asarray(represent(T.embed(v, Re)).run({f"J_{J.uid}": Jv, f"v_{v.uid}": vv})["result"], float)
        cols.append(out)
    rep_frame = np.array(cols).T  # cod × r, columns = represent's realization frame
    svd_frame = _ffs_frame_value("img", Jv)  # cod × r
    assert svd_frame.shape == (_Red, r)
    # Same column span (orthogonal projectors agree) — the per-column sign is a gauge choice.
    P_rep = rep_frame @ rep_frame.T
    P_svd = svd_frame @ svd_frame.T
    np.testing.assert_allclose(P_rep, P_svd, atol=1e-10)


def test_noninj_img_gauge_embed_fd() -> None:
    # embed through a moving NON-injective Img(J): value in the DEFAULT (SVD) ProjectedBasis represent
    # lowers, gauge = default qr, opaque frame tower fed via ffs_gauge_feed.  FD-matches to ~machine
    # precision (the silent-wrong QR/dom-dim path is gone).
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(516)
    Rt, Re, J, sub, x, DJ, Jv, DJv, D2Jv, Jval_of, hv, Rtd, Red, r = _noninj_img_world(rng)
    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    vv = rng.standard_normal(r)
    Dvv = rng.standard_normal((Rtd, r))
    wv = rng.standard_normal(Red)
    feed = {f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv,
            f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"w_{w.uid}": wv}

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    try:
        jd = diff(e, x, deps={v.uid: Dv, J.uid: DJ})
        extra = ffs_gauge_feed(sub, DJ, feed)
    finally:
        clear_subspace_gauges()
    seed = jd.vars[0]
    jh = runp(jd.body, {**feed, **extra, f"{seed.name}_{seed.uid}": hv})
    dvh = runp(T.Eval(Dv, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    eps = 1e-6  # perturb x along hv; rebuild J(x₀±eps·h) EXACTLY rank-r via Jval_of (no rank drift)
    fp = runp(e, {**feed, f"J_{J.uid}": Jval_of(eps * hv), f"v_{v.uid}": vv + eps * dvh})
    fm = runp(e, {**feed, f"J_{J.uid}": Jval_of(-eps * hv), f"v_{v.uid}": vv - eps * dvh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / (2 * eps)).ravel(), atol=1e-6)


def test_noninj_img_gauge_project_fd() -> None:
    # project through a moving NON-injective Img(J) = di⁺ of the SVD frame, value in the DEFAULT
    # ProjectedBasis, opaque tower fed via ffs_gauge_feed.  FD-matches to ~machine precision.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(517)
    Rt, Re, J, sub, x, DJ, Jv, DJv, D2Jv, Jval_of, hv, Rtd, Red, r = _noninj_img_world(rng)
    z = T.var("z", Re)  # ambient value to project into Img(J)
    Dz = T.var("Dz", S.oprod(S.dual(Rt), Re))
    wc = T.var("wc", S.dual(sub))  # covector in the subspace's coordinate frame
    e = T.Eval(wc, (0,), (T.project(z, sub),))
    zv = rng.standard_normal(Red)
    Dzv = rng.standard_normal((Rtd, Red))
    wcv = rng.standard_normal(r)
    feed = {f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv,
            f"z_{z.uid}": zv, f"Dz_{Dz.uid}": Dzv, f"wc_{wc.uid}": wcv}

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    try:
        jd = diff(e, x, deps={z.uid: Dz, J.uid: DJ})
        extra = ffs_gauge_feed(sub, DJ, feed)
    finally:
        clear_subspace_gauges()
    seed = jd.vars[0]
    jh = runp(jd.body, {**feed, **extra, f"{seed.name}_{seed.uid}": hv})
    dzh = _run(T.Eval(Dz, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    eps = 1e-6
    fp = runp(e, {**feed, f"J_{J.uid}": Jval_of(eps * hv), f"z_{z.uid}": zv + eps * dzh})
    fm = runp(e, {**feed, f"J_{J.uid}": Jval_of(-eps * hv), f"z_{z.uid}": zv - eps * dzh})
    np.testing.assert_allclose(np.ravel(jh), ((fp - fm) / (2 * eps)).ravel(), atol=1e-6)


def test_noninj_img_gauge_second_order_embed_fd() -> None:
    # 2nd-order embed through a moving NON-injective Img(J), the "stable region" construction: M,N
    # LINEAR in x ⇒ J = M·N is degree-2 ⇒ the derivative tower [DJ, D2J] is FINITE and EXACT (D³J≡0),
    # so ffs_gauge_feed's Taylor reconstruction is EXACTLY rank-r at every step (no drift).  diff² in
    # the DEFAULT (SVD) ProjectedBasis, FD vs a central SECOND difference whose J-perturbation uses
    # Jval_of(±s·h) — the EXACT rank-r geometry.  The O(δ²) drift that bit 1st-order-tower
    # reconstructions is gone because the degree-2 tower is exact, so this matches SHARPLY.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(519)
    Rt, Re, J, sub, x, DJ, Jv, DJv, D2Jv, Jval_of, hv, Rtd, Red, r = _noninj_img_world(rng)
    D2J = T.var("D2J", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), Re))
    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    D2v = T.var("D2v", S.oprod(S.dual(Rt), S.dual(Rt), sub))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    vv = rng.standard_normal(r)
    Dvv = rng.standard_normal((Rtd, r))
    D2vv = _sym(rng.standard_normal((Rtd, Rtd, r)), 2)
    wv = rng.standard_normal(Red)
    dval = rng.standard_normal(Rtd)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv,
        f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"D2v_{D2v.uid}": D2vv, f"w_{w.uid}": wv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    deps1 = {v.uid: Dv, J.uid: DJ}
    deps2 = {v.uid: Dv, J.uid: DJ, Dv.uid: D2v, DJ.uid: D2J}
    try:
        H = diff(diff(e, x, deps1), x, deps2)
        extra = ffs_gauge_feed(sub, [DJ, D2J], feed)
    finally:
        clear_subspace_gauges()
    assert isinstance(H, T.LLam) and isinstance(H.body, T.LLam)
    h2, h1, body = H.vars[0], H.body.vars[0], H.body.body
    got = runp(body, {**feed, **extra, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval})

    def perturb(s):  # EXACT rank-r J(x₀+s·dval) via Jval_of; v Taylored to 2nd order.
        dv = s * np.einsum("ak,a->k", Dvv, dval) + 0.5 * s * s * np.einsum(
            "abk,a,b->k", D2vv, dval, dval)
        return {**feed, f"J_{J.uid}": Jval_of(s * dval), f"v_{v.uid}": vv + dv}

    fd2 = _second_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd2), rtol=2e-2, atol=5e-3)


def test_noninj_img_gauge_third_order_embed_fd() -> None:
    # 3rd-order embed through a moving NON-injective Img(J), same "stable region" construction: M,N
    # LINEAR in x ⇒ J = M·N is degree-2 ⇒ the J-derivative tower is finite & EXACT to ALL orders
    # (D³J≡0), so ffs_gauge_feed's Taylor reconstruction is EXACTLY rank-r at every step — no drift.
    # But the SVD frame Q(J) is NONLINEAR, so its 3rd derivative ∂³Q is real & nonzero: this exercises
    # the tower walking to depth 3 over the rank-deficient SVD frame.  diff³ in the DEFAULT (SVD)
    # ProjectedBasis, FD vs the Richardson _third_central whose J-perturbation uses Jval_of(±s·h) (the
    # EXACT rank-r geometry).  Matches SHARPLY (rtol/atol=1e-4) — no rank drift even at 3rd order.
    from grassmann.lower.represent import represent
    from grassmann.transform.diff import (
        clear_subspace_gauges,
        ffs_gauge_feed,
        set_subspace_gauge,
    )

    rng = np.random.default_rng(520)
    Rt, Re, J, sub, x, DJ, Jv, DJv, D2Jv, Jval_of, hv, Rtd, Red, r = _noninj_img_world(rng)
    D2J = T.var("D2J", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), Re))
    D3J = T.var("D3J", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), S.dual(Rt), Re))
    D3Jv = np.zeros((Rtd, Rtd, Rtd, Rtd, Red))  # J is degree-2 ⇒ D³J ≡ 0 (the tower var must exist)
    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    D2v = T.var("D2v", S.oprod(S.dual(Rt), S.dual(Rt), sub))
    D3v = T.var("D3v", S.oprod(S.dual(Rt), S.dual(Rt), S.dual(Rt), sub))
    w = T.var("w", S.dual(Re))
    e = T.Eval(w, (0,), (T.embed(v, Re),))
    vv = rng.standard_normal(r)
    Dvv = rng.standard_normal((Rtd, r))
    D2vv = _sym(rng.standard_normal((Rtd, Rtd, r)), 2)
    D3vv = _sym(rng.standard_normal((Rtd, Rtd, Rtd, r)), 3)
    wv = rng.standard_normal(Red)
    dval = rng.standard_normal(Rtd)
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv, f"D2J_{D2J.uid}": D2Jv, f"D3J_{D3J.uid}": D3Jv,
        f"v_{v.uid}": vv, f"Dv_{Dv.uid}": Dvv, f"D2v_{D2v.uid}": D2vv, f"D3v_{D3v.uid}": D3vv,
        f"w_{w.uid}": wv,
    }

    def runp(t, ff):
        return np.asarray(represent(t).run(ff)["result"], float)

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    deps1 = {v.uid: Dv, J.uid: DJ}
    deps2 = {**deps1, Dv.uid: D2v, DJ.uid: D2J}
    deps3 = {**deps2, D2v.uid: D3v, D2J.uid: D3J}
    try:
        H = diff(diff(diff(e, x, deps1), x, deps2), x, deps3)
        extra = ffs_gauge_feed(sub, [DJ, D2J, D3J], feed)
    finally:
        clear_subspace_gauges()
    h3, h2, h1 = H.vars[0], H.body.vars[0], H.body.body.vars[0]
    body = H.body.body.body
    got = runp(body, {
        **feed, **extra,
        f"{h3.name}_{h3.uid}": dval, f"{h2.name}_{h2.uid}": dval, f"{h1.name}_{h1.uid}": dval,
    })

    def perturb(s):  # EXACT rank-r J(x₀+s·dval) via Jval_of; v Taylored to 3rd order.
        dv = (
            s * np.einsum("ak,a->k", Dvv, dval)
            + 0.5 * s * s * np.einsum("abk,a,b->k", D2vv, dval, dval)
            + (s ** 3 / 6.0) * np.einsum("abck,a,b,c->k", D3vv, dval, dval, dval)
        )
        return {**feed, f"J_{J.uid}": Jval_of(s * dval), f"v_{v.uid}": vv + dv}

    fd3 = _third_central(runp, e, perturb)
    np.testing.assert_allclose(np.ravel(got), np.ravel(fd3), rtol=1e-4, atol=1e-4)


def test_noninj_img_immersion_gauge_raises() -> None:
    # The closed-form immersion/polar gauges assume an INJECTIVE immersion (full column rank).  For a
    # non-injective Img(J) they have no valid frame (Img realizes in the SVD frame) — diff must raise a
    # clear DiffError rather than lower a wrong closed-form gauge.
    from grassmann.transform.diff import clear_subspace_gauges, set_subspace_gauge

    rng = np.random.default_rng(518)
    Rt, Re, J, sub, x, DJ, _Jv, _DJv, _D2Jv, _Jof, _hv, _Rtd, _Red, _r = _noninj_img_world(rng)
    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    e = T.embed(v, Re)
    for gauge in ("immersion", "polar"):
        clear_subspace_gauges()
        set_subspace_gauge(J, gauge)
        try:
            with pytest.raises(DiffError, match="[Ii]njective"):
                diff(e, x, deps={v.uid: Dv, J.uid: DJ})
        finally:
            clear_subspace_gauges()


def test_diff_logdet_det_of_nonbij_endomorphism_is_welltyped() -> None:
    """diff of det/logdet of an x-dependent square endomorphism that is NOT a bij-asserted Var
    (here ``M0{bij} + bump(x)``, a Sum) must produce a WELL-TYPED Jacobian.

    Jacobi's formula ``d det = det·tr(f⁻¹ df)`` needs ``f⁻¹`` = ``Solve(f)``, which requires ``f``
    asserted ``bij``.  ``diff`` now emits that assertion (``Assert(f, …, bij)``); regression guard
    that (a) the derivative type-checks, (b) it matches finite differences, (c) the load-bearing
    assertion survives ``simplify``/``normal_form``/``beta``."""
    from grassmann.core.props import bij
    from grassmann.infer.synth import synth_space
    import grassmann as G

    V = S.base("V", 3)
    x = T.var("x", V)
    M0 = T.var("M0", S.oprod(S.dual(V), V), props=bij)
    bump = T.scale(0.05, T.oprod_v(T.Eval(T.flat(V), (0,), (x,)), x))
    Mx = T.sum_(M0, bump)  # an x-dependent square endomorphism; NOT itself asserted bij

    M0v = 1.5 * np.eye(3) + 0.3 * np.random.randn(3, 3)
    for build in (T.LogDet, T.Det):
        F = build(Mx)
        J = diff(F, x)
        assert isinstance(J, T.LLam)
        synth_space(J.body)  # must type-check (raised InferError before the fix)
        feed = {"M0_%d" % M0.uid: M0v, _xkey(x): np.random.randn(3)}
        h = np.random.randn(3)
        got = _run(J.body, {**feed, "%s_%d" % (J.vars[0].name, J.vars[0].uid): h}).ravel()
        fd = _fd_jacobian(F, x, feed, 3) @ h
        np.testing.assert_allclose(got, fd, atol=1e-4)
        # the bij assertion must survive every value-preserving rewrite
        for rw in (G.simplify, G.normal_form, G.beta):
            r = rw(J.body)
            synth_space(r.node if hasattr(r, "node") else r)


def test_direct_sum_diagonal_is_linear_and_second_diff_is_zero() -> None:
    """``x ⊕ x`` uses x twice, so its Jacobian is the direct-sum DIAGONAL ``h ↦ h⊕h``.  ``⊕`` is
    additive (parallel slots), so that is a valid LINEAR map (h consumed once, not twice), and the
    second derivative of the linear ``Mo(x⊕x)`` is the zero map.  Regression: the exactly-once
    linearity check used to treat OPlusV as multiplicative and reject ``h⊕h`` (a LinearityError
    leaked from the second differentiation)."""
    from grassmann.infer.linearity import check_linearity, LinearityError

    V = S.base("V", 3)
    h = T.var("h", V)
    check_linearity(T.LLam((h,), T.OPlusV((h, h))))                 # diagonal: linear
    check_linearity(T.LLam((h,), T.OPlusV((h, T.zero(V)))))         # injection: linear
    with pytest.raises(LinearityError):                            # tensor diagonal: nonlinear
        check_linearity(T.LLam((h,), T.OProdV((h, h))))
    with pytest.raises(LinearityError):                            # nonzero const slot: affine
        check_linearity(T.LLam((h,), T.OPlusV((h, T.var("c", V)))))

    x = T.var("x", V)
    W = S.oplus(V, V)
    Mo = T.var("Mo", S.oprod(S.dual(W), V))
    F = T.Eval(Mo, (0,), (T.OPlusV((x, x)),))  # linear in x
    assert isinstance(diff(diff(F, x), x), T.Zero)  # 2nd derivative of a linear term is 0
