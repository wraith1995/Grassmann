"""Tests for total covariant differentiation (:mod:`grassmann.transform.diff`).

``diff(e, x, deps)`` returns the total derivative of a closed term ``e`` w.r.t. a value
variable ``x`` as a coordinate-free map ``X ⊸ space(e)``. Cases are checked against
central-difference Jacobians of the value term, closed forms, and — for the covariant
case — a hand-computed Levi-Civita connection.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
import grassmann.stdlib as L
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.infer.linearity import check_linearity
from grassmann.transform.diff import DiffError, diff


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


@pytest.mark.parametrize("op", [L.sqrt, L.reciprocal])
def test_scalar_chain_rules_vs_fd(op) -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    dotxx = T.Eval(T.inner(V), (0, 1), (x, x))
    e = op(G.wrap(dotxx)).node
    J = diff(e, x)
    xval = np.array([1.3, -0.7])
    feed = {_xkey(x): xval}
    np.testing.assert_allclose(_run(J, feed).ravel(), _fd_jacobian(e, x, feed, 2).ravel(), atol=1e-4)


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
    np.testing.assert_allclose(Jv, np.einsum("imk,m->ki", DAv, vval), atol=1e-6)


def test_inverse_through_map_is_rejected() -> None:
    # The APPLIED inverse SolveApply(A, b) with A marked still routes the seed through the bulk op
    # ⇒ rejected.  (The BARE form Eval(Solve(A), b) is now supported — see below.)
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
    np.testing.assert_allclose(Jv, hand, atol=1e-6)


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
    np.testing.assert_allclose(Jv, hand, atol=1e-6)


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
    Jv = _run(J, feed)  # stored (dom, cod, X)

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
        np.testing.assert_allclose(Jv[:, :, i].T, dT_op + corr, atol=1e-6)


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
    lhs = secv @ Jcv + omv @ Jtv  # ⟨∇ω,v⟩ + ⟨ω,∇v⟩, indexed by direction i
    rhs = Domv @ secv + Dsecv @ omv  # d⟨ω,v⟩ = (∂ω)v + ω(∂v)
    np.testing.assert_allclose(lhs, rhs, atol=1e-6)


# ---------------------------------------------------------------------------
# Frame / chart-Jacobian connection (task 3): ∇ from a moving frame J(x), the FEM case.
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
    # space) now COMPUTES the realization derivative ∂(f⁺)·z (the un-gated pseudoinverse derivative)
    # instead of silently dropping it / bailing — no longer a Zero or a raise.
    W, V = S.base("W", 2), S.base("V", 4)
    f = T.var("f", S.oprod(S.dual(W), V), props=G.inj)
    z = T.var("z", V)
    J = diff(T.Project(z, S.img(f)), f)  # d/df of the projection w.r.t. its defining map
    assert not isinstance(J, T.Zero)  # the ∂f contribution is no longer silently dropped
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

    jd = diff(e, x, deps={v.uid: Dv, J.uid: DJ})
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
    # Inner here was a SILENT Zero — now a loud DiffError (audit bug B).  A CONSTANT metric is
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
