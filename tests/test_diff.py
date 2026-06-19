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
    V = S.base("V", 2)
    x = T.var("x", V)
    A = T.var("A", S.oprod(S.dual(V), V), props=G.bij)
    e = T.SolveApply(A, x)  # A⁻¹ x  (seed through the vector)
    J = diff(e, x)
    Aval = np.random.randn(2, 2) + 3 * np.eye(2)
    xval = np.random.randn(2)
    feed = {f"A_{A.uid}": Aval, _xkey(x): xval}
    Jv = _run(J, feed)
    fd = _fd_jacobian(e, x, feed, 2)
    assert np.allclose(Jv, fd, atol=1e-4) or np.allclose(Jv, fd.T, atol=1e-4)


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
    V = S.base("V", 2)
    x = T.var("x", V)
    A = T.var("A", S.oprod(S.dual(V), V), props=G.bij)
    b = T.var("b", V)
    DA = T.var("DA", S.oprod(S.dual(V), S.oprod(S.dual(V), V)))
    with pytest.raises(DiffError):
        diff(T.SolveApply(A, b), x, deps={A.uid: DA})


# ---------------------------------------------------------------------------
# Covariant connection (tangent space with a non-constant metric).
# ---------------------------------------------------------------------------


def test_flat_metric_recovers_total_derivative() -> None:
    # A constant (unmarked) metric ⇒ the connection correction is exactly Zero.
    from grassmann.transform.diff import _covariant_correction, _DCtx

    V = S.base("V", 2)
    x = T.var("x", V)
    g = T.var("g", S.oprod(S.dual(V), S.dual(V)))
    e = T.Remetric(T.var("sec", V), g)
    ctx = _DCtx(x=x, h=T.var("dx", V), deps={})  # g absent ⇒ constant
    corr = _covariant_correction(e, G.infer.synth_space(e), ctx)
    assert isinstance(corr, T.Zero)


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
    from grassmann.core.space import with_inner
    from grassmann.transform.diff import _connection_map

    M = S.base("M", 2)
    h = T.var("dx", M)
    g = T.var("g", S.oprod(S.dual(M), S.dual(M)))
    Dg = T.var("Dg", S.oprod(S.dual(M), S.dual(M), S.dual(M)))
    v = T.var("v", M)
    MR = with_inner(M, g)
    conn = _connection_map(M, MR, Dg, h)  # Γ(h) : M ⊸ M (seed-carrying)
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
# Failure modes.
# ---------------------------------------------------------------------------


def test_higher_order_is_rejected() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    # a Frame (basis binder) on the path ⇒ DiffError.
    with pytest.raises(DiffError):
        diff(T.NEval(T.var("f", V), (0,), (x,)), x)


def test_surface_diff_reachable() -> None:
    V = G.space("V", 3)
    x = G.var("x", V)
    e = G.inner(V)(x, x)  # ⟨x, x⟩
    J = G.diff(e, x)
    xval = np.random.randn(3)
    out = np.asarray(G.run(J, {_xkey(x.node): xval})["result"])
    np.testing.assert_allclose(out.ravel(), 2 * xval, atol=1e-6)
