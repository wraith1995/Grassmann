"""Phase-C geometry: the ``Parameterization`` surface — tangent = ``Img(dφ)``, and the
immersion-gauge **covariant derivative** ``∇v = Γ·v`` validated against analytic references
(sphere / Monge patches, the same closed-form Christoffel ground truth used by
``test_diff_analytic.py::test_covariant_derivative_via_christoffel``)."""

from __future__ import annotations

import numpy as np
import pytest

from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.syntax.geometry import Parameterization


# --- analytic immersion references (grassmann (k=2,n=3) layout: di[a,p]=∂iₚ/∂uₐ) ----------

def _sphere(u):
    u1, u2 = u
    s1, c1, s2, c2 = np.sin(u1), np.cos(u1), np.sin(u2), np.cos(u2)
    di = np.array([[c1 * c2, c1 * s2, -s1], [-s1 * s2, s1 * c2, 0.0]])
    d2i = np.zeros((2, 2, 3))
    d2i[0, 0] = [-s1 * c2, -s1 * s2, -c1]
    d2i[0, 1] = [-c1 * s2, c1 * c2, 0.0]
    d2i[1, 0] = d2i[0, 1]
    d2i[1, 1] = [-s1 * c2, -s1 * s2, 0.0]
    return di, d2i


def _monge(u):
    u1, u2 = u
    fu1 = u1 + 0.7 * u2 + 0.6 * u1 * u1 + 0.3 * u1 * u2
    fu2 = 0.7 * u1 - 0.6 * u2 - 0.3 * u2 * u2 + 0.15 * u1 * u1
    di = np.array([[1.0, 0.0, fu1], [0.0, 1.0, fu2]])
    d2i = np.zeros((2, 2, 3))
    d2i[0, 0, 2] = 1.0 + 1.2 * u1 + 0.3 * u2
    d2i[0, 1, 2] = 0.7 + 0.3 * u1
    d2i[1, 0, 2] = d2i[0, 1, 2]
    d2i[1, 1, 2] = -0.6 - 0.6 * u2
    return di, d2i


def _gauss(di, d2i):
    g = di @ di.T
    ginv = np.linalg.inv(g)
    dg = np.einsum("abp,dp->abd", d2i, di) + np.einsum("bp,adp->abd", di, d2i)
    gamma = 0.5 * np.einsum("cd,abd->cab", ginv,
                            dg + dg.transpose(1, 0, 2) - dg.transpose(2, 1, 0))
    return ginv, gamma


# --- the subspace surface --------------------------------------------------

def test_four_subspaces():
    par = Parameterization("phi", 2, 3)
    assert par.tangent == S.img(par.J)
    assert par.normal == S.coker(par.J)
    assert par.cotangent == S.dual(S.img(par.J))
    assert par.conormal == S.dual(S.coker(par.J))


# --- the covariant derivative ----------------------------------------------

@pytest.mark.parametrize("name, imm, u0", [
    ("sphere", _sphere, np.array([0.9, 0.5])),
    ("monge", _monge, np.array([0.3, -0.4])),
])
def test_covariant_derivative_matches_christoffel(name, imm, u0):
    di, d2i = imm(u0)
    ginv, gamma = _gauss(di, d2i)

    par = Parameterization(name, 2, 3)
    x = par.point("x")
    v, Dv = par.tangent_vector("v")
    w = T.var("w", S.dual(par.ambient))                 # a covector to read ambient components
    e = T.Eval(w, (0,), (par.embed(v),))                # ⟨w, embed(v)⟩
    jd = par.covariant_diff(e, x, marks={v: Dv})        # immersion-gauge covariant derivative
    seed = jd.vars[0]                                    # the direction binder h
    env = par.basis_env(v, Dv)

    vv = np.random.default_rng(7 + len(name)).standard_normal(2)   # constant tangent coords
    base = {par.key(par.J): di, par.key(par.DJ): d2i,
            par.key(v): vv, par.key(Dv): np.zeros((2, 2))}

    # AD[a,p] = ∂ₐ(embed v)_p — the directional-derivative matrix
    AD = np.zeros((2, 3))
    for a in range(2):
        for p in range(3):
            AD[a, p] = par.run(jd.body, env,
                               {**base, f"w_{w.uid}": np.eye(3)[p],
                                f"{seed.name}_{seed.uid}": np.eye(2)[a]})

    # (1) ∂(embed v) == ∂²i·v  (immersion gauge is closed-form, machine precision)
    np.testing.assert_allclose(AD, np.einsum("abp,b->ap", d2i, vv), atol=1e-10)
    # (2) the tangential part == Γ·v  (the covariant derivative ∇v)
    tang = np.einsum("cd,dp,ap->ac", ginv, di, AD)
    np.testing.assert_allclose(tang, np.einsum("cab,b->ca", gamma, vv).T, atol=1e-10)
