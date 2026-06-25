"""Analytic-manifold (Stiefel/Grassmann geodesic) reference tests for the gauge-aware moving-subspace
AD (:mod:`grassmann.transform.diff`), for the FULL FFS matrix.

Instead of validating ``diff`` against finite differences (as ``test_diff.py`` does), these tests
validate it against a KNOWN geometry with a CLOSED-FORM derivative at all orders.  The realization
frame ``Q(x)`` of a moving fundamental subspace is forced to be a GEODESIC of the Stiefel manifold:
with ``A`` a fixed ambient skew generator and ``θ(x)=c·x`` (``c`` a fixed covector on the diff space
``Rt``),

    Q(x) = expm(θ(x)·A) · Q₀ ,   so   ∂ⁿQ · h^⊗ⁿ = (c·h)ⁿ · Aⁿ · Q(x)   exactly, ∀n.

We build the immersion ``J(x)`` so that ``represent``'s realized frame IS that rotating ``Q(x)``
(rotating the SVD/QR factor that ``represent`` lowers).  Then for a CONSTANT subspace section ``v``,

    ∂ⁿ(embed v) · h^⊗ⁿ = (c·h)ⁿ · Aⁿ · Q₀ · v ,
    ∂ⁿ(project z) · h^⊗ⁿ = (c·h)ⁿ · (Aⁿ·Q₀)ᵀ · z   (the co-frame side, project = Qᵀ z),

are EXACT analytic references — no finite differences on the reference side.  The AD output uses the
NUMERICAL gauge feed (:func:`ffs_gauge_feed`); the residual against the exact reference therefore
measures the feed's own accuracy (machine-tight where the feed's ∂Q is analytic, nested-FD-limited at
higher order).

WHICH FFS frames admit this reference.  The geodesic trick pins the frame only when the relevant
columns are pinned by DISTINCT NONZERO singular values (so the SVD/QR basis tracks ``expm(A)·Q₀``
smoothly and sign-stably):

  * ``Img(J)`` injective   — frame ``qr_pos(Jᵀ)`` (sign-clean QR), rotate ``J(x)ᵀ=expm·Q₀·R₀``;
  * ``Img(J)`` rank-r      — frame ``U[:,:r]`` (leading left-singular block), rotate ``U(x)=expm·U₀``;
  * ``CoImg(J)``           — frame ``V[:,:r]`` (leading right-singular block), rotate ``V(x)=expm·V₀``.

``Ker(J)`` (frame ``V[:,r:]``) and ``CoKer(J)`` (frame ``U[:,r:]``) realize in the COMPLEMENT block,
whose singular values are all zero — a DEGENERATE block whose orthonormal basis numpy's SVD picks
arbitrarily.  Rotating ``J`` rigidly moves that complement SUBSPACE rigidly (its projector tracks
``expm(A)·Pₖ·expm(A)ᵀ`` exactly), but the BASIS within it does not track ``expm(A)·Q₀`` — so
``embed``/``project`` (basis-dependent) have no closed-form geodesic reference there.  ``Ker``/``CoKer``
stay covered by the FD matrix in ``test_diff.py``; :func:`test_complement_frame_basis_is_svd_arbitrary`
documents the obstruction here (the subspace rotates rigidly; the frame basis does not).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.transform.diff import (
    clear_subspace_gauges,
    diff,
    ffs_gauge_feed,
    set_subspace_gauge,
)

# A small generator scale keeps the SVD ordering/sign stable across the geodesic arc and the feed's
# tiny FD steps (no singular-value crossing, no sign flip).  Injective Img uses sign-clean QR and is
# unconditionally stable; the SVD cases (rank-r Img, CoImg) use this scale plus a fixed seed VERIFIED
# stable below (each world ASSERTS represent's frame == the geodesic Q(x) at sample points).
_GEN_SCALE = 0.3


def _qr_pos(M: np.ndarray) -> np.ndarray:
    """Positive-diagonal QR factor of ``M`` — the deterministic orthonormal frame ``represent`` lowers
    for an injective ``Img`` (== grassmann ``ProjectedBasis``)."""
    q, r = np.linalg.qr(M)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s


def _represent_frame(sub, J, Jv, amb, amb_dim, delta):
    """``represent``'s realized orthonormal frame of the moving subspace ``sub`` at immersion ``Jv``
    (ambient ``amb``, dim ``amb_dim``, subspace dim ``delta``) — the ``(amb_dim, delta)`` matrix whose
    columns are the ambient images of the subspace basis vectors.  Probed numerically by embedding each
    basis vector (exactly as the FFS probes do)."""
    from grassmann.lower.represent import represent

    v = T.var("v", sub)
    w = T.var("w", S.dual(amb))
    e = T.Eval(w, (0,), (T.embed(v, amb),))
    cols = []
    for i in range(delta):
        vv = np.zeros(delta)
        vv[i] = 1.0
        col = [
            float(
                np.ravel(
                    represent(e).run(
                        {
                            f"J_{J.uid}": Jv,
                            f"v_{v.uid}": vv,
                            f"w_{w.uid}": np.eye(amb_dim)[j],
                        }
                    )["result"]
                )[0]
            )
            for j in range(amb_dim)
        ]
        cols.append(col)
    return np.array(cols).T  # (amb_dim, delta)


# kind -> (Rt_dim, Re_dim, rank, δ, ambient side, U-side?, props builder).
#   U-side (rotate U): rank-r Img (frame U[:,:r]), ambient = cod.
#   V-side (rotate V): CoImg     (frame V[:,:r]), ambient = dom.
#   img_inj is special-cased (QR, rotate the whole orthonormal Q₀).
_ANALYTIC_CASES = {
    "img_inj": dict(Rtd=2, Red=3, rank=2, delta=2, amb="cod", side="U", props="inj"),
    "img_rankr": dict(Rtd=3, Red=4, rank=2, delta=2, amb="cod", side="U", props="rank"),
    "coimg": dict(Rtd=4, Red=2, rank=2, delta=2, amb="dom", side="V", props="surj"),
}
# Fixed seeds VERIFIED (below, and asserted in every world) to give a sign/order-stable geodesic frame
# for the SVD cases at ``_GEN_SCALE``.  img_inj (QR) is stable for any seed.
_CASE_SEED = {"img_inj": 0, "img_rankr": 1, "coimg": 0}


def _geodesic_world(kind: str, seed: int):
    """A moving FFS subspace whose realized frame is the EXACT Stiefel geodesic ``Q(x)=expm((c·x)A)·Q₀``.

    Returns the term scaffolding plus ``A`` (ambient skew generator), ``c`` (covector on ``Rt``), ``Q₀``
    (``represent``'s realized frame at ``x₀=0``, ``(amb_dim, δ)``), and ``Jval`` (``x ↦`` grassmann
    ``(dom, cod)`` immersion).  ASSERTS ``represent``'s frame == ``Q(x)`` at sample ``x`` (no SVD sign
    flip / no singular-value crossing) so the analytic reference is valid."""
    cfg = _ANALYTIC_CASES[kind]
    Rtd, Red, rank, delta = cfg["Rtd"], cfg["Red"], cfg["rank"], cfg["delta"]
    rng = np.random.default_rng(seed)
    Rt, Re = S.base("Rt", Rtd), S.base("Re", Red)

    if kind == "img_inj":
        # Injective Img: J(x)ᵀ = expm((c·x)A)·Q₀·R₀ ⇒ qr_pos(Jᵀ) = expm·Q₀ EXACTLY (sign-clean QR).
        A = rng.standard_normal((Red, Red))
        A = A - A.T
        Q0a = _qr_pos(rng.standard_normal((Red, delta)))
        R0 = np.triu(rng.standard_normal((delta, delta)))
        R0[np.diag_indices(delta)] = np.abs(np.diag(R0)) + 1.0
        c = rng.standard_normal(Rtd)

        def Jval(xv):
            return (expm((c @ xv) * A) @ Q0a @ R0).T

        props = G.inj
        ctor = S.img
    elif kind == "img_rankr":
        # Rank-r Img: rotate the LEFT-singular frame U(x)=expm((c·x)A)·U₀; Σ (distinct, rank r), V₀ fixed.
        # Frame = U(x)[:,:r] = expm·U₀[:,:r] (left-mult commutes with column slicing).
        A = (rng.standard_normal((Red, Red)))
        A = (A - A.T) * _GEN_SCALE
        U0 = _qr_pos(rng.standard_normal((Red, Red)))
        V0 = _qr_pos(rng.standard_normal((Rtd, Rtd)))
        sig = np.linspace(3.0, 1.5, rank)  # distinct nonzero ⇒ leading block pinned
        c = rng.standard_normal(Rtd)

        def Jval(xv):
            Sig = np.zeros((Red, Rtd))
            for i in range(rank):
                Sig[i, i] = sig[i]
            return (expm((c @ xv) * A) @ U0 @ Sig @ V0.T).T

        props = G.rank(rank)
        ctor = S.img
    else:  # coimg
        # CoImg: rotate the RIGHT-singular frame V(x)=expm((c·x)A)·V₀; Σ, U₀ fixed.  Frame = V(x)[:,:r].
        A = rng.standard_normal((Rtd, Rtd))
        A = (A - A.T) * _GEN_SCALE
        U0 = _qr_pos(rng.standard_normal((Red, Red)))
        V0 = _qr_pos(rng.standard_normal((Rtd, Rtd)))
        sig = np.linspace(3.0, 1.5, rank)
        c = rng.standard_normal(Rtd)

        def Jval(xv):
            Sig = np.zeros((Red, Rtd))
            for i in range(rank):
                Sig[i, i] = sig[i]
            return (U0 @ Sig @ (expm((c @ xv) * A) @ V0).T).T

        props = G.surj
        ctor = S.coimg

    J = T.var("J", S.oprod(S.dual(Rt), Re), props=props)
    sub = ctor(J)
    amb = Re if cfg["amb"] == "cod" else Rt
    amb_dim = Red if cfg["amb"] == "cod" else Rtd
    Q0 = _represent_frame(sub, J, Jval(np.zeros(Rtd)), amb, amb_dim, delta)

    # SANITY: represent's frame must equal the geodesic Q(x)=expm((c·x)A)·Q₀ at sample x (sign/order
    # stable).  If this ever fails (an SVD sign flip or singular-value crossing), the analytic reference
    # is invalid for that seed — assert rather than silently comparing against a flipped frame.
    chk = np.random.default_rng(seed + 1)
    for _ in range(5):
        xv = chk.standard_normal(Rtd) * 0.3
        Fx = _represent_frame(sub, J, Jval(xv), amb, amb_dim, delta)
        Qf = expm((c @ xv) * A) @ Q0
        assert np.allclose(Fx, Qf, atol=1e-8), (
            f"{kind}: represent frame != geodesic Q(x) (sign flip / crossing); "
            f"max |Δ| = {np.max(np.abs(Fx - Qf)):.2e}"
        )

    return dict(
        kind=kind, Rt=Rt, Re=Re, J=J, sub=sub, amb=amb, amb_dim=amb_dim,
        delta=delta, Rtd=Rtd, Red=Red, A=A, c=c, Q0=Q0, Jval=Jval,
    )


def _jval_tower(world, order):
    """The EXACT immersion-derivative tower ``[J, ∂J, …, ∂^order J]`` in grassmann ``(dom, cod)`` layout,
    leading ``|X|`` axes first (``∂ᵐJ`` shape ``(|X|,)*m + (dom, cod)``).

    For the U-side (Img) cases ``J(x)ᵀ = expm((c·x)A)·B`` (``B = J(0)ᵀ``) ⇒ ``∂ᵐ(Jᵀ) = (c⊗ᵐ)·Aᵐ·B``.
    For CoImg ``J(x)ᵀ = C·expm(-(c·x)A)`` (``C = J(0)ᵀ``) ⇒ ``∂ᵐ(Jᵀ) = (c⊗ᵐ)·C·(−A)ᵐ``.  No FD."""
    A, c, Rtd = world["A"], world["c"], world["Rtd"]
    base = world["Jval"](np.zeros(Rtd)).T  # cod × dom, the fixed factor (expm(0)=I)
    u_side = world["kind"] in ("img_inj", "img_rankr")
    out, Am = [], np.eye(A.shape[0])
    for deg in range(order + 1):
        Jm_deg = (Am @ base) if u_side else (base @ (Am * ((-1.0) ** deg)))  # cod × dom
        cprod = np.array(1.0)
        for _ in range(deg):
            cprod = np.multiply.outer(cprod, c)  # c⊗ᵐ over the |X| axes
        out.append(np.multiply.outer(cprod, Jm_deg.T))  # (|X|,)*deg + (dom, cod)
        Am = Am @ A
    return out


def _analytic_reference(world, order, arm, h, v_or_z, covec):
    """The EXACT analytic ``∂ⁿ⟨covec, op(·)⟩ · h^⊗ⁿ`` for a CONSTANT section/value:
    embed ⇒ ``covecᵀ·(c·h)ⁿ·Aⁿ·Q₀·v``; project ⇒ ``covecᵀ·(c·h)ⁿ·(Aⁿ·Q₀)ᵀ·z``."""
    ch = world["c"] @ h
    An = np.linalg.matrix_power(world["A"], order)
    Q0 = world["Q0"]
    if arm == "embed":
        return float(covec @ ((ch ** order) * (An @ Q0 @ v_or_z)))
    return float(covec @ ((ch ** order) * ((An @ Q0).T @ v_or_z)))


def _ad_value(world, order, arm, rng):
    """Run the gauge-aware AD of ``∂ⁿ⟨covec, op(const)⟩`` (op = embed/project) through the geodesic
    moving subspace, with the NUMERICAL :func:`ffs_gauge_feed` tower, and return ``(ad_scalar, analytic
    reference)`` evaluated along a single direction ``h`` repeated ``order`` times."""
    from grassmann.lower.represent import represent

    Rt, Re, J, sub = world["Rt"], world["Re"], world["J"], world["sub"]
    amb, amb_dim, delta, Rtd = world["amb"], world["amb_dim"], world["delta"], world["Rtd"]
    tower = _jval_tower(world, order)
    DJ_names = ["DJ"] + [f"D{m}J" for m in range(2, order + 1)]
    DJ_vars = [
        T.var(DJ_names[m - 1], S.oprod(*([S.dual(Rt)] * (m + 1)), Re)) for m in range(1, order + 1)
    ]
    x = T.var("x", Rt)
    feed = {f"J_{J.uid}": tower[0]}
    for m, dv in enumerate(DJ_vars, start=1):
        feed[f"{dv.name}_{dv.uid}"] = tower[m]

    if arm == "embed":
        val = T.var("v", sub)
        covar = T.var("w", S.dual(amb))
        e = T.Eval(covar, (0,), (T.embed(val, amb),))
        valv = rng.standard_normal(delta)
        covecv = rng.standard_normal(amb_dim)
        # CONSTANT section ⇒ every section derivative is zero (the geodesic motion is all in the frame).
        Dval = [
            T.var(f"Dv{m}" if m > 1 else "Dv", S.oprod(*([S.dual(Rt)] * m), sub))
            for m in range(1, order + 1)
        ]
        feed[f"v_{val.uid}"] = valv
        feed[f"w_{covar.uid}"] = covecv
        for m, dv in enumerate(Dval, start=1):
            feed[f"{dv.name}_{dv.uid}"] = np.zeros((Rtd,) * m + (delta,))
    else:
        val = T.var("z", amb)
        covar = T.var("wc", S.dual(sub))
        e = T.Eval(covar, (0,), (T.project(val, sub),))
        valv = rng.standard_normal(amb_dim)
        covecv = rng.standard_normal(delta)
        Dval = [
            T.var(f"Dz{m}" if m > 1 else "Dz", S.oprod(*([S.dual(Rt)] * m), amb))
            for m in range(1, order + 1)
        ]
        feed[f"z_{val.uid}"] = valv
        feed[f"wc_{covar.uid}"] = covecv
        for m, dv in enumerate(Dval, start=1):
            feed[f"{dv.name}_{dv.uid}"] = np.zeros((Rtd,) * m + (amb_dim,))

    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    try:
        cur = e
        for d in range(order):
            deps = {val.uid: Dval[0], J.uid: DJ_vars[0]}
            for k in range(1, d + 1):
                deps[Dval[k - 1].uid] = Dval[k]
                deps[DJ_vars[k - 1].uid] = DJ_vars[k]
            cur = diff(cur, x, deps)
        extra = ffs_gauge_feed(sub, DJ_vars, feed)
    finally:
        clear_subspace_gauges()

    # Peel the nested LLam binders; seed every direction with the same h.
    h = rng.standard_normal(Rtd)
    node, seeds = cur, []
    for _ in range(order):
        seeds.append(node.vars[0])
        node = node.body
    run_feed = {**feed, **extra}
    for sd in seeds:
        run_feed[f"{sd.name}_{sd.uid}"] = h
    ad = float(np.ravel(np.asarray(represent(node).run(run_feed)["result"], float))[0])
    ref = _analytic_reference(world, order, arm, h, valv, covecv)
    return ad, ref


# The analytic matrix (case × order × arm).  Tolerances bound the ACTUAL measured worst-case residual
# class (over many random h / covector directions, both index modes) — they are NOT loosened below what
# the numerical feed achieves, only set to its real accuracy:
#   order 1 — the feed's ∂Q IS the analytic geodesic connection ⇒ MACHINE precision (measured ≲5e-11);
#   order 2 — one level of nested Richardson FD in the feed ⇒ measured ≲1e-8 (SVD cases worse than QR);
#   order 3 — two levels of nested FD ⇒ measured ≲6e-7.
# A snug margin above the measured worst case keeps the test a real check of the feed's accuracy.
_ORDER_TOL = {1: 1e-9, 2: 1e-7, 3: 1e-5}


@pytest.mark.parametrize("kind", list(_ANALYTIC_CASES))
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("arm", ["embed", "project"])
def test_geodesic_frame_ad_matches_analytic(kind, order, arm) -> None:
    # AD (numerical ffs_gauge_feed) of ∂ⁿ⟨covec, op(const)⟩ through a GEODESIC moving FFS subspace vs the
    # EXACT closed form (c·h)ⁿ·Aⁿ·Q₀ — no finite differences on the reference side.  Covers the three FFS
    # frames pinned by distinct nonzero singular values (Img injective / rank-r, CoImg), both arms, orders
    # 1–3.  Residual tracks the feed's nested-FD depth (machine-tight at order 1).
    world = _geodesic_world(kind, _CASE_SEED[kind])
    rng = np.random.default_rng(2025 + order * 31 + len(kind) * 7 + (arm == "project"))
    ad, ref = _ad_value(world, order, arm, rng)
    assert abs(ad - ref) < _ORDER_TOL[order], (
        f"{kind} order {order} {arm}: AD {ad:.12g} vs analytic {ref:.12g}, "
        f"resid {abs(ad - ref):.3e} > tol {_ORDER_TOL[order]:.0e}"
    )


@pytest.mark.parametrize("kind", list(_ANALYTIC_CASES))
def test_feed_dQ_matches_analytic_connection(kind) -> None:
    # Direct feed-vs-truth: the FIRST tower value ∂Q from ffs_gauge_feed must equal the analytic geodesic
    # connection (c·h)·A·Q₀ — a clean check that the numerical feed reproduces the exact Stiefel
    # connection (machine precision; this is what makes order-1 AD machine-tight above).
    from grassmann.transform.diff import (
        _QR_AUTO_VARS,
        _QR_NEXT,
        _ffs_kind,
        _subspace_defining_map,
    )

    world = _geodesic_world(kind, _CASE_SEED[kind])
    Rt, Re, J, sub = world["Rt"], world["Re"], world["J"], world["sub"]
    amb, amb_dim, delta, Rtd = world["amb"], world["amb_dim"], world["delta"], world["Rtd"]
    A, c, Q0 = world["A"], world["c"], world["Q0"]
    tower = _jval_tower(world, 1)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    feed = {f"J_{J.uid}": tower[0], f"DJ_{DJ.uid}": tower[1]}
    # A throwaway embed diff so the auto QR tower vars get created, then read the feed's Q₁ value.
    v = T.var("v", sub)
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sub))
    w = T.var("w", S.dual(amb))
    e = T.Eval(w, (0,), (T.embed(v, amb),))
    feed[f"v_{v.uid}"] = np.zeros(delta)
    feed[f"w_{w.uid}"] = np.zeros(amb_dim)
    feed[f"Dv_{Dv.uid}"] = np.zeros((Rtd, delta))
    clear_subspace_gauges()
    set_subspace_gauge(J, "qr")
    try:
        diff(e, T.var("x", Rt), deps={v.uid: Dv, J.uid: DJ})
        extra = ffs_gauge_feed(sub, [DJ], feed)
        kind_str = _ffs_kind(sub)
        Jmap = _subspace_defining_map(sub)
        Q0var = _QR_AUTO_VARS[(Jmap.uid, kind_str)]
        Q1var = _QR_NEXT[Q0var.uid]
        Q1 = np.asarray(extra[f"{Q1var.name}_{Q1var.uid}"], float).reshape(Rtd, delta, amb_dim)
    finally:
        clear_subspace_gauges()
    h = np.random.default_rng(_CASE_SEED[kind]).standard_normal(Rtd)
    feed_dQ = np.tensordot(h, Q1, axes=([0], [0]))  # (δ, ambient) directional ∂Q·h
    analytic_dQ = ((c @ h) * (A @ Q0)).T  # (c·h)·A·Q₀ in (δ, ambient) layout
    np.testing.assert_allclose(feed_dQ, analytic_dQ, atol=1e-9)


def test_complement_frame_basis_is_svd_arbitrary() -> None:
    # WHY Ker/CoKer have no geodesic reference (honest negative result).  Rotating J rigidly moves the
    # COMPLEMENT subspace (Ker frame = V[:,r:]) rigidly — its orthogonal PROJECTOR tracks the geodesic
    # exactly — but the complement's singular values are all ZERO (degenerate block), so numpy's SVD
    # picks an ARBITRARY orthonormal basis within it that does NOT track expm(A)·Q₀.  Hence embed/project
    # (basis-dependent) cannot be referenced against (c·h)ⁿ·Aⁿ·Q₀ there; Ker/CoKer stay FD-covered in
    # test_diff.py.  This asserts BOTH facts: subspace (projector) rotates rigidly, frame basis does not.
    Rtd, Red, rank, delta = 4, 2, 2, 2
    rng = np.random.default_rng(31)
    Rt, Re = S.base("Rt", Rtd), S.base("Re", Red)
    A = rng.standard_normal((Rtd, Rtd))
    A = (A - A.T) * _GEN_SCALE
    U0 = _qr_pos(rng.standard_normal((Red, Red)))
    V0 = _qr_pos(rng.standard_normal((Rtd, Rtd)))
    sig = np.linspace(3.0, 1.5, rank)
    c = rng.standard_normal(Rtd)

    def Jval(xv):
        Sig = np.zeros((Red, Rtd))
        for i in range(rank):
            Sig[i, i] = sig[i]
        return (U0 @ Sig @ (expm((c @ xv) * A) @ V0).T).T

    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.surj)
    sub = S.ker(J)  # Ker frame = V[:,r:] (the degenerate complement block)
    Q0 = _represent_frame(sub, J, Jval(np.zeros(Rtd)), Rt, Rtd, delta)
    subspace_rigid = True
    basis_geodesic = True
    for _ in range(4):
        xv = rng.standard_normal(Rtd) * 0.3
        Fx = _represent_frame(sub, J, Jval(xv), Rt, Rtd, delta)
        Qf = expm((c @ xv) * A) @ Q0
        if not np.allclose(Fx @ Fx.T, Qf @ Qf.T, atol=1e-8):  # projectors == subspace as a set
            subspace_rigid = False
        if not np.allclose(Fx, Qf, atol=1e-7):  # frame basis tracks the geodesic?
            basis_geodesic = False
    assert subspace_rigid, "Ker subspace projector should rotate rigidly with the geodesic"
    assert not basis_geodesic, (
        "Ker frame basis unexpectedly tracked expm(A)·Q₀ — if this holds, a geodesic embed/project "
        "reference for Ker/CoKer is achievable and should be added"
    )
