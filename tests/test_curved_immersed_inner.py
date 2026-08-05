"""Task-1 acceptance: differentiating the inner product of an x-dependent IMMERSED
tangent/cotangent space ``Img(J(x))`` keeps the metric *riding free*.

The pointwise FEM backend lives a value in an immersed tangent space ``Img(J)`` (cotangent =
``dual(Img(J))``), ``J : Rᵗ ⊸ Rᵉ`` the (curved) chart Jacobian.  The immersed metric is implied
by ``J`` — it is NEVER handed to ``diff`` as an array or a markable metric ``Var``.  ``represent``
realizes ``Img(J)`` in the orthonormal ``ProjectedBasis = qr_pos(Jᵀ)``, so ``Inner(Img(J))`` lowers
to the IDENTITY in that frame (the immersed metric rides entirely in the frame, not the operator —
cf. ``test_metric_seam`` / the represent probes).  Two consequences this file pins:

* The **full-contraction** inner ``⟨u,v⟩`` over ``Img(J(x))`` differentiates correctly today, and
  its derivative does NOT depend on the chart's motion ``∂J``: the Levi-Civita connection of the
  realization frame is skew, so the two connection terms ``⟨Γu,v⟩ + ⟨u,Γv⟩`` cancel (metric
  compatibility).  Hence ``diff`` may treat ``Inner(Img(J))`` as constant — and the result matches
  a finite difference of the realized inner to machine precision, on a genuinely curved ``J``.

* The metric is **never materialized**: the lowered derivative program references neither ``∂J``
  (``DJ``) nor any gram ``JᵀJ`` / ``Sqrt`` / ``Pinv(JᵀJ)``.  ``J`` itself appears only to realize
  the coordinate slots of the supplied derivative maps ``Du``/``Dv`` (the operand spaces are
  ``Img(J)``), not to build a metric.

The complementary cases — where ``∂J`` GENUINELY enters because the differentiated object is not a
full contraction (``embed`` / ``project`` of an immersed value, the covariant derivative of a tangent
VECTOR field) — are covered by ``test_diff.py`` (``test_parameter_frame_immersion_ad_and_split``,
``test_frame_connection_matches_metric_connection``, ``test_gauss_decomposition_of_embed``) and
``test_diff_analytic.py`` (``test_covariant_derivative_via_christoffel``).  There the connection is
``Γ(h) = J⁺·(∂J·h)`` (the frame connection, ``diff.py:_frame_connection_map``) — still metric-free
(no ``JᵀJ`` built, only ``J``, ``J⁺``, ``∂J``).
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.lower.represent import represent
from grassmann.transform.diff import clear_subspace_gauges, diff


def _run(term: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(represent(term).run(feed)["result"], dtype=float)


def _var_names(node: object) -> set[str]:
    """All ``Var`` names referenced anywhere in a term (walks dataclass children)."""
    from dataclasses import fields, is_dataclass

    out: set[str] = set()

    def walk(n: object) -> None:
        if isinstance(n, T.Var):
            out.add(n.name)
        if is_dataclass(n):
            for f in fields(n):
                val = getattr(n, f.name)
                for c in (val if isinstance(val, (tuple, list)) else (val,)):
                    walk(c)

    walk(node)
    return out


@pytest.mark.parametrize("kind", ["tangent", "cotangent"])
def test_immersed_inner_differentiates_metric_free(kind: str) -> None:
    """``diff(⟨u,v⟩_{Img(J(x))}, x)`` on a CURVED chart matches FD, and the lowered derivative
    never mentions ``∂J`` (``DJ``) or a gram — the immersed metric rides free."""
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    sp = imgJ if kind == "tangent" else S.dual(imgJ)

    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    u, v = T.var("u", sp), T.var("v", sp)
    Du = T.var("Du", S.oprod(S.dual(Rt), sp))
    Dv = T.var("Dv", S.oprod(S.dual(Rt), sp))

    e = G.dot(u, v).node  # ⟨u, v⟩ : Scalar, the immersed (co)tangent inner

    clear_subspace_gauges()
    jd = diff(e, x, deps={u.uid: Du, v.uid: Dv, J.uid: DJ})

    # (1) the metric rides free: the chart's MOTION never enters the derivative, nor any gram.
    names = _var_names(jd)
    assert "DJ" not in names, f"∂J leaked into the immersed-inner derivative: {sorted(names)}"
    body = repr(jd.body)
    assert "Sqrt" not in body and "Dagger" not in body, "a metric gram was materialized"

    # (2) numeric correctness on a genuinely curved J: diff == FD of the realized inner, where x
    #     enters through u, v (along Du, Dv) AND through the chart J (along DJ).
    rng = np.random.default_rng(20 + len(kind))
    Jv = rng.standard_normal((2, 3))
    DJv = rng.standard_normal((2, 2, 3))  # curved: ∂J ≠ 0
    uc, vc = rng.standard_normal(2), rng.standard_normal(2)
    Duv, Dvv = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))
    hv = rng.standard_normal(2)
    seed = jd.vars[0]
    feed = {
        f"J_{J.uid}": Jv, f"DJ_{DJ.uid}": DJv,
        f"u_{u.uid}": uc, f"v_{v.uid}": vc,
        f"Du_{Du.uid}": Duv, f"Dv_{Dv.uid}": Dvv,
    }
    got = float(_run(jd.body, {**feed, f"{seed.name}_{seed.uid}": hv}))

    duh = _run(T.Eval(Du, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    dvh = _run(T.Eval(Dv, (0,), (x,)), {**feed, f"x_{x.uid}": hv})
    dJh = np.einsum("aij,a->ij", DJv, hv)
    eps = 1e-6

    def ev(s: float) -> float:
        return float(_run(e, {
            f"J_{J.uid}": Jv + s * eps * dJh,
            f"u_{u.uid}": uc + s * eps * duh,
            f"v_{v.uid}": vc + s * eps * dvh,
        }))

    fd = (ev(1.0) - ev(-1.0)) / (2 * eps)
    np.testing.assert_allclose(got, fd, atol=1e-6)


def test_immersed_inner_affine_equals_curved() -> None:
    """The covariant connection cancels in the full contraction, so the immersed-inner derivative
    is identical whether the chart is affine (``∂J=0``) or curved (``∂J≠0``) — confirming Γ never
    contributes (and the affine case reduces to the flat one)."""
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    x = T.var("x", Rt)
    DJ = T.var("DJ", S.oprod(S.dual(Rt), S.dual(Rt), Re))
    u, v = T.var("u", imgJ), T.var("v", imgJ)
    Du = T.var("Du", S.oprod(S.dual(Rt), imgJ))
    Dv = T.var("Dv", S.oprod(S.dual(Rt), imgJ))

    e = G.dot(u, v).node
    clear_subspace_gauges()
    jd = diff(e, x, deps={u.uid: Du, v.uid: Dv, J.uid: DJ})
    seed = jd.vars[0]

    rng = np.random.default_rng(99)
    Jv = rng.standard_normal((2, 3))
    uc, vc = rng.standard_normal(2), rng.standard_normal(2)
    Duv, Dvv = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))
    hv = rng.standard_normal(2)
    base = {
        f"J_{J.uid}": Jv, f"u_{u.uid}": uc, f"v_{v.uid}": vc,
        f"Du_{Du.uid}": Duv, f"Dv_{Dv.uid}": Dvv, f"{seed.name}_{seed.uid}": hv,
    }
    affine = float(_run(jd.body, {**base, f"DJ_{DJ.uid}": np.zeros((2, 2, 3))}))
    curved = float(_run(jd.body, {**base, f"DJ_{DJ.uid}": rng.standard_normal((2, 2, 3))}))
    np.testing.assert_allclose(affine, curved, atol=1e-12)
