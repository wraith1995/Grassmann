"""Task-2 acceptance (plan 45): the TWO distinct converts pointwise needs, over a curved ``Img(J)``.

pointwise's ``convert`` is really two operations that must NOT be merged:

* **Frame convert** at a fixed point — world-tangent ``Img(J)`` ↔ world-euclidean ``Rᵉ`` — is
  ``Embed`` / ``Project``.  ``pullback`` (``transform/pullback.py``) builds exactly these: the
  realized maps are grassmann's FFS embed/project (orthonormal realization frame), NOT "apply J".
  Round-trips identity on the subspace and preserves the immersed inner.

* **Coordinate convert** — reference ``Rᵗ`` ↔ world tangent ``Img(J)`` along the chart — is a
  DIFFERENT operation: an explicit application of the chart Jacobian ``J`` (``Eval(J, ·)``) and its
  pseudo-inverse ``J⁺`` (``Eval(Pinv(J), ·)``).  It is correct on a curved chart (``J`` is a linear
  map at each sampled point; the curvature only matters when DIFFERENTIATING — see Task 1).

DESIGN BOUNDARY (pinned by ``test_pullback_is_not_the_coordinate_convert``): ``pullback`` does the
frame/structural convert only.  Asking it to transport ``Rᵗ → Img(J)`` does NOT apply ``J`` — it falls
to the dim-equal ``Assert`` pass-through (an honest coordinate *relabel*, a deliberate feature, cf.
``test_pullback.py::test_dim_equal_unrelated_passes_through``).  pointwise must build the coordinate
convert explicitly with ``J`` / ``J⁺``; it must not route it through ``pullback``.
"""

from __future__ import annotations

import numpy as np

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.infer.synth import synth_space
from grassmann.lower.represent import represent
from grassmann.transform.pullback import pullback


def _run(term: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(represent(term).run(feed)["result"], dtype=float)


def _chart():
    Rt, Re = S.base("Rt", 2), S.base("Re", 3)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    return Rt, Re, J, S.img(J)


def test_frame_convert_roundtrip_and_inner_preserved() -> None:
    """``Img(J) → Rᵉ → Img(J)`` is identity on the subspace, and the immersed inner survives the
    frame convert — on an arbitrary (curved-sample) chart."""
    Rt, Re, J, imgJ = _chart()
    u, v = T.var("u", imgJ), T.var("v", imgJ)

    embed_v = pullback(v, Re)            # Img(J) ↪ Rᵉ  (an Embed)
    assert isinstance(embed_v, T.Embed)
    back = pullback(embed_v, imgJ)       # Rᵉ ↠ Img(J)  (a Project)

    rng = np.random.default_rng(11)
    Jv = rng.standard_normal((2, 3))
    uc, vc = rng.standard_normal(2), rng.standard_normal(2)
    feed = {f"J_{J.uid}": Jv, f"u_{u.uid}": uc, f"v_{v.uid}": vc}

    # (1) embed∘project is identity on the subspace.
    np.testing.assert_allclose(_run(back, {f"J_{J.uid}": Jv, f"v_{v.uid}": vc}), vc, atol=1e-10)

    # (2) the immersed inner is preserved through the frame convert (orthonormal realization frame):
    #     ⟨embed u, embed v⟩_{Rᵉ} == ⟨u, v⟩_{Img(J)}.
    amb = _run(G.dot(pullback(u, Re), pullback(v, Re)).node, feed)
    sub = _run(G.dot(u, v).node, feed)
    np.testing.assert_allclose(amb, sub, atol=1e-10)


def test_coordinate_convert_via_chart_jacobian() -> None:
    """Reference ↔ world-tangent along the chart: ``J·w`` lands in ``Img(J)``, ``J⁺(J·w)==w``, and
    the ambient pushforward agrees with embedding its ``Img(J)``-frame coordinates — curved chart."""
    Rt, Re, J, imgJ = _chart()
    w = T.var("w", Rt)

    push_amb = T.Eval(J, (0,), (w,))                 # J·w : Rᵉ (the ambient pushforward)
    push_img = T.Project(push_amb, imgJ)             # its Img(J)-frame coordinates
    assert synth_space(push_img) == imgJ
    pull_ref = T.Eval(T.Pinv(J), (0,), (push_amb,))  # J⁺(J·w) : Rᵗ

    rng = np.random.default_rng(12)
    Jv = rng.standard_normal((2, 3))                 # arbitrary curved-sample chart
    wv = rng.standard_normal(2)
    feed = {f"J_{J.uid}": Jv, f"w_{w.uid}": wv}

    # (1) round-trip ref → world → ref is identity (J injective ⇒ J⁺J = I).
    np.testing.assert_allclose(_run(pull_ref, feed), wv, atol=1e-10)

    # (2) the two pictures agree: embed(project(J·w)) == J·w in the ambient.
    c = T.var("c", imgJ)
    cval = _run(push_img, feed)
    amb_from_c = _run(pullback(c, Re), {f"J_{J.uid}": Jv, f"c_{c.uid}": cval})
    np.testing.assert_allclose(amb_from_c, _run(push_amb, feed), atol=1e-10)


def test_pullback_is_not_the_coordinate_convert() -> None:
    """The design boundary: ``pullback(w:Rᵗ, Img(J))`` does NOT apply the chart ``J`` — it is the
    dim-equal ``Assert`` relabel, distinct from the coordinate pushforward ``J·w``."""
    Rt, Re, J, imgJ = _chart()
    w = T.var("w", Rt)

    relabel = pullback(w, imgJ)
    assert isinstance(relabel, T.Assert)  # dim-equal pass-through, not an application of J

    rng = np.random.default_rng(13)
    Jv = rng.standard_normal((2, 3))
    wv = rng.standard_normal(2)
    feed = {f"J_{J.uid}": Jv, f"w_{w.uid}": wv}

    # The relabel reinterprets w's coordinates as Img(J)-frame coords; the coordinate convert
    # instead projects the ambient pushforward J·w.  They differ (so the two must not be conflated).
    relabel_coords = _run(relabel, feed)
    pushforward_coords = _run(T.Project(T.Eval(J, (0,), (w,)), imgJ), feed)
    assert not np.allclose(relabel_coords, pushforward_coords)
