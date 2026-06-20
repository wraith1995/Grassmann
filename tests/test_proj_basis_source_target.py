"""`ProjectedBasis`/`ImageBasis` source & target threaded through cob (Stage B).

`source`/`target` (bases of `f`'s domain/codomain) make the realized FFS basis orthonormal
w.r.t. those bases' inner products: each is turned into the GSVD metric `G = (B·Bᵀ)⁻¹`, with
columns kept in the standard ambient frame so `cob` composes.  `ImageBasis(f, source)` maps
the source basis (`f·S`).
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann as G
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B

_RM = sys.modules["grassmann.lower.represent"]
_JMATH = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
_JSTORED = _JMATH.T


def _k(v) -> str:
    return f"{v.node.name}_{v.node.uid}"


def _realize(basis, feeds):
    prog = pa.Program("r")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("C", _RM.realize_basis_cols(ctx, basis))
    return np.asarray(prog.run(feeds)["C"], float)


def _cob(a, b, feeds):
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", _RM._basis_cob_matrix(ctx, a, b))
    return np.asarray(prog.run(feeds)["M"], float)


def _img():
    Rt, Re = G.space("Rt", 2), G.space("Re", 3)
    J = G.var("J", Rt >> Re, props=G.inj)
    return J, Re


def test_projected_basis_target_is_metric_orthonormal() -> None:
    """`ProjectedBasis(f,"image",target=T)` columns are orthonormal in `T`'s metric `(TTᵀ)⁻¹`."""
    J, Re = _img()
    t = [G.var(f"t{i}", Re) for i in range(3)]
    Tm = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    pb = B.ProjectedBasis(J.node, "image", target=B.list_basis(*[ti.node for ti in t]))
    feeds = {_k(J): _JSTORED, **{_k(t[i]): Tm[:, i] for i in range(3)}}
    C = _realize(pb, feeds)
    g_t = np.linalg.inv(Tm @ Tm.T)
    assert C.T @ g_t @ C == pytest.approx(np.eye(2))  # orthonormal in the target metric
    # and it spans the SAME image (range of J).
    assert np.linalg.matrix_rank(np.hstack([C, _JMATH])) == 2


def test_projected_basis_target_default_is_plain_q() -> None:
    """`target=None` is byte-identical to the plain orthonormal range `Q` (standard metric)."""
    J, _Re = _img()
    pb = B.ProjectedBasis(J.node, "image")
    C = _realize(pb, {_k(J): _JSTORED})
    assert C.T @ C == pytest.approx(np.eye(2))  # standard-orthonormal


def test_projected_basis_target_cob_roundtrips() -> None:
    """cob between the target- and standard-orthonormal ranges is an invertible pair."""
    J, Re = _img()
    t = [G.var(f"t{i}", Re) for i in range(3)]
    Tm = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    tgt = B.list_basis(*[ti.node for ti in t])
    pb_t = B.ProjectedBasis(J.node, "image", target=tgt)
    pb = B.ProjectedBasis(J.node, "image")
    feeds = {_k(J): _JSTORED, **{_k(t[i]): Tm[:, i] for i in range(3)}}
    m = _cob(pb_t, pb, feeds)
    back = _cob(pb, pb_t, feeds)
    assert m @ back == pytest.approx(np.eye(2))


def test_image_to_metric_projected_uses_gsvd_factor() -> None:
    """`cob(ImageBasis(f) → ProjectedBasis(f,"image",target=T))` exploits the GSVD factor
    (`Uᵀ·M_W·J`) — round-trips and maps coords to the same geometric vector."""
    J, Re = _img()
    t = [G.var(f"t{i}", Re) for i in range(3)]
    Tm = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    tgt = B.list_basis(*[ti.node for ti in t])
    imgb = B.ImageBasis(J.node)
    projb_t = B.ProjectedBasis(J.node, "image", target=tgt)
    feeds = {_k(J): _JSTORED, **{_k(t[i]): Tm[:, i] for i in range(3)}}
    m = _cob(imgb, projb_t, feeds)
    back = _cob(projb_t, imgb, feeds)
    assert m @ back == pytest.approx(np.eye(2))  # invertible pair
    u = _realize(projb_t, feeds)
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.standard_normal(2)
        assert u @ (m @ s) == pytest.approx(_JMATH @ s)  # U·c = J·s (same vector)


def test_image_basis_source_maps_the_source_basis() -> None:
    """`ImageBasis(f, source=S)` realizes `f` applied to `S` (`f·S`)."""
    Rt, Re = G.space("Rt", 2), G.space("Re", 3)
    J = G.var("J", Rt >> Re, props=G.inj)
    s0, s1 = G.var("s0", Rt), G.var("s1", Rt)
    Sm = np.array([[1.0, 0.0], [1.0, 1.0]])
    imgs = B.ImageBasis(J.node, source=B.list_basis(s0.node, s1.node))
    C = _realize(imgs, {_k(J): _JSTORED, _k(s0): Sm[:, 0], _k(s1): Sm[:, 1]})
    assert C == pytest.approx(_JMATH @ Sm)


def test_primal_dual_peeling_is_identity() -> None:
    """`PrimalBasis(DualBasis(b))` cancels to `b` ⇒ cob is the identity (variance peeling)."""
    J, _Re = _img()
    imgb = B.ImageBasis(J.node)
    pd = B.primal_basis(B.dual_basis(imgb))
    assert _cob(pd, imgb, {_k(J): _JSTORED}) == pytest.approx(np.eye(2))

