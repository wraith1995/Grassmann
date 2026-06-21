"""All-pairs conversions among the bases of each **four-fundamental-subspace** (`Img`, `Ker`,
`CoImg`, `CoKer`) — the FFS coverage the node-level closure matrix can't express (it has a
single `ProjectedBasis` node, exercised only as `image`).

For every selector `which ∈ {image, ker, coimg, coker}` we fix a map whose subspace is
non-trivial and build its valid bases — the orthonormal `ProjectedBasis`, a metric-twisted
`ProjectedBasis` (source/target ⇒ GSVD), a term `ListBasis` spanning the subspace, and (for
the image) the literal `ImageBasis` with/without a source — then convert **every ordered pair
in both directions** and assert each round-trips with its inverse.

The spanning `ListBasis` is **FFS-typed** (its vectors are `k`-dim, living in the subspace's
OWN coordinates — `C.img(f)`/`C.ker(f)`/`C.coimg(f)`/`C.coker(f)`), matching the in-space
`realize_basis_cols` contract; FFS realize is square (`k×k`) and every cob stays square.
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B

_RM = sys.modules["grassmann.lower.represent"]


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _invertible(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n)), -1) + np.diag(2.0 + np.arange(n))


def _ctx_run(build, feeds):
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", build(ctx))
    return np.asarray(prog.run(feeds).get("M"), float)


def _realize(basis, feeds):
    return _ctx_run(lambda ctx: _RM.realize_basis_cols(ctx, basis), feeds)


def _cob(a, b, feeds):
    return _ctx_run(lambda ctx: _RM._basis_cob_matrix(ctx, a, b), feeds)


def _listb(space, name, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    return B.list_basis(*vs), {_k(v): cols[:, i] for i, v in enumerate(vs)}


# (which, dom, cod, stored-map-feed-builder) — the subspace lives in dom (ker/coimg) or cod
# (image/coker); pick a map so it is a proper, non-trivial subspace.
_FFS = [
    ("image", 2, 3),   # injective ℝ²↪ℝ³  ⇒ image dim 2 in cod
    ("coker", 2, 3),   # same map         ⇒ coker dim 1 in cod
    ("ker", 3, 2),     # surjective ℝ³↠ℝ²  ⇒ ker dim 1 in dom
    ("coimg", 3, 2),   # same map         ⇒ coimg dim 2 in dom
]


_FFS_SPACE = {"image": C.img, "coker": C.coker, "ker": C.ker, "coimg": C.coimg}


def _ffs_bases(which: str, dom: int, cod: int):
    """The valid bases of the ``which`` subspace + a shared feed dict."""
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var(f"F_{which}", C.linear_map_space(Rt, Re))
    Jm = _invertible(max(dom, cod))[:cod, :dom]   # full-rank cod×dom
    feeds = {_k(f): Jm.T}

    projb = B.projected_basis(f, which)
    cols = _realize(projb, feeds)                 # k×k in-space columns (= identity)
    n, k = cols.shape                             # in-space realize is square ⇒ n == k
    ambient = Re if which in ("image", "coker") else Rt
    # A term basis spanning the same subspace — FFS-typed: its vectors are k-dim, living in
    # the subspace's OWN coords (``C.img(f)``/…), so the ListBasis realizes in-space (k×k).
    sub_space = _FFS_SPACE[which](f)
    listb, fl = _listb(sub_space, f"t_{which}", _invertible(k))
    # A metric-twisted ProjectedBasis (source on the dom side, target on the cod side); the
    # metric basis is an AMBIENT basis (it metricizes the dom/cod, not the subspace).
    side = "target" if which in ("image", "coker") else "source"
    amb_dim = cod if which in ("image", "coker") else dom
    mb, fm = _listb(ambient, f"m_{which}", _invertible(amb_dim))
    pmetric = B.projected_basis(f, which, **{side: mb})

    bases = [("ProjectedBasis", projb, {}),
             ("ListBasis", listb, fl)]
    # A metric only re-orthonormalizes the *metric-independent* subspaces (image = range,
    # ker = null space); for the complements (coimg = ker^⊥, coker = image^⊥) a different
    # metric spans a DIFFERENT subspace, so it is not the same basis to convert against.
    if which in ("image", "ker"):
        bases.append(("ProjectedBasis(metric)", pmetric, fm))
    if which == "image":
        src, fs = _listb(Rt, f"s_{which}", _invertible(dom))
        bases += [("ImageBasis", B.image_basis(f), {}),
                  ("ImageBasis(source)", B.image_basis(f, source=src), fs)]
    return bases, feeds


def _pairs():
    out = []
    for which, dom, cod in _FFS:
        bases, base_feeds = _ffs_bases(which, dom, cod)
        for ni, bi, fi in bases:
            for nj, bj, fj in bases:
                out.append((which, ni, bi, nj, bj, {**base_feeds, **fi, **fj}))
    return out


_PAIRS = _pairs()


@pytest.mark.parametrize("which,na,ba,nb,bb,feeds", _PAIRS,
                         ids=[f"{w}:{a}->{b}" for (w, a, ba, b, bb, f) in _PAIRS])
def test_ffs_subspace_basis_conversions(which, na, ba, nb, bb, feeds) -> None:
    """Within an FFS subspace, every basis converts to every other, both directions, invertibly."""
    m = _cob(ba, bb, feeds)
    back = _cob(bb, ba, feeds)
    n = m.shape[0]
    assert np.all(np.isfinite(m))
    assert m @ back == pytest.approx(np.eye(n)), f"{which}: {na}->{nb} not invertible"
