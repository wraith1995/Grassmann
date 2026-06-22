"""Probe 2: target-bearing ImageBasis via the GENERIC standard route
(ImageBasis(target) -> metric ProjectedBasis, and -> ListBasis), and the special
canonical arm (ImageBasis(target) -> metric-free ProjectedBasis('image'))."""
from __future__ import annotations
import sys
import numpy as np
import polyarray as pa
import grassmann.core as C
import grassmann.lower.represent  # noqa
from grassmann.core import basis as B

_RM = sys.modules["grassmann.lower.represent"]


def _k(v):
    return f"{v.name}_{v.uid}"


def _inv(n):
    return np.tril(np.ones((n, n)), -1) + np.diag(2.0 + np.arange(n))


def _ctx_run(build, feeds):
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", build(ctx))
    return np.asarray(prog.run(feeds).get("M"), float)


def _cob(a, b, feeds):
    return _ctx_run(lambda ctx: _RM._basis_cob_matrix(ctx, a, b), feeds)


def _listb(space, name, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    return B.list_basis(*vs), {_k(v): cols[:, i] for i, v in enumerate(vs)}


def main():
    dom, cod = 2, 3
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var("F", C.linear_map_space(Rt, Re))
    Jm = _inv(max(dom, cod))[:cod, :dom]
    feeds = {_k(f): Jm.T}
    t1, ft1 = _listb(Re, "t1", _inv(cod))                # ambient target basis
    mb, fm = _listb(Re, "m", _inv(cod) + np.eye(cod) * 0.3)  # ambient metric basis for proj
    allf = {**feeds, **ft1, **fm}

    img_t = B.image_basis(f, target=t1)        # target-bearing, no source
    img_n = B.image_basis(f)                    # no target

    # vs metric-free ProjectedBasis("image") -- the SPECIAL canonical arm
    proj = B.projected_basis(f, "image")
    for tag, img in [("img(target)", img_t), ("img(none)", img_n)]:
        m = _cob(img, proj, allf)
        back = _cob(proj, img, allf)
        rt = np.allclose(m @ back, np.eye(m.shape[0]))
        print(f"{tag} <-> proj(image): roundtrip={rt}")

    # target should not change the cob to proj
    m_t = _cob(img_t, proj, allf)
    m_n = _cob(img_n, proj, allf)
    print("target-independence (img->proj):", np.allclose(m_t, m_n))

    # vs metric ProjectedBasis -> GENERIC standard route
    pmetric = B.projected_basis(f, "image", target=mb)
    for tag, img in [("img(target)", img_t), ("img(none)", img_n)]:
        m = _cob(img, pmetric, allf)
        back = _cob(pmetric, img, allf)
        rt = np.allclose(m @ back, np.eye(m.shape[0]))
        print(f"{tag} <-> proj(metric): roundtrip={rt}")
    m_t = _cob(img_t, pmetric, allf)
    m_n = _cob(img_n, pmetric, allf)
    print("target-independence (img->proj_metric):", np.allclose(m_t, m_n))


if __name__ == "__main__":
    main()
