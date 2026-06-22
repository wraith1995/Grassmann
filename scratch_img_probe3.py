"""Probe 3: (a) Dual-wrapped ImageBasis as a cob operand;
(b) non-injective/over-determined-source ImageBasis WITH target (rectangular R)."""
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
    t1, ft1 = _listb(Re, "t1", _inv(cod))
    s1, fs1 = _listb(Rt, "s1", _inv(dom))
    allf = {**feeds, **ft1, **fs1}

    # (a) Dual-wrapped ImageBasis -> Dual-wrapped ImageBasis (same image, diff source)
    A = B.image_basis(f, source=s1, target=t1)
    Bn = B.image_basis(f)
    try:
        m = _cob(B.dual_basis(A), B.dual_basis(Bn), allf)
        back = _cob(B.dual_basis(Bn), B.dual_basis(A), allf)
        print("(a) Dual(img)->Dual(img) roundtrip:",
              np.allclose(m @ back, np.eye(m.shape[0])))
        # primal-cob should be inverse-transpose related
        mp = _cob(A, Bn, allf)
        print("(a) dual == (primal)^-T:",
              np.allclose(m, np.linalg.inv(mp).T))
    except Exception as e:
        print("(a) ERROR:", type(e).__name__, e)

    # (b) over-determined source: source has MORE columns than rank(image) -> rectangular R
    # dom=2 injective so image rank 2; give a source basis of dom (2 cols) - that's square.
    # To get over-determined we need a source with >2 cols, but source is a basis of dom=R^2
    # (max 2 indep). Instead: make f NON-injective (rank 1) so image rank=1 < source cols.
    g = C.var("G", C.linear_map_space(Rt, Re))
    Jg = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # rank-1 cod x dom
    gf = {_k(g): Jg.T}
    sg, fsg = _listb(Rt, "sg", _inv(dom))   # 2-col source, image rank 1 -> R is 1x2 (rect)
    tg, ftg = _listb(Re, "tg", _inv(cod))
    allg = {**gf, **fsg, **ftg}
    imgA = B.image_basis(g, source=sg, target=tg)   # target-bearing, non-injective
    imgB = B.image_basis(g, source=None, target=None)
    try:
        m = _cob(imgA, imgB, allg)
        back = _cob(imgB, imgA, allg)
        print("(b) non-inj img(target) cob shape:", m.shape, "back:", back.shape)
        # rectangular -> not invertible both ways; check the in-space realize is consistent
        print("(b) m @ back:\n", np.round(m @ back, 6))
    except Exception as e:
        print("(b) ERROR:", type(e).__name__, e)


if __name__ == "__main__":
    main()
