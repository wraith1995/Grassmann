"""Probe 4: inspect shapes for the non-injective over-determined-source ImageBasis."""
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


def _listb(space, name, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    return B.list_basis(*vs), {_k(v): cols[:, i] for i, v in enumerate(vs)}


dom, cod = 2, 3
Rt, Re = C.base("Rt", dom), C.base("Re", cod)
g = C.var("G", C.linear_map_space(Rt, Re))
Jg = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # rank-1
gf = {_k(g): Jg.T}
sg, fsg = _listb(Rt, "sg", _inv(dom))
tg, ftg = _listb(Re, "tg", _inv(cod))
allg = {**gf, **fsg, **ftg}

imm_std = _ctx_run(lambda c: _RM._ffs_immersion_std(c, B.image_basis(g, source=sg)), allg)
print("immersion_std (f·source) shape:", imm_std.shape)
print(np.round(imm_std, 4))

ud = _ctx_run(lambda c: _RM._realize_ffs_by_which(c, "image", g, source=None, target=None).cols, allg)
print("U_d shape:", ud.shape)
print(np.round(ud, 4))

insp = _ctx_run(lambda c: _RM._realize_ffs_in_space(c, B.image_basis(g, source=sg)), allg)
print("in-space (U_d^T·imm) shape:", insp.shape)
print(np.round(insp, 4))

# the metric-free ProjectedBasis image realize (the canonical delta)
proj = _ctx_run(lambda c: _RM.realize_basis_cols(c, B.projected_basis(g, "image")), allg)
print("proj(image) realize shape:", proj.shape)
