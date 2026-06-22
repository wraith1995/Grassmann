"""Scratch probe: ImageBasis with BOTH source and target, cob to same-image ImageBasis
with DIFFERENT source AND target.  Verify (a) round-trip, (b) target-independence
(cob must equal the pure source-basis cob), (c) correctness vs an independent realize."""
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


def _realize(b, feeds):
    return _ctx_run(lambda ctx: _RM.realize_basis_cols(ctx, b), feeds)


def _imm(b, feeds):
    # ambient immersion (codomain coords, honoring target)
    return _ctx_run(lambda ctx: _RM._ffs_immersion(ctx, b), feeds)


def _listb(space, name, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    return B.list_basis(*vs), {_k(v): cols[:, i] for i, v in enumerate(vs)}


def main():
    dom, cod = 2, 3   # injective R^2 -> R^3
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var("F", C.linear_map_space(Rt, Re))
    Jm = _inv(max(dom, cod))[:cod, :dom]   # cod x dom full-rank
    feeds = {_k(f): Jm.T}

    # two DIFFERENT source bases of dom, two DIFFERENT target bases of cod (ambient)
    s1, fs1 = _listb(Rt, "s1", _inv(dom))
    s2, fs2 = _listb(Rt, "s2", _inv(dom) + np.array([[0.0, 1.0], [3.0, 0.0]]))
    t1, ft1 = _listb(Re, "t1", _inv(cod))
    t2, ft2 = _listb(Re, "t2", _inv(cod) + np.eye(cod) * 0.5)

    allfeeds = {**feeds, **fs1, **fs2, **ft1, **ft2}

    A = B.image_basis(f, source=s1, target=t1)
    Bb = B.image_basis(f, source=s2, target=t2)

    m = _cob(A, Bb, allfeeds)
    back = _cob(Bb, A, allfeeds)
    print("A->B cob:\n", m)
    print("round-trip m@back:\n", m @ back)
    print("round-trips:", np.allclose(m @ back, np.eye(m.shape[0])))

    # target-independence: cob should equal the cob between source bases only
    A0 = B.image_basis(f, source=s1)   # no target
    B0 = B.image_basis(f, source=s2)
    m0 = _cob(A0, B0, allfeeds)
    print("target-independence (m == m_no_target):", np.allclose(m, m0))

    # INDEPENDENT correctness check via ambient immersions (honoring targets).
    # A vector x has coords c_A in basis A (columns U_A in some ambient frame) and c_B in B.
    # U_A c_A = U_B c_B  =>  c_B = (U_B)^+ U_A c_A, i.e. cob(A->B) = pinv(U_B) U_A.
    # BUT U_A and U_B are in DIFFERENT ambient frames (t1 vs t2)!  So we must bring both into a
    # common frame.  Immersion with target gives columns in that target's coords; to compare we
    # express both in STANDARD codomain coords.
    UA_std = _imm(B.image_basis(f, source=s1), allfeeds)   # std cod coords
    UB_std = _imm(B.image_basis(f, source=s2), allfeeds)
    cob_indep = np.linalg.pinv(UB_std) @ UA_std
    print("independent cob (std-frame immersions):\n", cob_indep)
    print("dispatch == independent:", np.allclose(m, cob_indep))


if __name__ == "__main__":
    main()
