"""Subprocess helper for the GSVD metric-pinv two-route correctness check.

Invoked as ``python _gsvd_two_route_helper.py <route> <kind> <out.npy>``:

  route : ``std``  — operand ``f`` (and ``v``) pinned to a non-default basis, ``target=STD``
          ``ref``  — operands in their DEFAULT basis, ``target=AUTO`` (the ground-truth value)
  kind  : ``pinv`` | ``pinvapply``

Each invocation builds exactly ONE ``represent()`` and saves the run output to ``out.npy``.  Running
the two routes in SEPARATE processes avoids the in-process free-label/einsum-symbol accumulation that
otherwise raises a spurious numpy "does not contain the correct number of indices" ValueError across
many ``represent()`` calls (a probe artifact, not a real bug).
"""

import sys

import numpy as np

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower.represent import default_basis, represent

# A fixed non-default invertible 3×3 basis matrix (columns = basis vectors).
_LB = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])


def _vk(v):
    return f"{v.name}_{v.uid}"


def _vec_coords(value):
    return np.linalg.solve(_LB, value)


def _map_coords(value):
    return _LB.T @ value @ np.linalg.inv(_LB).T


def main(route, kind, out_path):
    V0 = C.base("V", 3)
    g = C.var("gm", C.linear_map_space(V0, V0))
    Vg = C.with_inner(V0, g)
    MS = C.linear_map_space(Vg, Vg)
    f = C.var("f", MS)
    Gm = np.diag([2.0, 3.0, 1.0])
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])  # square invertible

    av = [C.var(f"e{i}", Vg) for i in range(3)]
    lb = B.list_basis(*av)
    mb = B.basis_prod(B.dual_basis(lb), lb)
    frame = {f"e{i}_{av[i].uid}": _LB[:, i] for i in range(3)}

    if kind == "pinv":
        term = C.pinv(f)
        if route == "std":
            prog = represent(term, basis_env={f.uid: mb}, target=B.STD)
            feeds = {_vk(f): _map_coords(Mf), _vk(g): Gm, **frame}
        else:
            prog = represent(term, basis_env={f.uid: default_basis(MS)}, target=B.AUTO)
            feeds = {_vk(f): Mf, _vk(g): Gm}
    elif kind == "pinvapply":
        v = C.var("v", Vg)
        vv = np.array([1.0, 0, 2])
        term = C.pinv_apply(f, v)
        if route == "std":
            prog = represent(term, basis_env={f.uid: mb, v.uid: lb}, target=B.STD)
            feeds = {_vk(f): _map_coords(Mf), _vk(v): _vec_coords(vv), _vk(g): Gm, **frame}
        else:
            prog = represent(
                term, basis_env={f.uid: default_basis(MS), v.uid: default_basis(Vg)},
                target=B.AUTO,
            )
            feeds = {_vk(f): Mf, _vk(v): vv, _vk(g): Gm}
    else:
        raise SystemExit(f"unknown kind {kind!r}")

    out = np.asarray(prog.run(feeds)["result"], float)
    np.save(out_path, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
