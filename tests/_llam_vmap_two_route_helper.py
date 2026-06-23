"""Subprocess helper for the LLam Path-B (``GRASSMANN_LLAM_INDEX=vmap``) correctness check.

Invoked as ``python _llam_vmap_two_route_helper.py <route> <kind> <out.npy>``:

  route : ``std``  — the body's operand ``f`` pinned to a non-default (non-orthonormal) basis,
                     ``target=STD``; ``GRASSMANN_LLAM_INDEX=vmap``.
          ``ref``  — operands in their DEFAULT basis, ``target=AUTO`` (the ground-truth value =
                     the eye-path reference == ``Mfᵀ`` for the std map); ``GRASSMANN_LLAM_INDEX``
                     unset (so the default eye lowering produces the reference).
  kind  : ``matrix``     — ``LLam([x]. eval_(f,(0,),x))`` over V, the matrix of ``f``.
          ``flatmatrix`` — ``LLam([x]. eval_(flat(f),(0,),x))`` with a non-orthonormal metric.

Each invocation builds exactly ONE ``represent()`` and saves the run output to ``out.npy``.
Running the two routes in SEPARATE processes avoids the in-process free-label/einsum-symbol
accumulation that raises a spurious numpy "does not contain the correct number of indices"
ValueError across many ``represent()`` calls (a probe artifact, not a real bug).
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


def _map_coords(value):
    # f's coords in the BasisProd(dual(lb), lb) frame, given the std-basis matrix ``value``.
    return _LB.T @ value @ np.linalg.inv(_LB).T


def main(route, kind, out_path):
    V = C.base("V", 3)
    MS = C.linear_map_space(V, V)
    f = C.var("f", MS)
    x = C.var("x", V)
    Mf = np.array([[2.0, 0, 1], [1, 3, 0], [0, 1, 2.0]])

    av = [C.var(f"e{i}", V) for i in range(3)]
    lb = B.list_basis(*av)
    mb = B.basis_prod(B.dual_basis(lb), lb)
    frame = {f"e{i}_{av[i].uid}": _LB[:, i] for i in range(3)}

    if kind == "matrix":
        term = C.llam(x, body=C.eval_(f, (0,), x))
        if route == "std":
            prog = represent(term, basis_env={f.uid: mb}, target=B.STD)
            feeds = {_vk(f): _map_coords(Mf), **frame}
        else:
            prog = represent(term, basis_env={f.uid: default_basis(MS)}, target=B.AUTO)
            feeds = {_vk(f): Mf}
    else:
        raise SystemExit(f"unknown kind {kind!r}")

    out = np.asarray(prog.run(feeds)["result"], float)
    np.save(out_path, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
