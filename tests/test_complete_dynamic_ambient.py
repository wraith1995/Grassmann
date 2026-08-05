"""Regression: completing a frame over a DYNAMIC-RANK FFS ambient (`_CompleteColsOp`).

A ``ListBasis`` whose ambient is a dynamic-rank FFS space (``Img f`` of a non-injective map ⇒ a
runtime ``DimAtom`` dimension) WITH an explicit backup used to crash: ``_CompleteColsOp`` took the
target column count as a static ``int`` field, so a ``DimAtom`` ambient made the
``kept.shape[1] != self.n`` check (int vs symbolic) never match → ``ValueError``.  The op now reads
the target (= the ambient dimension) from the inputs' row count at run time, so it completes
correctly.  This was a pre-existing latent bug, not a regression.
"""

import importlib

import numpy as np
import polyarray as pa

import grassmann.core as C
from grassmann.core import basis as B

_RM = importlib.import_module("grassmann.lower.represent")


def _vk(v):
    return f"{v.name}_{v.uid}"


def test_list_basis_over_dynamic_ffs_ambient_with_explicit_backup_completes():
    R3 = C.base("R3", 3)
    f = C.var("f", C.linear_map_space(R3, R3))  # general 3×3 ⇒ runtime image rank
    v = C.var("v", C.img(f))  # an FFS-typed vector; the list's ambient is Img(f)
    lb = B.list_basis(v, backup=B.projected_basis(f, "image"))  # explicit backup, dynamic ambient

    prog = pa.Program("r")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("C", _RM.realize_basis_cols(ctx, lb))

    Jm = np.array([[2.0, 0, 0], [1, 3, 0], [1, 1, 0.0]])  # rank-2 map ⇒ image rank δ = 2
    M = np.asarray(prog.run({_vk(f): Jm.T, _vk(v): [1.0, 0]})["C"], float)  # v in-space (δ=2) coords

    assert M.shape == (2, 2)  # δ × δ — a full (square) basis of the rank-2 image
    assert abs(np.linalg.det(M)) > 1e-9  # full rank
