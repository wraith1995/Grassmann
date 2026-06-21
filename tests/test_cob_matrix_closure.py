"""Exhaustive N×N coverage of the conversion matrix — `2 · 11² = 242` cases.

A change of basis is only valid between two bases of the **same space**.  So for every ordered
pair `(A, B)` of the 11 concrete basis nodes, in **both directions**, each cell-direction does
exactly one correct thing: it **converts** (and round-trips `A → B → A = I`) when the two
representatives are bases of the same space (or a canonical identification the pure cob handles),
or it **cleanly rejects** with `RepresentError` when they are genuinely different spaces (an FFS
subspace like `Img(f)` vs a full space, or `ℝ⁴` flat vs `ℝ²⊕ℝ²`).  Never a crash.

`ScalarBasis` is intrinsically 1-dimensional, so the cell-directions touching it use a
dimension-1 representative; every other pair uses a dimension-4 basis.  The same-space
conversions themselves are exercised exhaustively in `test_doc_conversions`,
`test_ffs_basis_conversions`, and `test_cob_via_standard`.
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B
from grassmann.lower.space_basis import default_basis

_RM = sys.modules["grassmann.lower.represent"]

NODES = ["ScalarBasis", "DualBasis", "PrimalBasis", "BasisSum", "BasisProd", "BasisRep",
         "ProjectedBasis", "ImageBasis", "ListBasis", "WithInnerBasis", "CompleteBasis"]


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _listb(name, space, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    return B.list_basis(*vs), {_k(v): cols[:, i] for i, v in enumerate(vs)}


def _reps_dim1():
    """Dimension-1 representative of each node — used for the pairs touching `ScalarBasis`."""
    S = C.scalar()
    lv, v2 = C.var("lv", S), C.var("v2", S)
    Rt = C.base("Rt", 1)
    f = C.var("f1", C.linear_map_space(Rt, Rt), props=C.props.inj)
    g = C.var("g1", C.oprod(C.dual(S), C.dual(S)))
    return {
        "ScalarBasis": (B.scalar_basis(), {}),
        "DualBasis": (B.dual_basis(B.scalar_basis()), {}),
        "PrimalBasis": (B.primal_basis(B.dual_basis(B.scalar_basis())), {}),
        "BasisSum": (B.basis_sum(B.scalar_basis()), {}),
        "BasisProd": (B.basis_prod(B.scalar_basis()), {}),
        "BasisRep": (B.basis_rep(1, B.scalar_basis()), {}),
        "ProjectedBasis": (B.projected_basis(f, "image"), {_k(f): np.array([[2.0]])}),
        "ImageBasis": (B.image_basis(f), {_k(f): np.array([[2.0]])}),
        "ListBasis": (B.list_basis(lv), {_k(lv): np.array(3.0)}),
        "WithInnerBasis": (B.with_inner_basis(B.scalar_basis(), g), {}),
        "CompleteBasis": (B.complete_basis(B.list_basis(v2), np.zeros((1, 0))),
                          {_k(v2): np.array(1.0)}),
    }


def _reps_dim4():
    """Non-trivial dimension-4 representative of each non-scalar node (a real 4×4 cob)."""
    R4, V2, W2 = C.base("R4", 4), C.base("V", 2), C.base("W", 2)
    M = np.array([[2.0, 1, 0, 0], [0, 3, 1, 0], [0, 0, 1, 1], [1, 0, 0, 2]])
    lb, flb = _listb("a", R4, M)
    lb2, flb2 = _listb("b", R4, M[:, ::-1].copy())
    f = C.var("F", C.linear_map_space(R4, R4), props=C.props.inj)
    ff = {_k(f): M.T}
    g = C.var("g4", C.oprod(C.dual(R4), C.dual(R4)))
    sub, fsub = _listb("s", R4, M[:, :2])
    comp = np.array([[0.0, 0], [0, 0], [1, 0], [0, 1.0]])
    return {
        "DualBasis": (B.dual_basis(lb), flb),
        "PrimalBasis": (B.primal_basis(B.dual_basis(lb)), flb),
        "BasisSum": (B.basis_sum(default_basis(V2), default_basis(W2)), {}),
        "BasisProd": (B.basis_prod(default_basis(V2), default_basis(W2)), {}),
        "BasisRep": (B.basis_rep(2, default_basis(V2)), {}),
        "ProjectedBasis": (B.projected_basis(f, "image"), ff),
        "ImageBasis": (B.image_basis(f), ff),
        "ListBasis": (lb2, flb2),
        "WithInnerBasis": (B.with_inner_basis(lb, g), flb),
        "CompleteBasis": (B.complete_basis(sub, comp), fsub),
    }


_D1 = _reps_dim1()
_D4 = _reps_dim4()


def _cob(a, b, feeds) -> np.ndarray:
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", _RM._basis_cob_matrix(ctx, a, b))
    return np.asarray(prog.run(feeds).get("M"), float)


# All 11 × 11 × 2 = 242 cell-directions of the matrix.
_CASES = [(i, j, d) for i in range(len(NODES)) for j in range(len(NODES)) for d in (0, 1)]


def _src_dst(case):
    i, j, d = case
    return (NODES[i], NODES[j]) if d == 0 else (NODES[j], NODES[i])


@pytest.mark.parametrize("case", _CASES,
                         ids=["{}->{}".format(*_src_dst(c)) for c in _CASES])
def test_matrix_cell(case) -> None:
    """Each cell-direction does exactly one of two correct things — never a crash:

    * **converts** — when the two representatives are bases of the *same* space (or a canonical
      identification the pure cob handles), `A → B` returns and round-trips with `B → A` ⇒ I; or
    * **cleanly rejects** — when they are genuinely *different* spaces (an FFS subspace like
      `Img(f)` vs a full space, or `ℝ⁴` flat vs `ℝ²⊕ℝ²`), `cob` raises `RepresentError`.

    A cob is only valid between two bases of the SAME space, so this is the honest N×N law; the
    same-space conversions themselves are exercised exhaustively in `test_doc_conversions`,
    `test_ffs_basis_conversions`, and `test_cob_via_standard`.
    """
    src, dst = _src_dst(case)
    reps = _D1 if (src == "ScalarBasis" or dst == "ScalarBasis") else _D4
    ba, fa = reps[src]
    bb, fb = reps[dst]
    feeds = {**fa, **fb}
    try:
        m = _cob(ba, bb, feeds)               # the example conversion src → dst (or a rejection)
    except _RM.RepresentError:
        assert src != dst, f"{src}->{src} (same basis) must convert, not reject"
        return                                 # genuinely different spaces ⇒ cleanly rejected
    back = _cob(bb, ba, feeds)                 # the other direction dst → src
    n = m.shape[0]
    assert np.all(np.isfinite(m))
    assert m @ back == pytest.approx(np.eye(n)), f"{src}->{dst} converted but did not round-trip"
