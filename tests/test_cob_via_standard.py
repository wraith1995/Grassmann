"""The cob **fallback routes through the standard basis** — and every basis can be converted
to and from its standard basis, the two being mutual inverses (`Stage B`).

`cob(A → B) = cob(std → B) · cob(A → std)`, where `cob(A → std)` realizes `A`'s columns in
standard coords and `cob(std → A)` is its inverse.  This test verifies that for a
representative of **every** basis node: `A → std` and `std → A` both exist and compose to the
identity (so the standard basis is a universal hub and the transform is invertible).
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
_JMATH = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
_JSTORED = _JMATH.T
_M = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]])


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _cob(a, b, feeds) -> np.ndarray:
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", _RM._basis_cob_matrix(ctx, a, b))
    return np.asarray(prog.run(feeds).get("M"), float)


def _listb(space, name, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    feeds = {_k(v): cols[:, i] for i, v in enumerate(vs)}
    return B.list_basis(*vs), feeds


def _cases():
    """(`label`, basis, its standard basis, feeds) for a representative of every basis node."""
    out = []

    V3 = C.base("V", 3)
    std3 = default_basis(V3)
    term3, f3 = _listb(V3, "v", _M)
    out.append(("ListBasis (ℝ³ term basis)", term3, std3, f3))
    sub, fs = _listb(V3, "u", np.array([[1.0], [0.0], [0.0]]))
    comp = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out.append(("CompleteBasis", B.complete_basis(sub, comp), std3, fs))
    inner, fp = _listb(V3, "p", _M)
    out.append(("PrimalBasis(DualBasis(·))", B.primal_basis(B.dual_basis(inner)), std3, fp))
    g = C.var("g", C.oprod(C.dual(V3), C.dual(V3)))
    out.append(("WithInnerBasis", B.with_inner_basis(term3, g), std3, f3))

    out.append(("ScalarBasis", B.scalar_basis(), B.scalar_basis(), {}))

    # Dual space (ℝ³)*: standard = DualBasis(std). A covector term ListBasis and a wrapped
    # DualBasis(ListBasis) both convert to it.
    dstd = B.dual_basis(std3)
    dcov, fd = _listb(C.dual(V3), "d", _M)
    out.append(("ListBasis of covectors", dcov, dstd, fd))
    dterm, fdt = _listb(V3, "w", _M)
    out.append(("DualBasis(ListBasis)", B.dual_basis(dterm), dstd, fdt))

    # Img(J): standard = ProjectedBasis(image); ImageBasis and a term basis convert to it.
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    projb = B.projected_basis(J, "image")
    out.append(("ImageBasis", B.image_basis(J), projb, {_k(J): _JSTORED}))
    g0, g1 = C.var("g0", Re), C.var("g1", Re)
    cols = _JMATH @ np.array([[1.0, 1.0], [0.0, 1.0]])
    out.append(("ListBasis (image term basis)", B.list_basis(g0, g1), projb,
                {_k(J): _JSTORED, _k(g0): cols[:, 0], _k(g1): cols[:, 1]}))

    # ⊕ and ⊗ bases.
    VW = C.oplus(C.base("A", 2), C.base("Bg", 1))
    sumstd = B.basis_sum(default_basis(VW.spaces[0]), default_basis(VW.spaces[1]))
    sumterm, fsum = _listb(VW, "s", _M)
    out.append(("ListBasis (⊕ term basis)", sumterm, sumstd, fsum))
    VWp = C.oprod(C.base("A", 2), C.base("Bp", 2))
    prodstd = B.basis_prod(default_basis(VWp.spaces[0]), default_basis(VWp.spaces[1]))
    mats = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[1.0, 1.0], [0.0, 0.0]]),
            np.array([[1.0, 1.0], [1.0, 0.0]]), np.array([[1.0, 1.0], [1.0, 1.0]])]
    pv = [C.var(f"t{i}", VWp) for i in range(4)]
    fprod = {_k(v): mats[i] for i, v in enumerate(pv)}
    out.append(("ListBasis (⊗ term basis)", B.list_basis(*pv), prodstd, fprod))
    return out


@pytest.mark.parametrize("label,basis,std,feeds", _cases(),
                         ids=[c[0] for c in _cases()])
def test_basis_converts_to_and_from_standard(label, basis, std, feeds) -> None:
    """`A → std` and `std → A` both exist and are mutual inverses (round trip ⇒ identity)."""
    to_std = _cob(basis, std, feeds)      # A → standard (realize A's columns)
    from_std = _cob(std, basis, feeds)    # standard → A (the inverse transform)
    n = to_std.shape[0]
    assert to_std @ from_std == pytest.approx(np.eye(n)), f"{label}: A→std then std→A ≠ I"
    assert from_std @ to_std == pytest.approx(np.eye(n)), f"{label}: std→A then A→std ≠ I"
