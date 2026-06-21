"""The cob **fallback routes through the standard basis** — tested aggressively, because it is
the catch-all path and most bases carry internal options (a `source`, a `target`, a
`complement`, a count, an arity, a metric, a nesting).

`cob(A → B) = cob(std → B) · cob(A → std)`: realizing a basis IS its cob to standard, and
`cob(std → A)` is the inverse.  For a representative of **every basis node and every variant of
its options**, this asserts `A → std` and `std → A` both exist and are mutual inverses (round
trip ⇒ identity).  Each new basis (or new option on an existing basis) MUST add a case here.
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


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _cob(a, b, feeds) -> np.ndarray:
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", _RM._basis_cob_matrix(ctx, a, b))
    return np.asarray(prog.run(feeds).get("M"), float)


def _invertible(n: int) -> np.ndarray:
    """A deterministic invertible n×n (unit lower-triangular + a varied diagonal)."""
    M = np.tril(np.ones((n, n)), -1) + np.diag(2.0 + np.arange(n))
    return M


def _listb(space, name, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    return B.list_basis(*vs), {_k(v): cols[:, i] for i, v in enumerate(vs)}


def _cases():
    """(`label`, basis, standard basis of its space, feeds) over every node and its options."""
    out: list = []

    # --- ScalarBasis (ℝ) -------------------------------------------------
    out.append(("ScalarBasis", B.scalar_basis(), B.scalar_basis(), {}))

    # --- ListBasis (term basis of ℝⁿ), n = 1..5, primal + covector -------
    for n in (1, 2, 3, 4, 5):
        Vn = C.base("V", n)
        lb, f = _listb(Vn, f"v{n}", _invertible(n))
        out.append((f"ListBasis ℝ^{n}", lb, default_basis(Vn), f))
    for n in (2, 3):
        Vn = C.base("V", n)
        lb, f = _listb(C.dual(Vn), f"dc{n}", _invertible(n))
        out.append((f"ListBasis covectors (ℝ^{n})*", lb, B.dual_basis(default_basis(Vn)), f))

    # --- DualBasis / PrimalBasis nestings --------------------------------
    for n in (2, 3):
        Vn = C.base("V", n)
        lb, f = _listb(Vn, f"d{n}", _invertible(n))
        out.append((f"DualBasis(ListBasis) (ℝ^{n})*",
                    B.dual_basis(lb), B.dual_basis(default_basis(Vn)), f))
        lb2, f2 = _listb(Vn, f"pr{n}", _invertible(n))
        out.append((f"PrimalBasis(DualBasis(ListBasis)) ℝ^{n}",
                    B.primal_basis(B.dual_basis(lb2)), default_basis(Vn), f2))

    # --- WithInnerBasis: WithInner(V, g) is its OWN space, so the "standard" basis here is the
    #     g-orthonormal WithInnerBasis(std, g) — NOT the bare V default (crossing to V needs an
    #     explicit remetric).  Both bases are of WithInner(V, g).
    for n in (2, 3):
        Vn = C.base("V", n)
        g = C.var(f"g{n}", C.oprod(C.dual(Vn), C.dual(Vn)))
        lb, f = _listb(Vn, f"wi{n}", _invertible(n))
        out.append((f"WithInnerBasis(ListBasis,g) ℝ^{n}",
                    B.with_inner_basis(lb, g), B.with_inner_basis(default_basis(Vn), g), f))

    # --- CompleteBasis: vary ambient n and sub width k -------------------
    for n in (2, 3, 4):
        for k in range(1, n):
            Vn = C.base("V", n)
            M = _invertible(n)
            sub, f = _listb(Vn, f"sub{n}_{k}", M[:, :k])
            out.append((f"CompleteBasis ℝ^{n} (sub {k}, RREF-completed)",
                        B.complete_basis(sub, default_basis(Vn), reduce=B.Reduce.RREF),
                        default_basis(Vn), f))

    # --- BasisSum: arities + per-summand options -------------------------
    A2, W1, U2 = C.base("A", 2), C.base("W", 1), C.base("U", 2)
    la, fa = _listb(A2, "sa", _invertible(2))
    out.append(("BasisSum(ListBasis, std) V⊕W",
                B.basis_sum(la, default_basis(W1)),
                B.basis_sum(default_basis(A2), default_basis(W1)), fa))
    lb3, fb3 = _listb(U2, "sb", _invertible(2))
    out.append(("BasisSum(ListBasis, std, ListBasis) ⊕³",
                B.basis_sum(la, default_basis(W1), lb3),
                B.basis_sum(default_basis(A2), default_basis(W1), default_basis(U2)),
                {**fa, **fb3}))

    # --- BasisProd: arities ---------------------------------------------
    lpa, fpa = _listb(A2, "pa", _invertible(2))
    out.append(("BasisProd(ListBasis, std) V⊗W",
                B.basis_prod(lpa, default_basis(U2)),
                B.basis_prod(default_basis(A2), default_basis(U2)), fpa))

    # --- BasisRep: counts + inner options --------------------------------
    for count in (1, 2, 3):
        V2 = C.base("V", 2)
        inner, fi = _listb(V2, f"rep{count}", _invertible(2))
        out.append((f"BasisRep({count}, ListBasis)",
                    B.basis_rep(count, inner), B.basis_rep(count, default_basis(V2)), fi))

    # --- ImageBasis / ProjectedBasis (FFS), source/target, shapes --------
    # square injective, tall injective, and image with a non-standard source.
    for (dom, cod) in ((2, 2), (3, 3), (2, 3), (3, 5)):
        tag, asc = f"{dom}→{cod}", f"{dom}to{cod}"
        Rt, Re = C.base("Rt", dom), C.base("Re", cod)
        f = C.var(f"F{asc}", C.linear_map_space(Rt, Re), props=C.props.inj)
        Jm = _invertible(cod)[:, :dom] if cod >= dom else _invertible(dom)
        feeds = {_k(f): Jm.T}
        projb = B.projected_basis(f, "image")
        out.append((f"ImageBasis {tag}", B.image_basis(f), projb, feeds))
        src, fs = _listb(Rt, f"src{asc}", _invertible(dom))
        out.append((f"ImageBasis(source) {tag}", B.image_basis(f, source=src), projb,
                    {**feeds, **fs}))
    # ProjectedBasis with a target metric ↔ the standard (metric-free) range.
    for dom, cod in ((2, 3), (3, 3)):
        Rt, Re = C.base("Rt", dom), C.base("Re", cod)
        f = C.var(f"Ft{dom}{cod}", C.linear_map_space(Rt, Re), props=C.props.inj)
        Jm = _invertible(cod)[:, :dom]
        tgt, ft = _listb(Re, f"tg{dom}{cod}", _invertible(cod))
        out.append((f"ProjectedBasis(target) {dom}→{cod}",
                    B.projected_basis(f, "image", target=tgt),
                    B.projected_basis(f, "image"), {_k(f): Jm.T, **ft}))
    # The other FFS selectors (ker / coimg / coker) via a non-injective map.
    Rt, Re = C.base("Rt", 3), C.base("Re", 2)
    h = C.var("H", C.linear_map_space(Rt, Re), props=C.props.surj)
    Hm = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])  # rank 2, ker dim 1
    for which in ("ker", "coimg", "coker"):
        pb = B.projected_basis(h, which)
        out.append((f"ProjectedBasis({which})", pb, pb, {_k(h): Hm.T}))

    return out


@pytest.mark.parametrize("label,basis,std,feeds", _cases(), ids=[c[0] for c in _cases()])
def test_basis_converts_to_and_from_standard(label, basis, std, feeds) -> None:
    """`A → std` and `std → A` both exist and are mutual inverses (round trip ⇒ identity)."""
    to_std = _cob(basis, std, feeds)      # A → standard (realize A's columns)
    from_std = _cob(std, basis, feeds)    # standard → A (the inverse transform)
    n = to_std.shape[0]
    assert np.all(np.isfinite(to_std)) and np.all(np.isfinite(from_std))
    assert to_std @ from_std == pytest.approx(np.eye(n)), f"{label}: A→std then std→A ≠ I"
    assert from_std @ to_std == pytest.approx(np.eye(n)), f"{label}: std→A then A→std ≠ I"
