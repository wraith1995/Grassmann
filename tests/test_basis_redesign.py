"""Specialized basis machinery & cob cases (the Reduce/backup/OrthonormalBasis redesign).

Covers the new data model + dispatch: `ListBasis` backup (RREF completion) and `reduce`
(orthonormalize), `CompleteBasis` with a `Basis` complement + RREF, the `OrthonormalBasis`
node (the explicit dual of remetric), and `ImageBasis` source/target — each exercised through
realization and the change-of-basis dispatch.
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.lower.represent  # noqa: F401
from grassmann.core import Reduce
from grassmann.core import basis as B
from grassmann.lower.space_basis import default_basis

_RM = sys.modules["grassmann.lower.represent"]


def _k(v) -> str:
    return f"{v.node.name}_{v.node.uid}" if hasattr(v, "node") else f"{v.name}_{v.uid}"


def _realize(basis, feeds):
    prog = pa.Program("r")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("C", _RM.realize_basis_cols(ctx, basis))
    return np.asarray(prog.run(feeds)["C"], float)


def _cob(a, b, feeds):
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", _RM._basis_cob_matrix(ctx, a, b))
    return np.asarray(prog.run(feeds)["M"], float)


def _listb(space, name, cols):
    vs = [C.var(f"{name}{i}", space) for i in range(cols.shape[1])]
    return vs, {_k(v): cols[:, i] for i, v in enumerate(vs)}


# --- ListBasis backup: RREF-complete a rank-deficient frame ---------------
def test_listbasis_backup_rref_completes_to_full_basis() -> None:
    V = C.base("V", 3)
    vs, f = _listb(V, "u", np.array([[1.0], [1.0], [0.0]]))  # one vector ⇒ rank-deficient
    lb = B.list_basis(*vs, backup=default_basis(V), reduce=Reduce.RREF)
    cols = _realize(lb, f)
    assert cols.shape == (3, 3)
    assert np.linalg.matrix_rank(cols) == 3  # completed to a full basis
    # cob to/from the standard basis round-trips (it is a genuine basis of ℝ³).
    m = _cob(lb, default_basis(V), f)
    back = _cob(default_basis(V), lb, f)
    assert m @ back == pytest.approx(np.eye(3))


# --- ListBasis ORTHONORMALIZE --------------------------------------------
def test_listbasis_orthonormalize_is_orthonormal() -> None:
    V = C.base("V", 3)
    M = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    vs, f = _listb(V, "w", M)
    cols = _realize(B.list_basis(*vs, reduce=Reduce.ORTHONORMALIZE), f)
    assert cols.T @ cols == pytest.approx(np.eye(3))  # QᵀQ = I


# --- CompleteBasis with a Basis complement + RREF ------------------------
def test_completebasis_basis_complement_rref() -> None:
    V = C.base("V", 3)
    vs, f = _listb(V, "s", np.array([[1.0], [1.0], [1.0]]))
    cb = B.complete_basis(B.list_basis(*vs), default_basis(V), reduce=Reduce.RREF)
    cols = _realize(cb, f)
    assert cols.shape == (3, 3) and np.linalg.matrix_rank(cols) == 3
    assert _cob(cb, default_basis(V), f) @ _cob(default_basis(V), cb, f) == pytest.approx(np.eye(3))


# --- OrthonormalBasis: the explicit dual of remetric ---------------------
def test_orthonormalbasis_g_orthonormal_and_cob() -> None:
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    M = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Gm = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vs, f = _listb(V, "y", M)
    feeds = {**f, _k(g): Gm}
    wib = B.with_inner_basis(B.list_basis(*vs), g)
    ob = B.orthonormal_basis(wib)
    cols = _realize(ob, feeds)
    assert cols.T @ Gm @ cols == pytest.approx(np.eye(3))  # g-orthonormal
    # raw WithInner ↔ its OrthonormalBasis: same metricized space ⇒ an invertible cob (the
    # orthonormalization factor), never silent — both are WithInner(V, g) bases.
    m = _cob(wib, ob, feeds)
    back = _cob(ob, wib, feeds)
    assert m @ back == pytest.approx(np.eye(3))


def test_orthonormalbasis_to_orthonormalbasis_is_orthogonal_reorthonormalize() -> None:
    # Two OrthonormalBasis of the SAME metricized space (different inner bases): the cob is the
    # orthogonal reorthonormalization Q₂ᵀ·G·Q₁ — NOT the underlying (non-orthogonal) cob.  (This
    # was a bug: recursing on the inner bases gave the wrong matrix.)
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    Bm = np.array([[2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    Gm = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    av = [C.var(f"a{i}", V) for i in range(3)]
    bv = [C.var(f"b{i}", V) for i in range(3)]
    feeds = {**{_k(av[i]): A[:, i] for i in range(3)}, **{_k(bv[i]): Bm[:, i] for i in range(3)},
             _k(g): Gm}
    o1 = B.orthonormal_basis(B.with_inner_basis(B.list_basis(*av), g))
    o2 = B.orthonormal_basis(B.with_inner_basis(B.list_basis(*bv), g))
    Q1, Q2 = _realize(o1, feeds), _realize(o2, feeds)
    m = _cob(o1, o2, feeds)
    assert m == pytest.approx(Q2.T @ Gm @ Q1)                 # orthogonal reorthonormalization
    assert m @ _cob(o2, o1, feeds) == pytest.approx(np.eye(3))  # round-trips
    s = np.array([1.3, -0.7, 0.4])
    assert Q2 @ (m @ s) == pytest.approx(Q1 @ s)               # maps to the same vector


def test_raw_to_orthonormal_cob_is_cholesky_factor() -> None:
    # cob(raw WithInner(b,g) → OrthonormalBasis) = Uᵀ·G·A = L⁻¹AᵀGA = Lᵀ, the Cholesky factor of
    # the metric Gram (exact, no pseudo-inverse).
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    Gm = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    av = [C.var(f"a{i}", V) for i in range(3)]
    feeds = {**{_k(av[i]): A[:, i] for i in range(3)}, _k(g): Gm}
    wib = B.with_inner_basis(B.list_basis(*av), g)
    m = _cob(wib, B.orthonormal_basis(wib), feeds)
    assert m == pytest.approx(np.linalg.cholesky(A.T @ Gm @ A).T)   # = Lᵀ


def test_projectedbasis_metric_rectangular_reorthonormalize_roundtrips() -> None:
    # A tall injective f (rank 2 < ambient 3): the metric ProjectedBasis range is a RECTANGULAR
    # g-orthonormal basis — the differing-metric cob uses the exact Uᵀ·M factor and round-trips.
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    t = [C.var(f"t{i}", Re) for i in range(3)]
    Tm = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    feeds = {_k(J): Jm.T, **{_k(t[i]): Tm[:, i] for i in range(3)}}
    free = B.projected_basis(J, "image")
    metric = B.projected_basis(J, "image", target=B.list_basis(*t))
    m = _cob(free, metric, feeds)
    assert m.shape == (2, 2)
    assert m @ _cob(metric, free, feeds) == pytest.approx(np.eye(2))


def test_qr_sign_convention_positive_diagonal_r() -> None:
    # All our QRs use the +diagonal-R convention (R upper-triangular with a positive diagonal ⇒
    # the QR is unique and Q keeps the orientation of the columns).  A matrix that numpy's qr
    # returns with negative diagonal entries must, here, give a positive-diagonal R.
    V = C.base("V", 3)
    A = np.array([[-2.0, 1.0, 0.0], [0.0, -3.0, 1.0], [1.0, 0.0, -1.0]])
    assert np.any(np.diag(np.linalg.qr(A)[1]) < 0)            # numpy's raw QR is mixed-sign here
    vs, f = _listb(V, "q", A)
    # OrthonormalBasis (the _QrColsOp path): R = Qᵀ·A has a positive diagonal.
    Q = _realize(B.orthonormal_basis(B.list_basis(*vs)), f)
    assert np.all(np.diag(Q.T @ A) > 0)
    # FFS image (the QrOp seam): R = Qᵀ·J has a positive diagonal.
    Rt, Re = C.base("Rt", 3), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    U = _realize(B.projected_basis(J, "image"), {_k(J): A.T})
    assert np.all(np.diag(U.T @ A) > 0)


def test_primalbasis_to_primalbasis_is_inverse_transpose() -> None:
    # A lone PrimalBasis realizes like a dual ((realize inner)⁻ᵀ), so PrimalBasis → PrimalBasis is
    # the inverse-transpose of the inner cob, exactly like DualBasis → DualBasis.
    V = C.base("V", 2)
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    Bm = np.array([[2.0, 0.0], [1.0, 1.0]])
    av, fa = _listb(V, "pa", A)
    bv, fb = _listb(V, "pb", Bm)
    feeds = {**fa, **fb}
    p1 = B.primal_basis(B.list_basis(*av))
    p2 = B.primal_basis(B.list_basis(*bv))
    m = _cob(p1, p2, feeds)
    assert m == pytest.approx(np.linalg.inv(np.linalg.inv(Bm) @ A).T)   # (cob lb1→lb2)⁻ᵀ
    assert m @ _cob(p2, p1, feeds) == pytest.approx(np.eye(2))           # round-trips


def test_imagebasis_different_maps_are_rejected() -> None:
    # Two ImageBasis of DIFFERENT maps are bases of different image spaces ⇒ no cob (the maps
    # must be the same).  The space guard rejects them rather than mapping between distinct images.
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    f1 = C.var("F1", C.linear_map_space(Rt, Re), props=C.props.inj)
    f2 = C.var("F2", C.linear_map_space(Rt, Re), props=C.props.inj)
    feeds = {_k(f1): np.zeros((2, 3)), _k(f2): np.zeros((2, 3))}
    with pytest.raises(_RM.RepresentError):
        _cob(B.image_basis(f1), B.image_basis(f2), feeds)


def test_basisrep_to_basissum_same_space_is_block_diag() -> None:
    # BasisRep(n, b) and BasisSum(b,…,b) spell the SAME ⊕ⁿ space; with a non-pure (term) inner the
    # cob is the block-diagonal of the per-summand cobs (not a full realize-and-invert).
    V = C.base("V", 2)
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B0 = np.array([[2.0, 0.0], [1.0, 1.0]])
    B1 = np.array([[1.0, 0.0], [0.0, 3.0]])
    ai, fa = _listb(V, "ri", A)
    s0, f0 = _listb(V, "s0", B0)
    s1, f1 = _listb(V, "s1", B1)
    feeds = {**fa, **f0, **f1}
    rep = B.basis_rep(2, B.list_basis(*ai))
    summ = B.basis_sum(B.list_basis(*s0), B.list_basis(*s1))
    m = _cob(rep, summ, feeds)
    assert m.shape == (4, 4)
    assert m[:2, 2:] == pytest.approx(np.zeros((2, 2)))    # block-diagonal (off-blocks vanish)
    assert m[2:, :2] == pytest.approx(np.zeros((2, 2)))
    assert m @ _cob(summ, rep, feeds) == pytest.approx(np.eye(4))   # round-trips


def test_orthonormalbasis_plain_is_qr() -> None:
    V = C.base("V", 3)
    M = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    vs, f = _listb(V, "p", M)
    cols = _realize(B.orthonormal_basis(B.list_basis(*vs)), f)
    assert cols.T @ cols == pytest.approx(np.eye(3))  # standard-orthonormal


# --- ImageBasis source/target & the same-map source cob ------------------
def test_imagebasis_source_cob_is_source_basis_cob() -> None:
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    s0, s1 = C.var("s0", Rt), C.var("s1", Rt)
    Sm = np.array([[1.0, 0.0], [1.0, 1.0]])
    feeds = {_k(J): Jm.T, _k(s0): Sm[:, 0], _k(s1): Sm[:, 1]}
    src = B.list_basis(s0, s1)
    # same map (`is`) ⇒ cob(ImageBasis(J,source=S) → ImageBasis(J)) is the source-basis cob = S.
    m = _cob(B.image_basis(J, source=src), B.image_basis(J), feeds)
    assert m == pytest.approx(Sm)


def test_imagebasis_cob_is_target_independent() -> None:
    # SAME map: differing codomain *target* must NOT change the cob (target is embed-only); the
    # cob is the source-basis cob whether or not the targets agree.
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    s0, s1 = C.var("s0", Rt), C.var("s1", Rt)
    Sm = np.array([[1.0, 0.0], [1.0, 1.0]])
    t = [C.var(f"t{i}", Re) for i in range(3)]
    Tm = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    feeds = {_k(J): Jm.T, _k(s0): Sm[:, 0], _k(s1): Sm[:, 1],
             **{_k(t[i]): Tm[:, i] for i in range(3)}}
    src, tgt = B.list_basis(s0, s1), B.list_basis(*t)
    plain = _cob(B.image_basis(J, source=src), B.image_basis(J), feeds)              # no targets
    with_t = _cob(B.image_basis(J, source=src, target=tgt), B.image_basis(J), feeds)  # diff target
    assert plain == pytest.approx(with_t)   # target-independent (target is embed-only)
    assert plain == pytest.approx(Sm)        # = the source-basis cob (S)


def test_projectedbasis_same_metric_is_identity() -> None:
    # Same FFS subspace & same source/target metric ⇒ identity; a differing target metric ⇒ a
    # non-trivial (invertible) reorthonormalization.
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    t = [C.var(f"t{i}", Re) for i in range(3)]
    Tm = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    feeds = {_k(J): Jm.T, **{_k(t[i]): Tm[:, i] for i in range(3)}}
    plain = B.projected_basis(J, "image")
    tgt = B.projected_basis(J, "image", target=B.list_basis(*t))
    assert _cob(plain, plain, feeds) == pytest.approx(np.eye(2))            # same metric ⇒ I
    m = _cob(plain, tgt, feeds)
    assert m @ _cob(tgt, plain, feeds) == pytest.approx(np.eye(2))           # diff metric ⇒ invertible
    assert not np.allclose(m, np.eye(2))                                     # ...and non-trivial


def test_imagebasis_target_reads_in_codomain_basis() -> None:
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    t = [C.var(f"t{i}", Re) for i in range(3)]
    Tm = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    tgt = B.list_basis(*t)
    feeds = {_k(J): Jm.T, **{_k(t[i]): Tm[:, i] for i in range(3)}}
    cols = _realize(B.image_basis(J, target=tgt), feeds)
    # columns expressed in the target basis: T · cols_target = J (the standard image columns).
    assert Tm @ cols == pytest.approx(Jm)
