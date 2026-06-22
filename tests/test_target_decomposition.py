"""2.3 match decomposition — a node realizes DIRECTLY in a structure-matching target (no cob).

Correctness of `target` is covered by ``test_target_correctness``; here we assert the
*optimization*: when the concrete/STD target matches the node's structure and the operands are
free (unpinned ⇒ declared directly in the sub-target), NO conversion cob is emitted — the result
is built straight in the target.  AUTO stays byte-identical.
"""

import numpy as np
import polyarray as pa

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent
from grassmann.lower.space_basis import default_basis


def _notes(prog):
    return [st.note for sub in pa.forward.iter_programs(prog) for st in sub.statements]


def _no_target_cob(prog):
    return not any(n.startswith("cob:target") for n in _notes(prog))


def _nstmt(prog):
    return sum(1 for sub in pa.forward.iter_programs(prog) for _ in sub.statements)


def _run(prog, feeds):
    return np.asarray(prog.run(feeds)["result"], float)


def _vk(v):
    return f"{v.name}_{v.uid}"


# ---------------------------------------------------------------------------
# Structural split — OProdV / OPlusV build each factor in its sub-target.
# ---------------------------------------------------------------------------


def test_oprod_splits_a_basisprod_target_no_cob():
    V = C.base("V", 3)
    u, w = C.var("u", V), C.var("w", V)  # unpinned ⇒ declared directly in the sub-target
    op = C.oprod_v(u, w)
    uc, wc = np.array([1.0, 0, 2]), np.array([0.0, 1, 3])
    feeds = {_vk(u): uc, _vk(w): wc}
    prog = represent(op, target=B.STD)  # default_basis(OProd) is BasisProd(std, std)
    assert _no_target_cob(prog)
    assert np.allclose(_run(prog, feeds), np.outer(uc, wc))


def test_oplus_splits_a_basissum_target_no_cob():
    V, W = C.base("V", 3), C.base("W", 2)
    x, y = C.var("x", W), C.var("y", V)
    op = C.oplus_v(x, y)
    xc, yc = np.array([1.0, 2]), np.array([3.0, 4, 5])
    feeds = {_vk(x): xc, _vk(y): yc}
    prog = represent(op, target=B.STD)
    assert _no_target_cob(prog)
    assert np.allclose(_run(prog, feeds), np.concatenate([xc, yc]))


# ---------------------------------------------------------------------------
# Direct realize — Zero (any basis), Identity (same-basis endo pair).
# ---------------------------------------------------------------------------


def test_zero_is_direct_in_any_target():
    V = C.base("V", 3)
    a = [C.var(f"a{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    Am = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    feeds = {f"a{i}_{a[i].uid}": Am[:, i] for i in range(3)}
    prog = represent(C.zero(V), target=lb)
    assert _no_target_cob(prog)
    assert np.allclose(_run(prog, feeds), 0.0)


def test_identity_is_direct_in_a_same_basis_endo_pair():
    V = C.base("V", 3)
    a = [C.var(f"a{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    Am = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    feeds = {f"a{i}_{a[i].uid}": Am[:, i] for i in range(3)}
    prog = represent(C.identity(V), target=B.basis_prod(B.dual_basis(lb), lb))
    assert _no_target_cob(prog)
    assert np.allclose(_run(prog, feeds), np.eye(3))


# ---------------------------------------------------------------------------
# Push-down — Remetric realizes the operand in a WithInner target's underlying basis.
# ---------------------------------------------------------------------------


def test_remetric_pushes_a_withinner_target_down():
    V = C.base("V", 3)
    u = C.var("u", V)  # unpinned
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    prog = represent(C.remetric(u, g), target=B.with_inner_basis(default_basis(V), g))
    assert _no_target_cob(prog)


# ---------------------------------------------------------------------------
# AUTO byte-identical: the decomposition arms are gated on a concrete/STD target.
# ---------------------------------------------------------------------------


def test_auto_unchanged_by_the_decomposition_arms():
    V, W = C.base("V", 3), C.base("W", 2)
    u, w = C.var("u", V), C.var("w", V)
    x, y = C.var("x", W), C.var("y", V)
    for t in (C.oprod_v(u, w), C.oplus_v(x, y), C.zero(V), C.identity(V)):
        assert _nstmt(represent(t)) == _nstmt(represent(t, target=B.AUTO))
