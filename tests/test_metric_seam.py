"""Metric seam over NON-orthonormal bases — Gram, not the identity fast path.

A non-orthonormal ``ListBasis``/``CompleteBasis`` (``reduce`` ≠ ``ORTHONORMALIZE``) has Gram
``CᵀC`` ≠ I, so every metric consumer (``Inner``/``dot``, ``Dagger``, …) must materialize the Gram
rather than assume the identity.  Regression for the pre-existing bug where ``_is_immersed_metric_basis``
flagged only ``ImageBasis``, so such operands silently took the identity fast path → wrong numbers.
"""

import numpy as np

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent


def _run(prog, feeds):
    return np.asarray(prog.run(feeds)["result"], float)


def _vk(v):
    return f"{v.name}_{v.uid}"


# A non-orthonormal frame of V=ℝ³ (Gram = AmᵀAm ≠ I).
_AM = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])


def _list_basis_of(V):
    a = [C.var(f"e{i}", V) for i in range(3)]
    return B.list_basis(*a), {_vk(a[i]): _AM[:, i] for i in range(3)}


def test_dagger_metric_adjoint_over_a_nonorthonormal_list_basis():
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V))
    lb, lb_feeds = _list_basis_of(V)
    opb = B.basis_prod(B.dual_basis(lb), lb)  # operator basis with non-orthonormal slots
    M = np.array([[1.0, 2, 0], [0, 1, 3], [2, 0, 1.0]])  # f's coords in opb
    feeds = {**lb_feeds, _vk(f): M}
    r = _run(represent(C.dagger(f), basis_env={f.uid: opb}), feeds)
    G = _AM.T @ _AM
    true_adjoint = G @ M.T @ np.linalg.inv(G)  # G·fᵀ·G⁻¹, NOT the plain transpose
    assert np.allclose(r, true_adjoint)
    assert not np.allclose(r, M.T)


def test_inner_uses_the_gram_over_a_nonorthonormal_list_basis():
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    lb, lb_feeds = _list_basis_of(V)
    ac, bc = np.array([1.0, 2, 0]), np.array([0.0, 1, 3])
    feeds = {**lb_feeds, _vk(a): ac, _vk(b): bc}
    dot = C.eval_(C.inner(V), (0, 1), a, b)
    r = _run(represent(dot, basis_env={a.uid: lb, b.uid: lb}), feeds)
    G = _AM.T @ _AM
    assert np.allclose(r, ac @ G @ bc)  # ⟨a,b⟩ = acᵀ·G·bc
    assert not np.allclose(r, ac @ bc)  # NOT the identity-metric dot


def test_orthonormalized_list_basis_keeps_the_identity_fast_path():
    # A reduce=ORTHONORMALIZE list IS orthonormal ⇒ identity metric ⇒ the dot is the plain
    # coordinate dot (no Gram), and dagger is a plain transpose.  Confirms the fix is scoped to
    # genuinely non-orthonormal bases.
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    e = [C.var(f"e{i}", V) for i in range(3)]
    olb = B.list_basis(*e, reduce=C.Reduce.ORTHONORMALIZE)
    # An orthonormal frame fed to the ORTHONORMALIZE list (QR keeps it orthonormal).
    Q = np.linalg.qr(np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]]))[0]
    feeds = {f"e{i}_{e[i].uid}": Q[:, i] for i in range(3)}
    ac, bc = np.array([1.0, 2, 0]), np.array([0.0, 1, 3])
    feeds.update({_vk(a): ac, _vk(b): bc})
    dot = C.eval_(C.inner(V), (0, 1), a, b)
    r = _run(represent(dot, basis_env={a.uid: olb, b.uid: olb}), feeds)
    assert np.allclose(r, ac @ bc)  # orthonormal ⇒ identity metric
