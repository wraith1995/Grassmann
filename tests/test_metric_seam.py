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


def test_inner_mixed_one_list_one_default_operand():
    # ⟨a,b⟩ is coordinate-free = a_std · b_std.  With a in lb (a_std = Am·ac) and b in the default
    # std basis (b_std = bc), the reconciliation must yield (Am·ac)·bc — not ac·bc.
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    lb, lb_feeds = _list_basis_of(V)
    ac, bc = np.array([1.0, 2, 0]), np.array([0.0, 1, 3])
    feeds = {**lb_feeds, _vk(a): ac, _vk(b): bc}
    dot = C.eval_(C.inner(V), (0, 1), a, b)
    r = _run(represent(dot, basis_env={a.uid: lb}), feeds)  # only a pinned to lb
    assert np.allclose(r, (_AM @ ac) @ bc)
    assert not np.allclose(r, ac @ bc)


def test_dagger_cod_neq_dom_over_nonorthonormal_lists():
    V, W = C.base("V", 3), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    av = [C.var(f"v{i}", V) for i in range(3)]
    aw = [C.var(f"w{i}", W) for i in range(4)]
    lbV, lbW = B.list_basis(*av), B.list_basis(*aw)
    Cv = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])
    Cw = np.array([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1.0]])
    M = np.arange(12, dtype=float).reshape(3, 4) + 1.0  # f's coords (dom=V rows, cod=W cols)
    feeds = {_vk(f): M}
    feeds.update({_vk(av[i]): Cv[:, i] for i in range(3)})
    feeds.update({_vk(aw[i]): Cw[:, i] for i in range(4)})
    opb = B.basis_prod(B.dual_basis(lbV), lbW)
    r = _run(represent(C.dagger(f), basis_env={f.uid: opb}), feeds)
    Gv, Gw = Cv.T @ Cv, Cw.T @ Cw
    assert np.allclose(r, Gw @ M.T @ np.linalg.inv(Gv))  # G_cod·fᵀ·G_dom⁻¹, shape (4,3)


def test_flat_sharp_round_trip_over_a_nonorthonormal_list():
    V = C.base("V", 3)
    v = C.var("v", V)
    lb, lb_feeds = _list_basis_of(V)
    vc = np.array([2.0, 1, 3])
    feeds = {**lb_feeds, _vk(v): vc}
    flat_v = C.eval_(C.flat(V), (0,), v)  # v ↦ v♭ (apply G)
    sharp_flat_v = C.eval_(C.sharp(V), (0,), flat_v)  # v♭ ↦ (v♭)♯ (apply G⁻¹)
    rt = _run(represent(sharp_flat_v, basis_env={v.uid: lb}), feeds)
    assert np.allclose(rt, vc)  # sharp ∘ flat = id (G⁻¹·G·v = v)


def test_dual_nonorthonormal_list_carries_the_inverse_gram():
    V = C.base("V", 3)
    a, b = C.var("a", C.dual(V)), C.var("b", C.dual(V))
    e = [C.var(f"e{i}", V) for i in range(3)]
    lb = B.list_basis(*e)
    dlb = B.dual_basis(lb)
    ac, bc = np.array([1.0, 2, 0]), np.array([0.0, 1, 3])
    feeds = {f"e{i}_{e[i].uid}": _AM[:, i] for i in range(3)}
    feeds.update({_vk(a): ac, _vk(b): bc})
    dot = C.eval_(C.inner(C.dual(V)), (0, 1), a, b)
    r = _run(represent(dot, basis_env={a.uid: dlb, b.uid: dlb}), feeds)
    G = _AM.T @ _AM
    assert np.allclose(r, ac @ np.linalg.inv(G) @ bc)  # cotangent metric = G⁻¹


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
