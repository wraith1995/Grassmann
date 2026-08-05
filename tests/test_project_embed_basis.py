"""project / embed honor the operand's basis.

``project``'s ``Pᵀ·⟦v⟧`` and ``embed``'s default-path ``P·⟦v⟧`` use a selection matrix ``P`` in the
*default* basis, so the operand must be realized in that default basis too — otherwise a non-default
operand pin contracts std-``P`` against non-std coords and silently returns the wrong vector (a
basis-honesty bug of the metric-seam class).  Two-route
ground truth: operand pinned to a non-default basis fed its coords (target=STD) == operand pinned
default fed the std vector.
"""

import numpy as np

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent
from grassmann.lower.space_basis import default_basis


def _vk(v):
    return f"{v.name}_{v.uid}"


def _run(prog, feeds):
    return np.asarray(prog.run(feeds)["result"], float)


_AM = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])  # a non-orthonormal frame of ℝ³


def _two_route(make_term):
    V = C.base("V", 3)
    u = C.var("u", V)
    vc = np.array([1.0, 2, 3])  # the underlying std vector
    t = make_term(u)
    ref = _run(represent(t, basis_env={u.uid: default_basis(V)}), {_vk(u): vc})
    a = [C.var(f"e{i}", V) for i in range(3)]
    lb = B.list_basis(*a)
    feeds = {f"e{i}_{a[i].uid}": _AM[:, i] for i in range(3)}
    feeds[_vk(u)] = np.linalg.solve(_AM, vc)  # the same vector, in lb coords
    got = _run(represent(t, basis_env={u.uid: lb}, target=B.STD), feeds)
    return ref, got


def test_project_honors_a_non_default_operand_basis():
    ref, got = _two_route(lambda u: C.project(u, C.base("V", 3)))
    assert np.allclose(ref, got)


def test_embed_honors_a_non_default_operand_basis():
    ref, got = _two_route(lambda u: C.embed(u, C.base("V", 3)))
    assert np.allclose(ref, got)
