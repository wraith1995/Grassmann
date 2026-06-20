"""Basis-invariance of the inner product across orthonormal and non-orthonormal FFS bases.

Regression for the metric-basis bug: Grassmann's ``dot`` used the identity metric for any
non-``WithInner`` space *regardless of the active basis*, so the same geometric vector
``t = J·s`` gave different inner products depending on whether it was expressed in the
orthonormal ``ProjectedBasis`` (Q of the QR) or the literal non-orthonormal ``ImageBasis``
(the ``f(eᵢ)`` columns).  The inner product must be basis-invariant:

    ⟨t, t⟩ = sᵀ(JᵀJ)s    (coordinate-free)

The fix builds the metric in the operands' *active* basis (`lower/represent.py`):
``ProjectedBasis``/``Q`` (orthonormal) ⇒ identity metric (the min-sym fast path), while the
non-orthonormal ``ImageBasis`` ⇒ the immersed Gram ``G = colsᵀ·G_cod·cols = JᵀJ`` (and its
dual carries ``(JᵀJ)⁻¹``), routed through the FFS realization seam (`73 Topic 2`, `50 §A`).
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.core import basis as B
from grassmann.lower.represent import represent


def _key(v: G.Expr) -> str:
    return f"{v.node.name}_{v.node.uid}"


# Three injective Jacobians: identity 2→2, skew 2→2, sub-ambient 2→3.
IDENTITY = np.eye(2)
SKEW = np.array([[2.0, 0.0], [1.0, 3.0]])
SUB_AMBIENT = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

CASES = [
    pytest.param(IDENTITY, 2, 2, id="identity-2x2"),
    pytest.param(SKEW, 2, 2, id="skew-2x2"),
    pytest.param(SUB_AMBIENT, 2, 3, id="sub-ambient-2to3"),
]


def _tangent_gram(Jmath: np.ndarray, dom: int, cod: int) -> np.ndarray:
    """The ``ImageBasis``-pinned ``dot`` kernel — should equal ``JᵀJ`` (the immersed Gram)."""
    Rt = G.space("Rt", dom)
    Re = G.space("Re", cod)
    J = G.var("J", Rt >> Re, props=G.inj)
    img = G.img(J)
    a = G.var("a", img)
    b = G.var("b", img)
    imgb = B.ImageBasis(J.node)
    prog = represent(G.dot(a, b).node, basis_env={a.node.uid: imgb, b.node.uid: imgb})
    return _kernel(prog, a, b, J, dom, Jmath.T)


def _cotangent_gram(Jmath: np.ndarray, dom: int, cod: int) -> np.ndarray:
    """The dual-``ImageBasis``-pinned ``dot`` kernel — should equal ``(JᵀJ)⁻¹``."""
    Rt = G.space("Rt", dom)
    Re = G.space("Re", cod)
    J = G.var("J", Rt >> Re, props=G.inj)
    dimg = G.dual(G.img(J))
    a = G.var("a", dimg)
    b = G.var("b", dimg)
    db = B.DualBasis(B.ImageBasis(J.node))
    prog = represent(G.dot(a, b).node, basis_env={a.node.uid: db, b.node.uid: db})
    return _kernel(prog, a, b, J, dom, Jmath.T)


def _kernel(prog, a, b, J, n: int, Jstored: np.ndarray) -> np.ndarray:
    """Probe a bilinear ``dot`` program at the basis pairs ``(eᵢ, eⱼ)`` to recover its Gram."""
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            ej = np.zeros(n)
            ej[j] = 1.0
            res = prog.run({_key(a): ei, _key(b): ej, _key(J): Jstored})
            out[i, j] = float(res["result"])
    return out


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
def test_inner_product_is_basis_invariant(Jmath: np.ndarray, dom: int, cod: int) -> None:
    """The same vector ``t = J·s`` gives the SAME inner product in both range bases.

    ``ProjectedBasis`` (orthonormal Q) holds the coordinates ``R·s``; ``ImageBasis`` (the
    literal columns) holds the coordinates ``s``.  Both must yield ``sᵀ(JᵀJ)s``.
    """
    Rt = G.space("Rt", dom)
    Re = G.space("Re", cod)
    J = G.var("J", Rt >> Re, props=G.inj)
    a = G.var("a", G.img(J))
    term = G.dot(a, a)

    Jstored = Jmath.T
    Qn, Rn = np.linalg.qr(Jmath)
    GtG = Jmath.T @ Jmath
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.standard_normal(dom)
        truth = float(s @ GtG @ s)

        projb = B.ProjectedBasis(J.node, "image")
        imgb = B.ImageBasis(J.node)

        def _run(basis, inp):
            prog = represent(term.node, basis_env={a.node.uid: basis})
            return float(
                prog.run({_key(a): np.asarray(inp, float), _key(J): Jstored})["result"]
            )

        assert _run(projb, Rn @ s) == pytest.approx(truth)
        assert _run(imgb, s) == pytest.approx(truth)


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
def test_image_basis_gram_is_JtJ(Jmath: np.ndarray, dom: int, cod: int) -> None:
    """The ``ImageBasis``-pinned ``dot`` kernel equals the immersed Gram ``JᵀJ`` (`73 T2`)."""
    gram = _tangent_gram(Jmath, dom, cod)
    assert gram == pytest.approx(Jmath.T @ Jmath)


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
def test_dual_image_basis_gram_is_inv_JtJ(Jmath: np.ndarray, dom: int, cod: int) -> None:
    """The dual-``ImageBasis``-pinned ``dot`` kernel equals the cotangent Gram ``(JᵀJ)⁻¹``."""
    gram = _cotangent_gram(Jmath, dom, cod)
    assert gram == pytest.approx(np.linalg.inv(Jmath.T @ Jmath))


def test_orthonormal_default_metric_is_identity() -> None:
    """The orthonormal-default (``ProjectedBasis``) ``dot`` kernel stays the identity metric."""
    Rt = G.space("Rt", 2)
    Re = G.space("Re", 3)
    J = G.var("J", Rt >> Re, props=G.inj)
    a = G.var("a", G.img(J))
    b = G.var("b", G.img(J))
    prog = represent(G.dot(a, b).node)  # no pin ⇒ choose picks the orthonormal default
    gram = _kernel(prog, a, b, J, 2, SKEW.T)
    assert gram == pytest.approx(np.eye(2))
