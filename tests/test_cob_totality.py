"""``cob`` is **total** between any two bases of the same space (`43 §2c`).

The pure-matrix :func:`grassmann.lower.cob.cob` refuses FFS bases (``ImageBasis``/
``ProjectedBasis``, realized through the SVD/QR seam).  The represent-level
``_basis_cob_matrix`` closes that gap: it goes between *any two bases of the same space* —
the FFS leaves, their duals, and structured (``BasisProd``/``BasisSum``) bases with FFS
leaves — by realizing toward the ambient coords and composing ``pinv(C_to)·C_from`` (the
`41 §2` structural rules thread through products/sums/duals).

For an FFS basis of ``Img(J)`` the ambient columns are known in closed form: ``ImageBasis``
= the literal columns ``J``; ``ProjectedBasis`` = ``Q`` of the QR.  The correct cob from
coords in basis 1 to coords in basis 2 is ``pinv(C₂)·C₁``; a dual pair adds the ``(·)⁻ᵀ``
rule; a product is the Kronecker product of the per-factor cobs.
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann as G
import grassmann.lower.represent  # noqa: F401  (ensure the module is imported)
from grassmann.core import basis as B

_RM = sys.modules["grassmann.lower.represent"]


# Injective J : Rt(2) → Re(3); its image carries two natural bases.
JMATH = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
JSTORED = JMATH.T  # stored (|dom|,|cod|)
Q, R = np.linalg.qr(JMATH)  # ProjectedBasis cols = Q; ImageBasis cols = JMATH


def _cob(b_from: B.Basis, b_to: B.Basis, J: G.Expr) -> np.ndarray:
    """Run ``_basis_cob_matrix(b_from → b_to)`` and return the numeric matrix."""
    prog = pa.Program("cob")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    m = _RM._basis_cob_matrix(ctx, b_from, b_to)
    prog.add_output("M", m)
    key = f"{J.node.name}_{J.node.uid}"
    return np.asarray(prog.run({key: JSTORED})["M"], float)


def _setup():
    Rt = G.space("Rt", 2)
    Re = G.space("Re", 3)
    J = G.var("J", Rt >> Re, props=G.inj)
    img = G.img(J)
    imgb = B.ImageBasis(J.node)
    projb = B.ProjectedBasis(J.node, "image")
    return J, img, imgb, projb


def test_cob_image_to_projected() -> None:
    """``cob(ImageBasis → ProjectedBasis) = pinv(Q)·J = R`` (the QR upper-triangular)."""
    J, _img, imgb, projb = _setup()
    got = _cob(imgb, projb, J)
    assert got == pytest.approx(np.linalg.pinv(Q) @ JMATH)
    assert got == pytest.approx(R)


def test_cob_projected_to_image() -> None:
    """``cob(ProjectedBasis → ImageBasis) = pinv(J)·Q = R⁻¹``."""
    J, _img, imgb, projb = _setup()
    got = _cob(projb, imgb, J)
    assert got == pytest.approx(np.linalg.pinv(JMATH) @ Q)
    assert got == pytest.approx(np.linalg.inv(R))


def test_cob_ffs_roundtrip_is_identity() -> None:
    """``cob(b→b') · cob(b'→b) = I`` for the two image bases (a genuine inverse pair)."""
    J, _img, imgb, projb = _setup()
    assert _cob(imgb, projb, J) @ _cob(projb, imgb, J) == pytest.approx(np.eye(2))


def test_cob_equal_ffs_basis_is_identity() -> None:
    """``cob(b→b)`` is the identity for an FFS basis (the equal-basis short-circuit)."""
    J, _img, imgb, projb = _setup()
    assert _cob(imgb, imgb, J) == pytest.approx(np.eye(2))
    assert _cob(projb, projb, J) == pytest.approx(np.eye(2))


def test_cob_dual_ffs_is_inverse_transpose() -> None:
    """``cob(dual b → dual b') = (cob(b→b'))⁻ᵀ`` (the `41 §2` DualBasis rule)."""
    J, _img, imgb, projb = _setup()
    primal = _cob(imgb, projb, J)
    dual = _cob(B.dual_basis(imgb), B.dual_basis(projb), J)
    assert dual == pytest.approx(np.linalg.inv(primal).T)


def test_cob_structural_product_with_ffs_leaf() -> None:
    """``cob`` over a ``BasisProd`` with an FFS leaf = Kron of the per-factor cobs.

    A map ``Img(J) → Re2`` is realized over ``BasisProd(DualBasis(img-basis), base(Re2))``.
    Changing the domain leaf ``ImageBasis→ProjectedBasis`` (covariant ⇒ ``R⁻ᵀ``) while the
    codomain stays the standard basis is ``kron(R⁻ᵀ, I₂)``.
    """
    J, _img, imgb, projb = _setup()
    # An explicit-vectors codomain basis (the standard basis carries no dim — a separate,
    # pre-existing limitation); here it is the same on both sides, so its leaf cob is I₂.
    base_re2 = B.base_basis("Re2", np.eye(2))
    from_b = B.basis_prod(B.dual_basis(imgb), base_re2)
    to_b = B.basis_prod(B.dual_basis(projb), base_re2)
    got = _cob(from_b, to_b, J)
    # domain leaf cob (dual): (cob(imgb→projb))⁻ᵀ = R⁻ᵀ ; codomain leaf: I₂.
    expect = np.kron(np.linalg.inv(R).T, np.eye(2))
    assert got == pytest.approx(expect)


def test_cob_standard_basis_to_rotated() -> None:
    """``cob`` from the *standard* basis (no explicit vectors) to a rotated one is total.

    The standard ``BaseBasis`` is the ambient identity but carries no dimension of its own;
    cob sizes it from the companion basis.  ``cob(std → rot) = pinv(ROT)·I = ROT⁻¹``.
    """
    ROT = np.array([[2.0, 1.0], [0.0, 3.0]])
    std = B.base_basis("V")           # standard, unsizeable alone
    rot = B.base_basis("V", ROT)      # explicit columns
    prog = pa.Program("cob")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    m = _RM._basis_cob_matrix(ctx, std, rot)
    prog.add_output("M", m)
    got = np.asarray(prog.run({})["M"], float)
    assert got == pytest.approx(np.linalg.inv(ROT))


def test_cob_applied_recovers_coordinates() -> None:
    """The cob actually maps ImageBasis coords ``s`` to ProjectedBasis coords ``R·s``."""
    J, _img, imgb, projb = _setup()
    M = _cob(imgb, projb, J)
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.standard_normal(2)            # ImageBasis coords of v = J·s
        assert M @ s == pytest.approx(R @ s)  # ProjectedBasis coords of the same v
