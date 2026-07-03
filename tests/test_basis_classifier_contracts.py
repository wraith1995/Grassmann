"""Regression (2026-07-03 audit #17 migration): the represent-layer basis classifiers
honor their documented contracts on wrapped/composite spellings.

Probe-verified bugs fixed here:
* ``_basis_variance`` read ``Orth(Dual(b))`` / ``Complete(dual, dual)`` as primal —
  ``_same_space``'s weakened variance guard could then pair a covector basis with a
  vector one (silent-wrong cob precondition).
* ``_is_immersed_metric_basis`` gave composite bases the identity-Gram fast path without
  inspecting factors — ``Prod(ImageBasis(J), std)`` has Gram ``JᵀJ ⊗ I ≠ I`` (Dagger's
  fast-path transpose would use the wrong metric).
"""

import importlib

from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.core import term as T

R = importlib.import_module("grassmann.lower.represent")
SB = importlib.import_module("grassmann.lower.space_basis")


def _fixture():
    V = S.OPlusRep(S.DimLit(3), S.Scalar())
    std = SB.default_basis(V)
    J = T.Var(name="J", space=S.OProd((S.Dual(V), V)))
    return V, std, J


def test_variance_recurses_through_orthonormal():
    _, std, _ = _fixture()
    assert R._basis_variance(None, B.dual_basis(std)) is True
    assert R._basis_variance(None, B.orthonormal_basis(B.dual_basis(std))) is True
    assert R._basis_variance(None, B.orthonormal_basis(std)) is False


def test_variance_recurses_through_complete():
    _, std, _ = _fixture()
    dual = B.dual_basis(std)
    assert R._basis_variance(None, B.complete_basis(dual, dual)) is True
    assert R._basis_variance(None, B.complete_basis(std, std)) is False


def test_immersed_gram_sees_composite_factors():
    _, std, J = _fixture()
    imm = B.image_basis(J)
    assert R._is_immersed_metric_basis(imm) is True
    assert R._is_immersed_metric_basis(B.basis_prod(imm, std)) is True
    assert R._is_immersed_metric_basis(B.basis_sum(imm, std)) is True
    assert R._is_immersed_metric_basis(B.basis_rep(S.DimLit(2), imm)) is True
    # composites of orthonormal factors keep the identity fast path (byte-identical default)
    assert R._is_immersed_metric_basis(B.basis_prod(std, std)) is False
    assert R._is_immersed_metric_basis(None) is False


def test_peel_orthonormal_metric_through_complete():
    """`Orth(Complete(sub, comp))` spans the COMPLEMENT's space (`_space_of_basis`), so a
    metric carried there must ride into the orthonormalization (2026-07-03 probe fix)."""
    V, std, J = _fixture()
    g = T.Var(name="g", space=S.OProd((S.Dual(V), S.Dual(V))))
    wib = B.with_inner_basis(std, g)
    got, parity = R._peel_orthonormal_metric(B.complete_basis(wib, wib))
    assert got is g and parity == 0
    # a metric-free completion still reports the standard product
    got2, _ = R._peel_orthonormal_metric(B.complete_basis(std, std))
    assert got2 is None
