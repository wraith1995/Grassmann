"""Two-feed equivariance guard for the covariant FFS seam (``_realize_image_covariant``).

plans/94 Phase 1/3.  The covariant realization of ``ProjectedBasis(f, "image", src, tgt)``
reads ``f``'s matrix in the ENDPOINT coordinates, ``A0 = T⁻¹·F·S``, factors THAT, and maps
the ``Q`` factor back out (``cols = T·Q0``).  Its whole justification is that when ``f`` is
natural w.r.t. the endpoint transport (the exterior ``alt`` under the transported ``⊗ᵏ``
frames), ``A0`` is the CONSTANT reference matrix — so ``Q0`` is reference-constant and the
cell dependence of ``cols`` rides ENTIRELY in the rational endpoint cob ``T``.  That is the
``k ≥ 2`` metric-free-readout / ``P(T) = I`` requirement.

This is the permanent form of the hand probe that settled the (a) blocker: realize the three
sandwich ingredients (``F``, ``S``, ``T``) at TWO ``J`` feeds and assert ``A0`` is IDENTICAL.
A single reference feed (``J = I``) makes every frame coincide and hides a convention mix; two
feeds catch it.  (Historically the sandwich did NOT cancel until ``f``'s matrix was coerced to
the default slot bases — ``_coerce_rep_basis`` in ``_realize_image_covariant``; this guards
that fix.)

NOTE: the covariant seam being equivariant is NECESSARY but not sufficient for
``P(T) = I`` on a PROPER sub-face at ``k ≥ 2`` (the TET ``Λ²`` lane): that additionally needs
the sub-entity trace geometry, which is a separate frontier (``test_feec_family`` xfails).
This test isolates and pins the seam itself.
"""
from __future__ import annotations

import numpy as np
import pytest

import grassmann.core as C
import grassmann.embed as G
import polyarray as pa
from grassmann.core import basis as B
from grassmann.lower.represent import (
    _Ctx,
    _coerce_rep_basis,
    _rep,
    _shape_of,
    _TransposeOp,
    _transpose_shape,
    realize_basis_cols,
)
from grassmann.stdlib.exterior import alt


def _k(v):
    return f"{v.name}_{v.uid}"


def _ingredients(n: int, k: int, feed_dom_cod: np.ndarray):
    """Realize (F = f_cd, S, T) of the covariant sandwich for ``alt(dual(Img J), k)`` at one feed."""
    Rn = C.base("Rn", n)
    J = C.var("J", C.linear_map_space(Rn, Rn), G.bij)
    Alt = alt(G.dual(G.img(J)), k)                       # dual(Img J)⊗ᵏ ⊸ dual(Img J)⊗ᵏ
    dib = B.dual_basis(B.image_basis(J))
    sb = B.basis_prod(*([dib] * k))
    tb = B.basis_prod(*([dib] * k))

    prog = pa.Program("cov", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    f_rep = _coerce_rep_basis(ctx, _rep(Alt.node, ctx), note="probe:f-to-default")
    [f_cd] = prog.emit_stmt(
        _TransposeOp(), [f_rep.array],
        [pa.OutSpec("f_cod_dom", _transpose_shape(_shape_of(f_rep.array)))], note="to-(cod,dom)",
    )
    prog.add_output("F", f_cd)
    prog.add_output("S", realize_basis_cols(ctx, sb))
    prog.add_output("T", realize_basis_cols(ctx, tb))
    out = prog.run({_k(J): feed_dom_cod})
    return {name: np.asarray(out[name], float) for name in ("F", "S", "T")}


def _a0(ingr) -> np.ndarray:
    F, S, T = ingr["F"], ingr["S"], ingr["T"]
    return np.linalg.inv(T) @ F @ S


# The reference feed (J = I) and two generic affine feeds; stored (dom, cod) = Jᵀ.
_REF = {2: np.eye(2), 3: np.eye(3)}
_FEEDS = {
    2: [np.array([[1.3, 0.4], [-0.2, 0.9]]), np.array([[0.7, 0.1], [0.5, 1.6]])],
    3: [np.array([[1.0, 0.3, 0.1], [0.2, 1.4, -0.2], [0.1, 0.0, 1.2]]),
        np.array([[0.9, -0.2, 0.3], [0.1, 1.1, 0.0], [-0.3, 0.4, 1.5]])],
}


@pytest.mark.parametrize("n,k", [(2, 2), (3, 2), (3, 3)])
def test_covariant_A0_is_feed_invariant(n: int, k: int) -> None:
    """``A0 = T⁻¹·F·S`` is IDENTICAL across ``J`` feeds — the naturality of ``alt`` makes the
    endpoint sandwich cancel, so the realized frame ``cols = T·Q0`` is affine-equivariant."""
    a_ref = _a0(_ingredients(n, k, _REF[n].T))
    for feed in _FEEDS[n]:
        a = _a0(_ingredients(n, k, feed.T))
        assert np.allclose(a, a_ref, atol=1e-10), (
            f"covariant A0 not feed-invariant (n={n}, k={k}): ‖ΔA0‖="
            f"{np.linalg.norm(a - a_ref):.2e}"
        )


@pytest.mark.parametrize("n,k", [(2, 2), (3, 2)])
def test_covariant_A0_is_the_reference_antisymmetrizer(n: int, k: int) -> None:
    """The feed-invariant ``A0`` is exactly ``alt``'s reference matrix — a projector onto the
    antisymmetric subspace (``A0² = A0``, ``rank A0 = C(n, k)``), NOT a per-cell rotation."""
    from math import comb

    a0 = _a0(_ingredients(n, k, _REF[n].T))
    assert np.allclose(a0 @ a0, a0, atol=1e-10), "A0 is not idempotent (not the antisymmetrizer)"
    assert np.linalg.matrix_rank(a0, tol=1e-9) == comb(n, k), "A0 rank ≠ C(n, k)"
