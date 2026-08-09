"""The rank-``k`` COVARIANT pullback off an immersed tangent, against the closed-form answer.

A rank-``k`` covariant tensor over ``Img(J)``, its components declared (pinned) in the immersed
frame ``DualBasis(ImageBasis(J))``, pulled back to the ambient ``Dual(ℝᵉ)^{⊗k}``.  The value is
``(J⁻ᵀ)^{⊗k}`` applied to the components — the ``k``-th tensor power of the order-1 rule, with no
second-derivative-of-the-chart term because the transport is a linear map.

⚠ **Array convention.**  grassmann stores a linear map ``V ⊸ W`` as ``(dom, cod)``, so feeding the
array ``A`` for ``jacobian`` means the map is ``x ↦ Aᵀx``.  The realized covariant transform on the
flattened components is therefore ``(A⁻¹)^{⊗k}`` — that is the same statement as ``(J⁻ᵀ)^{⊗k}`` read
in the stored layout, not a different one.

At ``k = 2`` this operator is **Kirby's Θ** — Robert C. Kirby, *A general approach to transforming
finite elements*, SMAI Journal of Computational Mathematics **4** (2018) 197–224, equation (3.33) —
the rank-2 block of his per-vertex ``V^c`` (3.34, ``[1, J⁻ᵀ, Θ⁻¹]``), which is what an H² element's
Hessian degrees of freedom transform by.  Θ acts on the symmetric-reduced triple
``(δˣˣ, δˣʸ, δʸʸ)``; the test reduces the realized 4×4 to that triple and compares entrywise.

⚠ **WHAT THIS PINS.**  ``represent._reconcile_pairing`` must match a multi-axis operand to its
contraction slot's dual partner by ⊗-LEAVES, not by top-level factor count.  A ``Kron`` map's fused
slot basis is FLAT (``_spliced_basis_prod``, in step with ``_synth_kron``'s spliced space) while a
value's basis mirrors its right-nested space tower, so from rank 2 up the two counts differ; a count
test refuses the reconciliation and the operand is contracted RAW — its components read in the basis
it was pinned to while the slot expects another.  The observable signature of that is a PINNED and an
UNPINNED realization that agree (the pin has no effect at all), realizing ``Q⊗Q`` for the orthonormal
frame ``Q`` instead of ``(A⁻¹)^{⊗2}``.  ``test_pin_is_honoured_at_rank_2`` is that tripwire.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import grassmann as G
from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.embed.ops import Expr
from grassmann.lower.represent import represent
from grassmann.lower.space_basis import default_basis, map_basis
from grassmann.transform.pullback import pullback

EDIM = 2
A = np.array([[2.0, -0.5], [1.0, 2.0]])          # the jacobian ARRAY (dom, cod)
AINV = np.linalg.inv(A)


def _setup() -> tuple[Expr, S.Space, S.Space, Callable[[S.Space], B.Basis | None]]:
    """``J : ℝᵗ ⊸ ℝᵉ`` injective, its image, and the immersed-frame leaf policy.

    Returns
    -------
    tuple
        ``(J, Img(J), ℝᵉ, leaf_policy)`` — the injective map expression, its image space, the
        ambient space, and the leaf policy :func:`~grassmann.lower.space_basis.map_basis` consults.
    """
    rt, re = G.space("Rt", EDIM), G.osum(*[G.R] * EDIM)
    jv = G.var("jacobian", rt >> re, props=G.inj)
    img = G.img(jv).node

    def immersed(sp: S.Space) -> B.Basis | None:
        """The immersed (literal transported) frame at the tangent leaf; default everywhere else."""
        if sp == img:
            return B.image_basis(jv.node, default_basis(rt.node), default_basis(re.node))
        return None

    return jv, img, re.node, immersed


def _tower(k: int, leaf: S.Space) -> S.Space:
    """``Dual(leaf) ⊗ … ⊗ Dual(leaf) ⊗ ℝ`` — the space a ``k``-th derivative of a scalar field
    lands in.

    FLAT, one axis per differentiation, exactly as ``transform.diff._tower_space`` builds it
    (`§flat-tower`); ``test_tower_matches_what_diff_builds`` pins the two together.  The
    alternative spelling, right-nested ``Dual ⊗ (Dual ⊗ (… ⊗ ℝ))`` as ``linear_map_space``
    curries, realizes as the ``(d, dᵏ⁻¹)`` MATRIX rather than ``k+1`` separate axes, and from
    ``k = 3`` up cannot meet the flat ``tensor_power`` product it is contracted against —
    ``unify`` raises ``AssociationMismatch``.

    Parameters
    ----------
    k
        The derivative order, i.e. the number of covariant slots.
    leaf
        The space each slot is dual to.

    Returns
    -------
    S.Space
        The flat ``k+1``-factor tower.
    """
    return S.oprod(*[S.dual(leaf)] * k, G.R.node)


def _realized_matrix(k: int, *, pinned: bool) -> np.ndarray:
    """The ``eᵈ`` columns of the realized rank-``k`` pullback, as an ``eᵏ × eᵏ`` matrix.

    Parameters
    ----------
    k
        The rank of the covariant tensor being pulled back.
    pinned
        Whether the source components are declared in the immersed frame (``True``) or left to
        the default, orthonormal one (``False``).

    Returns
    -------
    np.ndarray
        The realized transform, column ``c`` being the image of the ``c``-th unit component.
    """
    jv, img, amb, immersed = _setup()
    src, tgt = _tower(k, img), _tower(k, amb)
    v = G.var(f"v{k}", src)
    env = {v.node.uid: map_basis(src, immersed)} if pinned else None
    prog = represent(pullback(v.node, tgt), basis_env=env)
    vname = f"v{k}_{v.node.uid}"
    shape = {i.name: i.shape for i in prog.inputs}[vname]
    n = EDIM ** k
    out = np.zeros((n, n))
    for c in range(n):
        d = np.zeros(n)
        d[c] = 1.0
        out[:, c] = np.asarray(
            prog.run({f"jacobian_{jv.node.uid}": A, vname: d.reshape(shape)})["result"],
            dtype=float).ravel()
    return out


def _kron_power(m: np.ndarray, k: int) -> np.ndarray:
    """The ``k``-th Kronecker power ``m^{⊗k}`` (``k = 0`` is the 1×1 identity)."""
    out = np.array([[1.0]])
    for _ in range(k):
        out = np.kron(out, m)
    return out


def _kirby_theta(kmat: np.ndarray) -> np.ndarray:
    """Kirby (2018) eq. (3.33) on ``(δˣˣ, δˣʸ, δʸʸ)``, from ``kmat[a, i] = ∂x̂_a/∂x_i``."""
    return np.array([
        [kmat[0, 0] ** 2, 2 * kmat[0, 0] * kmat[1, 0], kmat[1, 0] ** 2],
        [kmat[0, 1] * kmat[0, 0],
         kmat[0, 0] * kmat[1, 1] + kmat[0, 1] * kmat[1, 0],
         kmat[1, 0] * kmat[1, 1]],
        [kmat[0, 1] ** 2, 2 * kmat[0, 1] * kmat[1, 1], kmat[1, 1] ** 2],
    ])


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_rank_k_covariant_pullback_is_the_kth_tensor_power(k: int) -> None:
    """``(A⁻¹)^{⊗k}`` — one rule, no per-order case.

    ``k = 3`` is the first degree at which the tower's SPELLING is observable (at ``k = 2`` the
    ⊗-unit rewrite bridges the one nesting level, ``(d,d)`` vs ``(d,d,1)``), so it is the case a
    nested tower cannot represent at all.  ``k = 4`` is cheap and pins that nothing about the rule
    is degree-2-or-3 special."""
    np.testing.assert_allclose(_realized_matrix(k, pinned=True), _kron_power(AINV, k),
                               atol=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_tower_matches_what_diff_builds(k: int) -> None:
    """ASSERT ON THE ARTIFACT: this file's ``_tower`` is the spelling ``diff`` actually allocates.

    The pullback above is only evidence about derivative jets if the space it pulls back is the one
    ``transform.diff`` mints for an order-``k`` derivative input.  Compare the two directly —
    ``_tower(k, ℝᵉ)`` against the space of the order-``k`` ``DerivInput`` a ``k``-fold ``diff`` of a
    scalar tracer allocates — so a producer-side respelling cannot leave this test passing against a
    layout nothing builds."""
    from grassmann.core import term as T
    # `grassmann.transform.diff` the ATTRIBUTE is the re-exported function, so name the module's
    # members directly rather than reaching through the package.
    from grassmann.transform.diff import clear_deriv_inputs, diff, enumerate_derivative_inputs

    amb = G.osum(*[G.R] * EDIM).node
    clear_deriv_inputs()
    try:
        u, x = T.Var("u", S.Scalar()), T.Var("x", amb)
        deps: dict[int, object] = {u.uid: None, x.uid: None}
        e: object = u
        for _ in range(k):
            e = G.beta(diff(e, x, deps))
            e = e.node if hasattr(e, "node") else e
            for rec in enumerate_derivative_inputs(e):
                deps.setdefault(rec.var.uid, None)
        (top,) = [r for r in enumerate_derivative_inputs(e) if r.order == k]
        assert top.var.space == _tower(k, amb)
    finally:
        clear_deriv_inputs()


def test_rank_2_block_is_kirby_theta() -> None:
    """The symmetric-reduced 3×3 of the rank-2 pullback IS Kirby (2018) eq. (3.33).

    Reduction: a symmetric ``δ`` has components ``(δ₀₀, δ₀₁, δ₁₁)``; the full 4×4 acts on
    ``(δ₀₀, δ₀₁, δ₁₀, δ₁₁)``, so the middle input column is the sum of the two mixed columns and
    the middle output row is either mixed row (they agree on a symmetric argument)."""
    m = _realized_matrix(2, pinned=True)
    reduced = np.array([
        [m[0, 0], m[0, 1] + m[0, 2], m[0, 3]],
        [m[1, 0], m[1, 1] + m[1, 2], m[1, 3]],
        [m[3, 0], m[3, 1] + m[3, 2], m[3, 3]],
    ])
    # `A` is stored (dom, cod), so the map is `Aᵀ` and Kirby's `∂x̂/∂x` matrix is `(Aᵀ)⁻¹ = A⁻ᵀ`.
    np.testing.assert_allclose(reduced, _kirby_theta(AINV.T), atol=1e-12)
    # the reduction is only legitimate because the image of a SYMMETRIC δ is symmetric
    for sym in ([1.0, 0, 0, 0], [0, 1.0, 1.0, 0], [0, 0, 0, 1.0]):
        out = m @ np.array(sym)
        np.testing.assert_allclose(out[1], out[2], atol=1e-12)


def test_pin_is_honoured_at_rank_2() -> None:
    """The immersed pin must CHANGE the rank-2 realization; a dropped pin leaves the two equal."""
    pinned = _realized_matrix(2, pinned=True)
    unpinned = _realized_matrix(2, pinned=False)
    assert not np.allclose(pinned, unpinned, atol=1e-9), (
        "rank-2 pullback ignored the basis pin: the pinned and unpinned realizations agree")
    # the unpinned one is the ORTHONORMAL frame's — orthogonal, so it cannot be `(A⁻¹)^{⊗2}`
    np.testing.assert_allclose(unpinned @ unpinned.T, np.eye(4), atol=1e-12)
