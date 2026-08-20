"""Exterior-**algebra** property tests.

Property-based numeric checks of the derived ``stdlib.wedge`` / ``exterior_power`` against
clean numpy oracles, lowered through ``represent`` + ``Program.run``.  They cover
antisymmetry across ``(n,k,l)`` and seeds, bilinearity / scaling / associativity /
graded-commutativity / dimension / Cauchy–Binet / top-power=det / ``Λᵏ=0`` (k>n).

Scope: exterior **algebra** + the Gram/determinant bridge.  NOT exterior calculus (forms,
``d``) — out of scope.

Each check loops over a grid of dims and rng seeds.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
from wedge_helpers import (
    antisym_tensor_norm_sq,
    image_basis_matrix,
    wedge_all,
    wedge_all_program,
)

import grassmann.embed as G
from grassmann.lower import represent
from grassmann.stdlib import exterior_power, tensor_power, wedge
from grassmann.stdlib.exterior import perms, sign

ATOL = 1e-9
SEEDS = (0, 42)


def _image_basis_wedge_program(V: G.Expr, k: int, l: int):  # noqa: E741
    """``wedge(a, b, k, l)`` lowered in the default ``ImageBasis(Λ^{k+l})`` basis.

    Operands are ``project(ambient V⊗, Λ)`` of ``V⊗ᵏ`` / ``V⊗ˡ`` input vars.  Returns
    ``(prog, name_a, name_b)``.
    """
    a = G.project(G.var("a_amb", tensor_power(V, k)), exterior_power(V, k))
    b = G.project(G.var("b_amb", tensor_power(V, l)), exterior_power(V, l))
    prog = represent(wedge(a, b, k, l).node)
    names = [i.name for i in prog.inputs]
    na = next(n for n in names if n.startswith("a"))
    nb = next(n for n in names if n.startswith("b"))
    return prog, na, nb


# ===========================================================================
# antisymmetry across (n,k,l) and many seeds.
# ===========================================================================


@pytest.mark.parametrize(("n", "k", "l"), [(3, 1, 1), (4, 1, 1), (4, 2, 1)])
@pytest.mark.parametrize("seed", SEEDS)
def test_wedge_antisymmetry_grid(n: int, k: int, l: int, seed: int) -> None:  # noqa: E741
    """``a ∧ b = (−1)^{kl} (b ∧ a)`` and (when k==l) ``a ∧ a = 0`` in the image basis."""
    V = G.space("V", n)
    prog, na, nb = _image_basis_wedge_program(V, k, l)
    rng = np.random.default_rng(seed)
    av = rng.standard_normal((n,) * k)
    bv = rng.standard_normal((n,) * l)

    ab = prog.run({na: av, nb: bv})["result"]
    # b ∧ a needs the (l, k) program (the wedge fixes operand degrees in its build).
    prog_ba, na2, nb2 = _image_basis_wedge_program(V, l, k)
    ba = prog_ba.run({na2: bv, nb2: av})["result"]

    grade_sign = (-1) ** (k * l)
    np.testing.assert_allclose(ab, grade_sign * ba, atol=ATOL)

    if k == l:
        aa = prog.run({na: av, nb: av})["result"]
        np.testing.assert_allclose(aa, 0.0, atol=ATOL)


# ===========================================================================
# bilinearity and per-slot scaling (one lowered program, linear combos).
# ===========================================================================


@pytest.mark.parametrize("seed", SEEDS)
def test_wedge_bilinearity_each_slot(seed: int) -> None:
    """``(αa+βa')∧b = α(a∧b)+β(a'∧b)`` and symmetrically in ``b`` (Λ¹∧Λ¹, V₄)."""
    n = 4
    V = G.space("V", n)
    prog, na, nb = _image_basis_wedge_program(V, 1, 1)
    rng = np.random.default_rng(seed)
    a1, a2 = rng.standard_normal(n), rng.standard_normal(n)
    b1, b2 = rng.standard_normal(n), rng.standard_normal(n)
    alpha, beta = 2.3, -1.7

    # slot 1
    lhs = prog.run({na: alpha * a1 + beta * a2, nb: b1})["result"]
    rhs = alpha * prog.run({na: a1, nb: b1})["result"] + beta * prog.run(
        {na: a2, nb: b1}
    )["result"]
    np.testing.assert_allclose(lhs, rhs, atol=ATOL)

    # slot 2
    lhs2 = prog.run({na: a1, nb: alpha * b1 + beta * b2})["result"]
    rhs2 = alpha * prog.run({na: a1, nb: b1})["result"] + beta * prog.run(
        {na: a1, nb: b2}
    )["result"]
    np.testing.assert_allclose(lhs2, rhs2, atol=ATOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_wedge_scaling_per_slot(seed: int) -> None:
    """``(αa)∧b = α(a∧b) = a∧(αb)`` — the cheap per-slot scaling guard."""
    n = 4
    V = G.space("V", n)
    prog, na, nb = _image_basis_wedge_program(V, 1, 1)
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    alpha = 3.14

    base = prog.run({na: a, nb: b})["result"]
    np.testing.assert_allclose(prog.run({na: alpha * a, nb: b})["result"], alpha * base, atol=ATOL)
    np.testing.assert_allclose(prog.run({na: a, nb: alpha * b})["result"], alpha * base, atol=ATOL)


# ===========================================================================
# embed/project round-trip: project(embed(x, V⊗ᵏ), Λᵏ) == x for x:Λᵏ.
# ===========================================================================


@pytest.mark.parametrize(("n", "k"), [(3, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("seed", (0, 7))
def test_embed_project_roundtrip_on_exterior_power(n: int, k: int, seed: int) -> None:
    """The ``Λᵏ ⊑ V⊗ᵏ`` embed used inside ``wedge`` is an isometric inclusion."""
    V = G.space("V", n)
    x = G.var("x", exterior_power(V, k))
    rt = G.project(G.embed(x, tensor_power(V, k)), exterior_power(V, k))
    prog = represent(rt.node)

    rng = np.random.default_rng(seed)
    xc = rng.standard_normal(math.comb(n, k))  # Λᵏ coords (length C(n,k))
    feed = {i.name: xc for i in prog.inputs}
    out = prog.run(feed)["result"]
    assert out.shape == (math.comb(n, k),)
    np.testing.assert_allclose(out, xc, atol=ATOL)


# ===========================================================================
# wedge component = minor; |v₁∧…∧v_m|² = det(Gram) (Cauchy–Binet).
# ===========================================================================


@pytest.mark.parametrize(("n", "m"), [(2, 2), (3, 2), (4, 2), (4, 3)])
@pytest.mark.parametrize("seed", (0, 99))
def test_wedge_component_is_minor(n: int, m: int, seed: int) -> None:
    """``T[i₀<…<i_{m-1}] == det(M[:, [i₀…i_{m-1}]])`` for the ambient antisym tensor."""
    V = G.space("V", n)
    rng = np.random.default_rng(seed)
    vs = [rng.standard_normal(n) for _ in range(m)]
    T = wedge_all(V, vs)
    M = np.stack(vs)  # (m, n)
    for cols in combinations(range(n), m):
        minor = np.linalg.det(M[:, cols])
        np.testing.assert_allclose(T[cols], minor, atol=ATOL)


@pytest.mark.parametrize(("n", "m"), [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3)])
@pytest.mark.parametrize("seed", (0, 555))
def test_cauchy_binet_gram_determinant(n: int, m: int, seed: int) -> None:
    """``|v₁∧…∧v_m|² = det(Gram)`` with ``Gram_{ij}=⟨v_i,v_j⟩`` (the highest-value law).

    Validates multilinearity + antisymmetry + dimension + the geometry bridge in one
    identity against a clean numpy oracle.  The squared norm is over the *sorted* tensor
    components (the minors), so this is exactly the Cauchy–Binet expansion of ``det(M Mᵀ)``.
    """
    V = G.space("V", n)
    rng = np.random.default_rng(seed)
    vs = [rng.standard_normal(n) for _ in range(m)]
    T = wedge_all(V, vs)
    M = np.stack(vs)
    gram_det = np.linalg.det(M @ M.T)
    np.testing.assert_allclose(antisym_tensor_norm_sq(T, m), gram_det, atol=1e-8)
    assert gram_det >= -1e-9  # Gram is PSD ⇒ det ≥ 0 (squared volume)


# ===========================================================================
# wedge with a linearly-dependent vector = 0 (general m).
# ===========================================================================


@pytest.mark.parametrize(
    ("n", "m"),
    [(3, 2), (4, 3), (4, 4)],
)
@pytest.mark.parametrize("seed", (0, 88))
def test_wedge_dependent_vector_is_zero(n: int, m: int, seed: int) -> None:
    """``v₁∧…∧v_m = 0`` when the ``v_i`` are linearly dependent (last = combo of others)."""
    V = G.space("V", n)
    rng = np.random.default_rng(seed)
    vs = [rng.standard_normal(n) for _ in range(m - 1)]
    coeffs = rng.standard_normal(m - 1)
    vs.append(sum(c * v for c, v in zip(coeffs, vs, strict=True)))
    T = wedge_all(V, vs)
    np.testing.assert_allclose(T, 0.0, atol=ATOL)
    # numpy oracle: rank-deficient stack ⇒ Gram det 0.
    M = np.stack(vs)
    np.testing.assert_allclose(np.linalg.det(M @ M.T), 0.0, atol=1e-8)


# ===========================================================================
# associativity (a∧b)∧c == a∧(b∧c) (nested-wedge operand path).
# ===========================================================================


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("seed", (0, 17))
def test_wedge_associativity(n: int, seed: int) -> None:
    """``(a∧b)∧c == a∧(b∧c)`` — both equal the 3-fold antisymmetrization (ambient V⊗³)."""
    V = G.space("V", n)

    def op(name: str) -> G.Expr:
        return G.project(G.var(f"{name}_amb", tensor_power(V, 1)), exterior_power(V, 1))

    a, b, c = op("a"), op("b"), op("c")
    left = wedge(wedge(a, b, 1, 1), c, 2, 1)
    right = wedge(a, wedge(b, c, 1, 1), 1, 2)
    prog_l = represent(G.embed(left, tensor_power(V, 3)).node)
    prog_r = represent(G.embed(right, tensor_power(V, 3)).node)

    rng = np.random.default_rng(seed)
    av, bv, cv = (rng.standard_normal(n) for _ in range(3))

    def feed(prog):
        f = {}
        for inp in prog.inputs:
            for p, val in (("a", av), ("b", bv), ("c", cv)):
                if inp.name.startswith(p):
                    f[inp.name] = val
        return f

    out_l = prog_l.run(feed(prog_l))["result"]
    out_r = prog_r.run(feed(prog_r))["result"]
    np.testing.assert_allclose(out_l, out_r, atol=ATOL)


# ===========================================================================
# top exterior power is the determinant (dim Λⁿ = 1).
# ===========================================================================


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("seed", (0, 314))
def test_top_power_component_is_determinant(n: int, seed: int) -> None:
    """``(v₁∧…∧vₙ)[0,1,…,n-1] == det[v₁…vₙ]`` and ``dim Λⁿ = 1`` (numeric)."""
    V = G.space("V", n)
    rng = np.random.default_rng(seed)
    vs = [rng.standard_normal(n) for _ in range(n)]
    T = wedge_all(V, vs)
    np.testing.assert_allclose(T[tuple(range(n))], np.linalg.det(np.stack(vs)), atol=ATOL)
    # dim Λⁿ = C(n,n) = 1.
    assert image_basis_matrix(V, n).shape[1] == 1


# ===========================================================================
# full Sₘ sign law: permuting operands multiplies by sgn(σ).
# ===========================================================================


@pytest.mark.parametrize(
    ("n", "m"),
    [(3, 3), (4, 3), (4, 4)],
)
def test_full_symmetric_group_sign_law(n: int, m: int) -> None:
    """Permuting the ``m`` operands of a wedge scales it by ``sgn(σ)`` for all σ∈Sₘ."""
    V = G.space("V", n)
    prog, prefixes = wedge_all_program(V, m)
    rng = np.random.default_rng(2)
    vs = [rng.standard_normal(n) for _ in range(m)]

    def run_ordered(ordered):
        f = {}
        for inp in prog.inputs:
            for i, p in enumerate(prefixes):
                if inp.name.startswith(p):
                    f[inp.name] = ordered[i]
        return prog.run(f)["result"]

    base = run_ordered(vs)
    for sigma in perms(m):
        permuted = [vs[sigma[i]] for i in range(m)]
        np.testing.assert_allclose(run_ordered(permuted), sign(sigma) * base, atol=ATOL)


# ===========================================================================
# graded commutativity for general k,l: a∧b == (−1)^{kl} (b∧a).
# ===========================================================================


@pytest.mark.parametrize(
    ("n", "k", "l"),
    [(4, 2, 1), (4, 1, 2), (4, 2, 2)],
)
@pytest.mark.parametrize("seed", (0, 13))
def test_graded_commutativity_general_kl(n: int, k: int, l: int, seed: int) -> None:  # noqa: E741
    """``a∧b == (−1)^{kl} (b∧a)`` for degree-≥2 operands (compared in ambient V⊗^{k+l})."""
    V = G.space("V", n)

    def operand(name: str, deg: int) -> G.Expr:
        return G.project(G.var(name, tensor_power(V, deg)), exterior_power(V, deg))

    # a : Λᵏ, b : Λˡ. Build a∧b and b∧a as two separate programs; the operand vars carry
    # the SAME names ("a_amb"/"b_amb") with their fixed degrees so a single feed serves both.
    a = operand("a_amb", k)
    b = operand("b_amb", l)
    prog_ab = represent(G.embed(wedge(a, b, k, l), tensor_power(V, k + l)).node)
    prog_ba = represent(G.embed(wedge(b, a, l, k), tensor_power(V, k + l)).node)

    rng = np.random.default_rng(seed)
    av = rng.standard_normal((n,) * k)
    bv = rng.standard_normal((n,) * l)

    def feed(prog):
        f = {}
        for inp in prog.inputs:
            if inp.name.startswith("a_amb"):
                f[inp.name] = av
            elif inp.name.startswith("b_amb"):
                f[inp.name] = bv
        return f

    ab = prog_ab.run(feed(prog_ab))["result"]
    ba = prog_ba.run(feed(prog_ba))["result"]
    np.testing.assert_allclose(ab, ((-1) ** (k * l)) * ba, atol=ATOL)


# ===========================================================================
# numeric dim Λᵏ = C(n,k) via the SVD seam.
# ===========================================================================


@pytest.mark.parametrize("n", [2, 3, 4])
def test_dim_exterior_power_is_binomial(n: int) -> None:
    """``ImageBasis(Λᵏ)`` has exactly ``C(n,k)`` orthonormal columns for all 1≤k≤n."""
    V = G.space("V", n)
    for k in range(1, n + 1):
        P = image_basis_matrix(V, k)
        assert P.shape == (n**k, math.comb(n, k)), (n, k, P.shape)
        np.testing.assert_allclose(P.T @ P, np.eye(math.comb(n, k)), atol=ATOL)


# ===========================================================================
# Λᵏ = 0 for k > n (rank-0 / empty-basis SVD-seam probe).
# ===========================================================================


@pytest.mark.parametrize(("n", "k"), [(2, 3), (3, 4), (2, 4), (3, 5)])
def test_exterior_power_vanishes_above_dimension(n: int, k: int) -> None:
    """``Λᵏ(Vₙ) = 0`` for k>n: ``ImageBasis`` has 0 columns (C(n,k)=0).

    Probes the rank-0 / empty-basis seam: the SVD seam must produce a well-formed
    ``(nᵏ, 0)`` basis matrix without error.
    """
    assert math.comb(n, k) == 0
    V = G.space("V", n)
    P = image_basis_matrix(V, k)
    assert P.shape == (n**k, 0)


@pytest.mark.parametrize("n", [2, 3])
def test_wedge_above_dimension_is_zero(n: int) -> None:
    """``v₁∧…∧v_{n+1} = 0`` — more vectors than dimensions wedge to the zero tensor."""
    V = G.space("V", n)
    rng = np.random.default_rng(0)
    vs = [rng.standard_normal(n) for _ in range(n + 1)]
    T = wedge_all(V, vs)
    np.testing.assert_allclose(T, 0.0, atol=ATOL)
