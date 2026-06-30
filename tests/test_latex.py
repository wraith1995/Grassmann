"""Golden-output tests for the LaTeX printer (`27` goal 4).

Builds representative expressions (algebra, the nonlinear/pointwise stdlib ops, spaces,
and a ``G.diff`` output) and asserts the exact LaTeX string, pinning notation,
precedence, and parenthesization. Pure printing — no lowering — so these run inline.
"""

from __future__ import annotations

import grassmann as G
from grassmann.core import kinds as K
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.printer.latex import latex as raw_latex

# ===========================================================================
# Fixtures — a small reusable cast.
# ===========================================================================

V = G.space("V", 3)              # ⊕³ ℝ  → ℝ^{3}
W = G.space("W", 2)              # ⊕² ℝ  → ℝ^{2}
U = G.svar("U")                  # an abstract base space (keeps its name)
f = G.var("f", V >> W)           # a linear map ℝ^{3} ⊸ ℝ^{2}
A = G.var("A", V >> V)           # an endomorphism
x = G.var("x", V)
y = G.var("y", V)


# ===========================================================================
# Algebra / application.
# ===========================================================================


def test_var_and_apply():
    assert G.latex(x) == "x"
    assert G.latex(f(x)) == "f(x)"


def test_sum_and_scale_precedence():
    # Scale binds tighter than Sum ⇒ no parens around the product, parens when a sum
    # appears under a scale.
    assert G.latex(G.scale(2.0, x) + y) == r"2 \cdot x + y"
    assert G.latex(G.scale(2.0, x + y)) == r"2 \cdot (x + y)"


def test_tensor_oplus_and_inner_pairing():
    assert G.latex(x * y) == r"x \otimes y"
    assert G.latex(x & y) == r"x \oplus y"
    assert G.latex(G.dot(x, y)) == r"\langle x, y\rangle"


def test_tensor_over_sum_parenthesizes():
    assert G.latex((x + y) * x) == r"(x + y) \otimes x"


def test_compose():
    assert G.latex(A @ A) == r"A \circ A"


def test_nondefault_positions_surface_as_subscripts():
    assert raw_latex(T.Eval(A.node, (0, 2), (A.node, A.node))) == r"A_{0,2}(A, A)"
    assert raw_latex(T.Compose(A.node, (0, 1), (A.node, A.node))) == r"A \circ_{0,1} (A, A)"


# ===========================================================================
# Inverses / adjoint / musical maps — distinct postfix superscripts.
# ===========================================================================


def test_adjoint_pinv_solve_are_distinct():
    assert G.latex(f.dagger) == r"f^{\dagger}"
    assert G.latex(f.pinv) == r"f^{+}"
    assert G.latex(A.solve) == r"A^{-1}"
    assert G.latex(A.solve(x)) == r"A^{-1}(x)"
    assert G.latex(f.pinv(x)) == r"f^{+}(x)"


def test_stacked_postfix_parenthesizes():
    # No illegal ``x^{a}^{b}`` — the inner postfix is wrapped.
    assert G.latex(f.dagger.dagger) == r"(f^{\dagger})^{\dagger}"
    assert raw_latex(T.Dagger(T.Solve(A.node))) == r"(A^{-1})^{\dagger}"


def test_musical_maps():
    assert G.latex(G.flat(V)(x)) == r"x^{\flat}"
    assert G.latex(G.sharp(V)(x)) == r"x^{\sharp}"
    assert G.latex(G.inner(V)) == r"\langle\cdot,\cdot\rangle_{\mathbb{R}^{3}}"


# ===========================================================================
# Nonlinear / pointwise stdlib ops (the new corpus).
# ===========================================================================


def test_elementwise_function_ops():
    assert G.latex(G.exp(x)) == r"\exp(x)"
    assert G.latex(G.log(x)) == r"\log(x)"
    assert G.latex(G.sin(x)) == r"\sin(x)"
    assert G.latex(G.tanh(x)) == r"\tanh(x)"
    assert G.latex(G.atan(x)) == r"\arctan(x)"
    assert G.latex(G.sigmoid(x)) == r"\sigma(x)"
    assert G.latex(G.softplus(x)) == r"\operatorname{softplus}(x)"
    assert G.latex(G.sign(x)) == r"\operatorname{sgn}(x)"
    # Nested application stays atomic (no spurious parens around the inner call).
    assert G.latex(G.sin(G.exp(x))) == r"\sin(\exp(x))"


def test_structural_scalar_ops():
    assert G.latex(G.sqrt(x)) == r"\sqrt{x}"
    assert G.latex(G.absval(x)) == r"\lvert x \rvert"
    assert G.latex(G.square(x)) == r"x^{2}"
    assert G.latex(G.reciprocal(x)) == r"\frac{1}{x}"
    assert G.latex(G.div(x, y)) == r"\frac{x}{y}"


def test_hadamard_is_infix():
    assert G.latex(G.hadamard(x, y)) == r"x \odot y"
    # Hadamard (product-level) over a sum parenthesizes the sum.
    assert G.latex(G.hadamard(x + y, x)) == r"(x + y) \odot x"


# ===========================================================================
# Spaces.
# ===========================================================================


def test_spaces():
    assert G.latex(V) == r"\mathbb{R}^{3}"
    assert G.latex(~U) == r"U^{*}"
    assert G.latex(V * W) == r"\mathbb{R}^{3} \otimes \mathbb{R}^{2}"
    assert G.latex(U & W) == r"U \oplus \mathbb{R}^{2}"
    # Map space ℝ^{3} ⊸ ℝ^{2} = (ℝ^{3})^{*} ⊗ ℝ^{2}; dual of a superscripted base wraps.
    assert G.latex(V >> W) == r"(\mathbb{R}^{3})^{*} \otimes \mathbb{R}^{2}"


def test_ffs_and_oplusrep():
    assert G.latex(G.ker(f.node)) == r"\ker f"
    assert G.latex(G.img(f.node)) == r"\operatorname{im} f"
    assert raw_latex(S.OPlusRep(S.DimVar("n"), S.SVar("U"))) == r"U^{\oplus n}"
    # DimNeg in a count folds into subtraction.
    rep = S.OPlusRep(S.DimSum((S.DimVar("n"), S.DimNeg(S.DimLit(1)))), S.Scalar())
    assert raw_latex(rep) == r"\mathbb{R}^{n - 1}"


# ===========================================================================
# Binders & diff output.
# ===========================================================================


def test_lam_binder():
    lam = T.Lam((("a", K.SPACE),), S.oprod(S.SVar("a"), S.SVar("a")))
    assert raw_latex(lam) == r"\Lambda a : \mathrm{Space}.\; a \otimes a"


def test_diff_linear_map():
    # d/dx f(x) = (dx ↦ f(dx)) — an LLam, rendered with \lambda.
    d = G.diff(f(x), x.node)
    assert G.latex(d) == r"\lambda dx.\; f(dx)"


def test_diff_inner_product_has_correct_structure():
    # d/dx ⟨x, x⟩ = (dx ↦ ⟨dx, x⟩ + ⟨x, dx⟩) — product rule, sum at the top level.
    d = G.diff(G.dot(x, x), x.node)
    assert G.latex(d) == r"\lambda dx.\; \langle dx, x\rangle + \langle x, dx\rangle"


def test_diff_chain_rule_through_exp():
    # d/dx exp(x) = (dx ↦ exp(x) ⊙ dx) — the elementwise chain rule.
    d = G.diff(G.exp(x), x.node)
    assert G.latex(d) == r"\lambda dx.\; \exp(x) \odot dx"


def test_diff_inner_product_advanced_eta_pretty():
    # Advanced-η (opt-in RIESZ_SOP) folds the noisy lambda d/dx⟨x,x⟩ into the covector 2·x♭.
    # The default-path pin above (test_diff_inner_product_has_correct_structure) is unchanged.
    from grassmann.transform.rewrite import RIESZ_SOP

    d = G.diff(G.dot(x, x), x.node)
    folded = G.normal_form(d, RIESZ_SOP)
    assert G.latex(folded) == r"2 \cdot x^{\flat}"


def test_latex_accepts_raw_term_and_expr():
    # Both the embed Expr handle and the raw core node render identically.
    assert G.latex(f(x)) == raw_latex(f(x).node)
