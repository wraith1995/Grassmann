"""Single source of truth for the Unicode surface syntax — glyphs + precedence.

The parser (:mod:`grassmann.syntax.parse`) and unparser (:mod:`grassmann.syntax.unparse`)
both read this table so the two stay in lock-step. A goal-04 LaTeX printer can consume the
SAME operator/precedence model (it only needs to swap the glyph strings for LaTeX macros).

Design (see ``plans/27-surface-syntax.md`` for the operator allocation):

- **Infix** binary operators carry a *binding power* (higher binds tighter). All are
  left-associative. They are deliberately **sort-polymorphic**: ``⊗`` is ``OProd`` on
  spaces, ``OProdV`` on vectors, ``BasisProd`` on bases; the resolver picks by operand sort.
- **Postfix** unary operators (``*`` dual, ``†`` dagger, ``⁻¹`` solve, ``⁺`` pinv) bind
  tighter than application.
- **Application** ``f(a, b)`` (Eval / App) is the tightest binary form.
- Glyphs that are not arithmetic (``⟨ ⟩`` inner, ``λ Λ`` binders, ``ℝ ⊥`` atoms, the
  ``Head(...)`` call forms) are handled structurally, not by binding power.

Round-trip fidelity hinges on the unparser parenthesizing exactly when a child's output
precedence is below the slot it sits in; the parser then re-derives the same tree.
"""

from __future__ import annotations

# ===========================================================================
# Headline glyphs (the source of truth shared with the LaTeX printer).
# ===========================================================================

# structure
INNER_L = "⟨"   # ⟨
INNER_R = "⟩"   # ⟩
COMPOSE = "∘"   # ∘
TENSOR = "⊗"    # ⊗
OPLUS = "⊕"     # ⊕
OPLUS_BOT = "⊥" # ⊥  (nullary ⊕)
HADAMARD = "⊙"  # ⊙
DAGGER = "†"    # †
INVERSE = "⁻¹"  # ⁻¹  (Solve)
PINV = "⁺"      # ⁺  (Pinv)
DUAL = "*"           # postfix dual (spaces / bases)
SCALE = "·"     # ·  (Scale on terms; DimProd on dims)
SUM = "+"
SUB = "-"            # binary dim subtraction / unary DimNeg
MOD = "%"            # DimMod
REP = "^"            # OPlusRep / BasisRep:  base ^ count
SCALAR = "ℝ"    # ℝ
ARROW = "→"     # →  (Arrow kinds / function types)
TURNSTILE = "⊢" # ⊢  (separates the typed-variable preamble from the body)
LAM = "λ"       # λ  (LLam — linear lambda)
BIGLAM = "Λ"    # Λ  (Lam / BLam — general binder, sort-directed)
SQRT = "√"      # √  (External "sqrt")
TYPE_L = "⟦"    # ⟦
TYPE_R = "⟧"    # ⟧  (SpaceAsType)
SCALAR_BASIS = "e" + SCALAR   # eℝ  (ScalarBasis)

# ===========================================================================
# Binding powers (precedence) — higher binds tighter. All left-associative.
# ===========================================================================

#: infix glyph -> binding power
INFIX_BP: dict[str, int] = {
    ARROW: 5,        # →  arrow kinds / function types (loosest)
    OPLUS: 10,       # ⊕  direct sum / stack  (loosest binary)
    SUM: 20,         # +  Sum / DimSum
    SUB: 20,         # -  DimSum with DimNeg
    SCALE: 30,       # ·  Scale (term) / DimProd (dim)
    MOD: 30,         # %  DimMod
    TENSOR: 40,      # ⊗  tensor
    COMPOSE: 50,     # ∘  Compose
    HADAMARD: 60,    # ⊙  Hadamard
    REP: 70,         # ^  OPlusRep / BasisRep
}

#: postfix glyph -> binding power (all tighter than application)
POSTFIX_BP: dict[str, int] = {
    DUAL: 80,
    DAGGER: 80,
    INVERSE: 80,
    PINV: 80,
}

#: binding power of application  f(...) / g[...]
APP_BP = 90
#: precedence of an atom (never needs wrapping)
ATOM_BP = 100

# Output-precedence each *node family* prints at — used by the unparser to decide
# parenthesization. Mirrors the binding powers above.
P_OPLUS = 10
P_SUM = 20
P_SCALE = 30
P_MOD = 30
P_TENSOR = 40
P_COMPOSE = 50
P_HADAMARD = 60
P_REP = 70
P_POSTFIX = 80
P_APP = 90
P_ATOM = 100


# ===========================================================================
# Reserved call-form heads (a NAME immediately interpreted as a constructor,
# not a variable).  The resolver dispatches on these; the lexer treats them as
# ordinary identifiers, so they are reserved only in *head* position.
# ===========================================================================

# term constructors
TERM_HEADS = frozenset({
    "id", "zero", "Eval", "NEval", "Compose", "Trace", "det", "logdet",
    "inner", "flat", "sharp", "Solve", "Pinv", "Dagger", "sqrtm",
    "SolveApply", "PinvApply", "project", "embed", "assert",
    "inj", "proj", "remetric", "lift", "unlift", "permute", "tpermute", "kron",
    "external",
})
# space constructors
SPACE_HEADS = frozenset({
    "Ker", "Img", "CoKer", "CoImg", "withinner", "App",
})
# basis constructors
BASIS_HEADS = frozenset({
    "primal", "pbasis", "ibasis", "lbasis", "withinner_basis",
    "orthonormal", "complete", "BApp",
})
# dim constructors
DIM_HEADS = frozenset({"dim", "delta", "mod"})
# type constructors
TYPE_HEADS = frozenset({"Pi", "TypeApp"})

#: nonlinear / pointwise stdlib External ops, by surface head.  Maps the surface
#: name to the ``grassmann.stdlib.scalar`` builder used to reconstruct the SAME
#: ``External(ExternalOp(name, fn), ...)`` node (identical fn object ⇒ ``==``).
EXTERNAL_OPS: dict[str, str] = {
    "sqrt": "sqrt", "abs": "absval", "sign": "sign", "div": "div",
    "reciprocal": "reciprocal", "exp": "exp", "log": "log", "square": "square",
    "sin": "sin", "cos": "cos", "tan": "tan", "sinh": "sinh", "cosh": "cosh",
    "tanh": "tanh", "atan": "atan", "asin": "asin", "acos": "acos",
    "sigmoid": "sigmoid", "softplus": "softplus", "hadamard": "hadamard",
    "ones": "ones",  # the diff unit-constant (1±g(u)); reconstructs via stdlib.scalar.ones
}
#: reverse: ExternalOp.name -> surface head (mostly identity; "abs" stays "abs").
EXTERNAL_NAME_TO_HEAD: dict[str, str] = {k: k for k in EXTERNAL_OPS}

ALL_HEADS = (
    TERM_HEADS | SPACE_HEADS | BASIS_HEADS | DIM_HEADS | TYPE_HEADS
    | frozenset(EXTERNAL_OPS)
)

# sort keywords usable as a Lam/BLam binder annotation or a Kind.
SORT_KEYWORDS = frozenset({"Space", "Nat", "Type", "Basis"})
