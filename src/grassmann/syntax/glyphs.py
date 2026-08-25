"""The one table the Unicode surface syntax is defined by — its glyphs and their precedence.

The parser (:mod:`grassmann.syntax.parse`) and the unparser
(:mod:`grassmann.syntax.unparse`) both read it, which is what keeps a round trip exact:
neither side can invent a glyph or a binding power the other does not know. The LaTeX
printer parenthesizes on the same principles but from its own table,
:mod:`grassmann.printer.precedence`, since a printer with no parser to match has no
reason to be constrained by this one.

The operators fall into a few kinds:

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
# The glyphs themselves — one name per surface symbol, so lexer, parser and
# unparser all spell it the same way.
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

# The precedence each node family prints at, which is what the unparser compares against
# its slot to decide on parentheses. Mirrors the binding powers above, so what the parser
# binds tightly the unparser also prints tightly.
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
# Call-form heads: names the resolver reads as constructors rather than as
# variables. The lexer treats them as ordinary identifiers, so a name is
# reserved only in head position — `Img` heading `Img(f)` is the constructor,
# while `Img` alone is just a variable name.
# ===========================================================================

# term constructors
TERM_HEADS = frozenset({
    "id", "zero", "Eval", "NEval", "Compose", "Trace", "det", "logdet",
    "inner", "flat", "sharp", "Solve", "Pinv", "corestrict", "Dagger", "sqrtm",
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

#: The nonlinear and pointwise stdlib ops, keyed by their surface head. Each maps to the
#: :mod:`grassmann.stdlib.scalar` builder that rebuilds it, so parsing reconstructs the
#: identical ``External(ExternalOp(name, fn), …)`` node — same ``fn`` object, hence equal
#: under ``==`` — rather than a look-alike that would break the round trip.
EXTERNAL_OPS: dict[str, str] = {
    "sqrt": "sqrt", "abs": "absval", "sign": "sign", "div": "div",
    "reciprocal": "reciprocal", "exp": "exp", "log": "log", "square": "square",
    "sin": "sin", "cos": "cos", "tan": "tan", "sinh": "sinh", "cosh": "cosh",
    "tanh": "tanh", "atan": "atan", "asin": "asin", "acos": "acos",
    "sigmoid": "sigmoid", "softplus": "softplus", "hadamard": "hadamard",
    "ones": "ones",  # the diff unit-constant (1±g(u)); reconstructs via stdlib.scalar.ones
}
#: The reverse map, from an ``ExternalOp.name`` back to its surface head. Almost the
#: identity, since an op is usually named after the head that builds it.
EXTERNAL_NAME_TO_HEAD: dict[str, str] = {k: k for k in EXTERNAL_OPS}

ALL_HEADS = (
    TERM_HEADS | SPACE_HEADS | BASIS_HEADS | DIM_HEADS | TYPE_HEADS
    | frozenset(EXTERNAL_OPS)
)

# The sort keywords, usable as a Lam/BLam binder annotation or as a Kind.
SORT_KEYWORDS = frozenset({"Space", "Nat", "Type", "Basis"})
