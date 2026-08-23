r"""The operator-precedence model, authored once for every printer that needs one.

Precedence is a property of the algebra, not of a notation, so an :class:`Op` carries the
binding strength together with one surface symbol per back end — the LaTeX macro and the
plaintext glyph — and no LaTeX string appears in the structure itself. The LaTeX printer
is the reader today; a plaintext printer takes the other field and shares :data:`BINOPS`
and :func:`paren` unchanged, so the two cannot disagree about what binds tighter than
what. The Unicode surface syntax parenthesizes from its own table,
:mod:`grassmann.syntax.glyphs`, which a parser reads as well as a printer.

Higher binds tighter, and :func:`paren` wraps a child whose precedence falls below what
its slot demands. An infix operator renders its operands at its own level, so equal
precedence never parenthesizes — every binary operator here is associative or commutative,
which makes left and right associativity indistinguishable in the output. A postfix
decoration (``^{\dagger}``, ``^{*}``, …) is the exception: it renders its base demanding
one notch above :data:`Prec.POSTFIX`, so a base that is itself a postfix gets parentheses
and the illegal stacked superscript ``x^{a}^{b}`` never appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Prec(IntEnum):
    """Binding strength of a form, where a higher value binds tighter."""

    SUM = 10        # +  (Sum)  and  ⊕ (direct sum / stack)
    PROD = 20       # ⊗ (tensor), scalar multiple, ⊙ (Hadamard), Kronecker
    COMPOSE = 30    # ∘ (composition)
    POSTFIX = 40    # ^{†}, ^{*}, ^{-1}, ^{+}, ^{2}, ^{⊕n}
    APP = 50        # function application  f(x)
    ATOM = 60       # leaves and fully-delimited forms ⟨·,·⟩, \frac{}{}, \sqrt{}


@dataclass(frozen=True)
class Op:
    """A binary operator: one binding strength, and one surface symbol per back end.

    ``latex`` is the infix LaTeX macro and ``text`` the Unicode glyph the plaintext
    unparser prints. Holding both against a single ``prec`` is what makes the precedence
    table authored once rather than per printer.
    """

    name: str
    latex: str
    text: str
    prec: Prec


#: The binary-operator vocabulary, keyed by a name neither back end owns.
BINOPS: dict[str, Op] = {
    "sum":      Op("sum",      r"+",       "+",  Prec.SUM),
    "oplus":    Op("oplus",    r"\oplus",  "⊕",  Prec.SUM),
    "oprod":    Op("oprod",    r"\otimes", "⊗",  Prec.PROD),
    "scale":    Op("scale",    r"\cdot",   "·",  Prec.PROD),
    "hadamard": Op("hadamard", r"\odot",   "⊙",  Prec.PROD),
    "kron":     Op("kron",     r"\otimes", "⊗",  Prec.PROD),
    "compose":  Op("compose",  r"\circ",   "∘",  Prec.COMPOSE),
}


def paren(s: str, prec: int, min_prec: int) -> str:
    """Wrap the rendered string ``s`` in parentheses when it binds too loosely for its slot.

    A child rendered at precedence ``prec`` in a context demanding at least ``min_prec``
    needs parentheses exactly when ``prec < min_prec``; equal precedence does not, since
    every binary operator in the table is associative or commutative.
    """
    return f"({s})" if prec < min_prec else s


__all__ = ["Prec", "Op", "BINOPS", "paren"]
