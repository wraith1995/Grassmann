"""Operator-precedence model shared by the printers (`27` LaTeX / goal 5 unparse).

Deliberately LaTeX-agnostic at the *structure* level: an :class:`Op` carries both the
precedence and the family of surface symbols (LaTeX now, a plaintext symbol for the
future Unicode unparser). A printer picks the field it needs and reuses :data:`BINOPS`
plus :func:`paren` for parenthesization, so the two back ends cannot drift apart on what
binds tighter than what.

Precedence: **higher binds tighter**. A child rendered in a context demanding a minimum
precedence ``min_prec`` is wrapped in parentheses iff its own precedence is below that
minimum (:func:`paren`). Infix associative operators render their operands at the
operator's own level (equal precedence ⇒ no parentheses, i.e. left/right associative is
treated uniformly — every binary op here is associative or commutative). A postfix
decoration (``^{\\dagger}``, ``^{*}``, …) renders its base at :data:`Prec.POSTFIX` so a
looser base — and crucially another postfix — is parenthesized, avoiding illegal stacked
superscripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Prec(IntEnum):
    """Binding strength; higher binds tighter."""

    SUM = 10        # +  (Sum)  and  ⊕ (direct sum / stack)
    PROD = 20       # ⊗ (tensor), scalar multiple, ⊙ (Hadamard), Kronecker
    COMPOSE = 30    # ∘ (composition)
    POSTFIX = 40    # ^{†}, ^{*}, ^{-1}, ^{+}, ^{2}, ^{⊕n}
    APP = 50        # function application  f(x)
    ATOM = 60       # leaves and fully-delimited forms ⟨·,·⟩, \frac{}{}, \sqrt{}


@dataclass(frozen=True)
class Op:
    """A binary operator: its precedence plus the surface symbol per back end.

    ``latex`` is the infix LaTeX symbol; ``text`` is the plaintext/Unicode symbol the
    goal-5 unparser will use. Keeping both here means the precedence table is authored
    once and the two printers stay in lock-step.
    """

    name: str
    latex: str
    text: str
    prec: Prec


# The binary-operator vocabulary, keyed by a back-end-neutral name.
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
    """Wrap ``s`` in parentheses iff its precedence ``prec`` is below ``min_prec``."""
    return f"({s})" if prec < min_prec else s


__all__ = ["Prec", "Op", "BINOPS", "paren"]
