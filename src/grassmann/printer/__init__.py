"""Pretty-printers that turn core IR back into mathematics a reader can check.

:mod:`grassmann.printer.latex` renders any :class:`~grassmann.core.term.Term` or
:class:`~grassmann.core.space.Space` as LaTeX, so a derivative built by
:func:`grassmann.transform.diff.diff` — an ordinary term — can be read as an equation
rather than as a node tree.

Where to parenthesize is not a LaTeX question, so it lives apart in
:mod:`grassmann.printer.precedence`, which holds the operator vocabulary and the binding
strengths with no LaTeX in the structure itself — each operator carries a plaintext glyph
beside its macro, so a second back end reuses the table rather than restating it. The
Unicode surface in :mod:`grassmann.syntax` keeps its own table,
:mod:`grassmann.syntax.glyphs`, because it must serve a parser as well as a printer.
"""

from __future__ import annotations

from .latex import latex

__all__ = ["latex"]
