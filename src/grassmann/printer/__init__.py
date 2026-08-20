"""Pretty-printers over the core IR.

The back end is a term-to-LaTeX printer (:mod:`grassmann.printer.latex`) that
renders any :class:`~grassmann.core.term.Term` or
:class:`~grassmann.core.space.Space`, and the output of
:func:`grassmann.transform.diff.diff`, as readable mathematical LaTeX.

The operator and precedence vocabulary lives in
:mod:`grassmann.printer.precedence`, kept free of LaTeX-specific strings so a
plaintext renderer can share the same precedence model.
"""

from __future__ import annotations

from .latex import latex

__all__ = ["latex"]
