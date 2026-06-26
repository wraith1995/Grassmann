"""Pretty-printers over the core IR (Goal 4).

Currently a single back end — a **term → LaTeX** printer (:mod:`grassmann.printer.latex`)
rendering any :class:`~grassmann.core.term.Term` / :class:`~grassmann.core.space.Space`
(and the output of :func:`grassmann.transform.diff.diff`) as readable mathematical LaTeX.

The operator/precedence vocabulary lives in :mod:`grassmann.printer.precedence`, kept
deliberately free of LaTeX-specific strings so a future plaintext unparser (goal 5) can
share the same precedence model.
"""

from __future__ import annotations

from .latex import latex

__all__ = ["latex"]
