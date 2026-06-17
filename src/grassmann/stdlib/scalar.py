"""IW5 — basic nonlinear scalar ops (derived ops, NOT core IR).

The elementwise nonlinear scalar operations the old FEMOracle ``base/algebra.py``
``BaseNonLinear`` provided (an "NEval-style" foreign op): ``sqrt`` / ``absval`` / ``sign``
/ ``div`` / ``reciprocal``. Each builds a core :class:`~grassmann.core.term.External`
node over ``Scalar`` operands, whose ``data`` is an
:class:`~grassmann.lower.represent.ExternalOp` naming the op and carrying the numpy
callable — so the node ``repr`` and the lowering provenance (``external:<op>``) are
meaningful, and :func:`grassmann.lower.represent.represent` lowers it to an elementwise
:class:`~polyarray.Stmt`.

Real-valued. Nothing here is a core node: these compose the existing ``External``
mechanism, exactly like the rest of the stdlib composes the embed surface.
"""

from __future__ import annotations

import numpy as np

import grassmann.embed as G
from grassmann.embed.ops import Expr
from grassmann.lower.represent import ExternalOp

__all__ = ["sqrt", "absval", "sign", "div", "reciprocal"]


def _external(name: str, fn: object, *args: Expr) -> Expr:
    """A ``Scalar``-valued ``External`` applying ``fn`` elementwise to ``args`` (`41 §3`)."""
    return G.external(G.R, ExternalOp(name, fn), *args)


def sqrt(x: Expr) -> Expr:
    """``√x`` — a ``Scalar`` (``np.sqrt`` elementwise)."""
    return _external("sqrt", np.sqrt, x)


def absval(x: Expr) -> Expr:
    """``|x|`` — a ``Scalar`` (``np.abs``). Named ``absval`` to avoid shadowing builtin ``abs``."""
    return _external("abs", np.abs, x)


def sign(x: Expr) -> Expr:
    """``sgn(x)`` — a ``Scalar`` (``np.sign``)."""
    return _external("sign", np.sign, x)


def div(x: Expr, y: Expr) -> Expr:
    """``x / y`` — a ``Scalar`` (elementwise division)."""
    return _external("div", np.divide, x, y)


def _reciprocal_fn(a: object) -> object:
    return np.divide(1.0, a)


def reciprocal(x: Expr) -> Expr:
    """``1 / x`` — a ``Scalar`` (elementwise reciprocal over floats)."""
    return _external("reciprocal", _reciprocal_fn, x)
