"""IW5 — basic nonlinear scalar ops (derived ops, NOT core IR).

The elementwise nonlinear scalar operations the old FEMOracle ``base/algebra.py``
``BaseNonLinear`` provided (an "NEval-style" foreign op): ``sqrt`` / ``absval`` / ``sign``
/ ``div`` / ``reciprocal``, extended with the elementary transcendental, trigonometric and
hyperbolic library a matrix-calculus front end needs (``exp`` / ``log`` / ``square`` /
``sin`` / ``cos`` / ``tan`` / ``sinh`` / ``cosh`` / ``tanh``). Each builds a core
:class:`~grassmann.core.term.External`
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
from grassmann.core import space as _S
from grassmann.embed.ops import Expr
from grassmann.infer.synth import synth_space
from grassmann.lower.represent import ExternalOp

__all__ = [
    "sqrt",
    "absval",
    "sign",
    "div",
    "reciprocal",
    # elementary transcendental
    "exp",
    "log",
    "square",
    # trigonometric
    "sin",
    "cos",
    "tan",
    # hyperbolic
    "sinh",
    "cosh",
    "tanh",
    # inverse trigonometric
    "atan",
    "asin",
    "acos",
    # activations
    "sigmoid",
    "softplus",
    # pointwise (Hadamard) — the external linear op
    "hadamard",
    # constants
    "ones",
]


def _is_euclidean(s: object) -> bool:
    """Report whether ``s`` is built purely from base/``Scalar`` spaces via ``Dual``/``OPlus``/``OProd``.

    Euclidean = standard inner product, standard basis: ``Scalar``, an abstract base space
    (``SVar``), and any ``Dual``/``OPlus``/``OProd``/``OPlusRep`` thereof. Explicitly NOT
    Euclidean: ``WithInner`` (a re-metricized space) and the FFS/moving-subspace spaces
    ``Img``/``Ker``/``CoImg``/``CoKer`` — elementwise ops are only meaningful in standard
    coordinates, so we refuse those rather than silently apply the kernel off-basis.
    """
    match s:
        case _S.Scalar() | _S.SVar():
            return True
        case _S.Dual(space=inner) | _S.OPlusRep(space=inner):
            return _is_euclidean(inner)
        case _S.OPlus(spaces=ss) | _S.OProd(spaces=ss):
            return all(_is_euclidean(t) for t in ss)
        case _:  # WithInner, Img/Ker/CoImg/CoKer, App, SpaceMeta, …  # audit: ok EX-WILDCARD — documented refuse-default; refusal is conservative [approved 2026-07-03]
            return False


def _euclidean_space_of(x: Expr) -> _S.Space:
    """Give the (Euclidean) space of operand ``x`` — the result space of an elementwise op.

    Elementwise ops preserve the operand's space (``exp`` of a vector is a vector). We infer
    it with :func:`synth_space` and require it Euclidean, so the kernel runs in standard
    coordinates on any value built from Euclidean spaces.
    """
    s = synth_space(x.node if isinstance(x, Expr) else x)
    if not _is_euclidean(s):
        raise ValueError(
            f"stdlib scalar op: operand space {s!r} is not Euclidean (no FFS / Remetric). "
            f"Elementwise nonlinear ops require a value built from Euclidean spaces."
        )
    return s


def _external(name: str, fn: object, *args: Expr) -> Expr:
    """Build an ``External`` applying ``fn`` elementwise to ``args``.

    The result lives in the operand's (Euclidean) space — so these ops run elementwise on any
    value built from Euclidean spaces (scalars, vectors, matrices, ⊗/⊕ thereof), not just
    ``Scalar``. ``BasisReq.STANDARD`` (the default) makes the kernel run in standard coords.
    """
    space = _euclidean_space_of(args[0])
    return G.external(space, ExternalOp(name, fn), *args)


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


# --- elementary transcendental ---------------------------------------------


def exp(x: Expr) -> Expr:
    """``eˣ`` — a ``Scalar`` (``np.exp``)."""
    return _external("exp", np.exp, x)


def log(x: Expr) -> Expr:
    """``ln x`` — a ``Scalar`` (``np.log``, natural log). Real-valued: ``x > 0``."""
    return _external("log", np.log, x)


def _square_fn(a: object) -> object:
    return np.square(a)


def square(x: Expr) -> Expr:
    """``x²`` — a ``Scalar`` (``np.square``)."""
    return _external("square", _square_fn, x)


# --- trigonometric ---------------------------------------------------------


def sin(x: Expr) -> Expr:
    """``sin x`` — a ``Scalar`` (``np.sin``)."""
    return _external("sin", np.sin, x)


def cos(x: Expr) -> Expr:
    """``cos x`` — a ``Scalar`` (``np.cos``)."""
    return _external("cos", np.cos, x)


def tan(x: Expr) -> Expr:
    """``tan x`` — a ``Scalar`` (``np.tan``)."""
    return _external("tan", np.tan, x)


# --- hyperbolic ------------------------------------------------------------


def sinh(x: Expr) -> Expr:
    """``sinh x`` — a ``Scalar`` (``np.sinh``)."""
    return _external("sinh", np.sinh, x)


def cosh(x: Expr) -> Expr:
    """``cosh x`` — a ``Scalar`` (``np.cosh``)."""
    return _external("cosh", np.cosh, x)


def tanh(x: Expr) -> Expr:
    """``tanh x`` — a ``Scalar`` (``np.tanh``)."""
    return _external("tanh", np.tanh, x)


# --- inverse trigonometric -------------------------------------------------


def atan(x: Expr) -> Expr:
    """``arctan x`` — a ``Scalar`` (``np.arctan``)."""
    return _external("atan", np.arctan, x)


def asin(x: Expr) -> Expr:
    """``arcsin x`` — a ``Scalar`` (``np.arcsin``). Real-valued: ``−1 ≤ x ≤ 1``."""
    return _external("asin", np.arcsin, x)


def acos(x: Expr) -> Expr:
    """``arccos x`` — a ``Scalar`` (``np.arccos``). Real-valued: ``−1 ≤ x ≤ 1``."""
    return _external("acos", np.arccos, x)


# --- activations -----------------------------------------------------------


def _sigmoid_fn(a: object) -> object:
    from scipy.special import expit

    return expit(a)


def sigmoid(x: Expr) -> Expr:
    """``σ(x) = 1/(1+e⁻ˣ)`` — the logistic function (``scipy.special.expit``)."""
    return _external("sigmoid", _sigmoid_fn, x)


def _softplus_fn(a: object) -> object:
    return np.logaddexp(0.0, a)


def softplus(x: Expr) -> Expr:
    """``softplus(x) = log(1+eˣ)`` — a ``Scalar`` (stable ``np.logaddexp(0, x)``)."""
    return _external("softplus", _softplus_fn, x)


# --- pointwise (Hadamard) product — the external linear op ------------------


def hadamard(x: Expr, y: Expr) -> Expr:
    """``x ⊙ y`` — the elementwise (Hadamard) product (``np.multiply``).

    ``x`` and ``y`` live in the same Euclidean space; the result is in that space. This is
    the pointwise *linear* op (linear in each operand) the matrix-calculus front end needs —
    e.g. the elementwise chain rule ``d f(u) = f'(u) ⊙ du`` differentiates through it.
    """
    return _external("hadamard", np.multiply, x, y)


# --- constants -------------------------------------------------------------


def _ones_fn(a: object) -> object:
    return np.ones_like(np.asarray(a, dtype=float))


def ones(x: Expr) -> Expr:
    """``𝟙`` — a constant value of ``1`` in ``x``'s (Euclidean) space, shaped like ``x``.

    The unit constant the derivative arms use for ``1 ± g(u)`` (e.g. ``atan'`` / ``sigmoid'``;
    see :mod:`grassmann.transform.diff`). Built elementwise (``np.ones_like``), so it
    broadcasts to any Euclidean shape; its derivative is zero. Public so the surface syntax
    (:mod:`grassmann.syntax`) can round-trip a derivative term that contains it.
    """
    return _external("ones", _ones_fn, x)
