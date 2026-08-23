"""The elementwise nonlinearities — square root, exponential, the trigonometric family.

Linear algebra is closed under grassmann's core operations; ``exp`` and ``sqrt`` are not,
so they enter the language the way any foreign operator does, as an
:class:`~grassmann.core.term.External` node. Each function here builds one, with an
:class:`~grassmann.lower.represent.ExternalOp` as its payload carrying both the operator's
name and its numpy kernel. Naming it is what makes the node's ``repr`` and the lowering
provenance ``external:<op>`` readable, and :func:`grassmann.lower.represent.represent`
turns the node into a single elementwise :class:`~polyarray.Stmt`.

The ops apply elementwise, so an operand need not be a ``Scalar``: any value built from
Euclidean spaces works — a vector, a matrix, a ⊗ or ⊕ of them — and the result stays in
the operand's own space. Euclidean is the real requirement, because a kernel runs on
coordinates and only the standard basis makes those coordinates the operand's components;
a metricized or fundamental-subspace operand is refused rather than silently computed
off-basis.

Everything is real-valued, and nothing here is a core node.
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
    """Report whether ``s`` is Euclidean — standard inner product, standard basis.

    That holds for ``Scalar``, for an abstract base space ``SVar``, and for any
    ``Dual`` / ``OPlus`` / ``OProd`` / ``OPlusRep`` built from those, so the test recurses
    through the structure. It fails for a ``WithInner`` space, whose metric is not the
    standard one, and for the fundamental subspaces ``Img`` / ``Ker`` / ``CoImg`` /
    ``CoKer``, whose frame is not the standard one. An elementwise kernel is only
    meaningful in standard coordinates, so those are refused rather than computed
    off-basis.
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
    """Give the Euclidean space of the operand ``x``, which is also the op's result space.

    An elementwise op preserves its operand's space — ``exp`` of a vector is a vector — so
    the two coincide. The space is synthesized with :func:`synth_space` and required to be
    Euclidean, so the kernel is only ever handed standard coordinates.

    Raises
    ------
    ValueError
        If the operand's space is not Euclidean.
    """
    s = synth_space(x.node if isinstance(x, Expr) else x)
    if not _is_euclidean(s):
        raise ValueError(
            f"stdlib scalar op: operand space {s!r} is not Euclidean (no FFS / Remetric). "
            f"Elementwise nonlinear ops require a value built from Euclidean spaces."
        )
    return s


def _external(name: str, fn: object, *args: Expr) -> Expr:
    """Build the ``External`` node applying the kernel ``fn`` elementwise to ``args``.

    The result space is the first operand's, checked Euclidean, so an op applies to a
    scalar, a vector, a matrix or any ⊗/⊕ of them alike. The default ``BasisReq.STANDARD``
    on the operator is what guarantees the kernel sees standard coordinates.
    """
    space = _euclidean_space_of(args[0])
    return G.external(space, ExternalOp(name, fn), *args)


def sqrt(x: Expr) -> Expr:
    """Take the elementwise square root of ``x``, staying in ``x``'s space."""
    return _external("sqrt", np.sqrt, x)


def absval(x: Expr) -> Expr:
    """Take the elementwise absolute value of ``x``, staying in ``x``'s space.

    Spelled ``absval`` so it does not shadow the builtin ``abs``.
    """
    return _external("abs", np.abs, x)


def sign(x: Expr) -> Expr:
    """Take the elementwise sign of ``x``, giving −1, 0 or +1 per component."""
    return _external("sign", np.sign, x)


def div(x: Expr, y: Expr) -> Expr:
    """Divide ``x`` by ``y`` elementwise, both operands sharing one space."""
    return _external("div", np.divide, x, y)


def _reciprocal_fn(a: object) -> object:
    return np.divide(1.0, a)


def reciprocal(x: Expr) -> Expr:
    """Take the elementwise reciprocal ``1 / x`` over floats."""
    return _external("reciprocal", _reciprocal_fn, x)


# --- elementary transcendental ---------------------------------------------


def exp(x: Expr) -> Expr:
    """Take the elementwise exponential ``eˣ``."""
    return _external("exp", np.exp, x)


def log(x: Expr) -> Expr:
    """Take the elementwise natural logarithm ``ln x``, which needs ``x > 0`` to stay real."""
    return _external("log", np.log, x)


def _square_fn(a: object) -> object:
    return np.square(a)


def square(x: Expr) -> Expr:
    """Square ``x`` elementwise — componentwise, not as a matrix product."""
    return _external("square", _square_fn, x)


# --- trigonometric ---------------------------------------------------------


def sin(x: Expr) -> Expr:
    """Take the elementwise sine of ``x``."""
    return _external("sin", np.sin, x)


def cos(x: Expr) -> Expr:
    """Take the elementwise cosine of ``x``."""
    return _external("cos", np.cos, x)


def tan(x: Expr) -> Expr:
    """Take the elementwise tangent of ``x``."""
    return _external("tan", np.tan, x)


# --- hyperbolic ------------------------------------------------------------


def sinh(x: Expr) -> Expr:
    """Take the elementwise hyperbolic sine of ``x``."""
    return _external("sinh", np.sinh, x)


def cosh(x: Expr) -> Expr:
    """Take the elementwise hyperbolic cosine of ``x``."""
    return _external("cosh", np.cosh, x)


def tanh(x: Expr) -> Expr:
    """Take the elementwise hyperbolic tangent of ``x``."""
    return _external("tanh", np.tanh, x)


# --- inverse trigonometric -------------------------------------------------


def atan(x: Expr) -> Expr:
    """Take the elementwise inverse tangent of ``x``."""
    return _external("atan", np.arctan, x)


def asin(x: Expr) -> Expr:
    """Take the elementwise inverse sine of ``x``, which needs ``−1 ≤ x ≤ 1`` to stay real."""
    return _external("asin", np.arcsin, x)


def acos(x: Expr) -> Expr:
    """Take the elementwise inverse cosine of ``x``, which needs ``−1 ≤ x ≤ 1`` to stay real."""
    return _external("acos", np.arccos, x)


# --- activations -----------------------------------------------------------


def _sigmoid_fn(a: object) -> object:
    from scipy.special import expit

    return expit(a)


def sigmoid(x: Expr) -> Expr:
    """Apply the logistic function ``σ(x) = 1/(1+e⁻ˣ)`` elementwise."""
    return _external("sigmoid", _sigmoid_fn, x)


def _softplus_fn(a: object) -> object:
    return np.logaddexp(0.0, a)


def softplus(x: Expr) -> Expr:
    """Apply ``softplus(x) = log(1+eˣ)`` elementwise, via the stable ``logaddexp`` form."""
    return _external("softplus", _softplus_fn, x)


# --- pointwise (Hadamard) product — the external linear op ------------------


def hadamard(x: Expr, y: Expr) -> Expr:
    """Multiply ``x`` and ``y`` elementwise — the Hadamard product ``x ⊙ y``.

    Both operands live in one Euclidean space and so does the result. Unlike its
    neighbours here this op is linear in each argument separately, which is what makes the
    elementwise chain rule ``d f(u) = f'(u) ⊙ du`` expressible and lets
    :mod:`grassmann.transform.diff` differentiate through it.
    """
    return _external("hadamard", np.multiply, x, y)


# --- constants -------------------------------------------------------------


def _ones_fn(a: object) -> object:
    return np.ones_like(np.asarray(a, dtype=float))


def ones(x: Expr) -> Expr:
    """Give the constant ``1`` in ``x``'s Euclidean space, shaped like ``x``.

    The unit the derivative arms need in order to write ``1 ± g(u)`` — the shape ``atan'``
    and ``sigmoid'`` take (see :mod:`grassmann.transform.diff`). Because it is built
    elementwise from the operand it takes any Euclidean shape, and its own derivative is
    zero. It is public so that :mod:`grassmann.syntax` can round-trip a derivative term
    containing it.
    """
    return _external("ones", _ones_fn, x)
