"""The standard library: mathematics composed out of the core, never added to it.

Nothing here is a ``Space``, ``Term`` or ``Basis`` node. Every function is written with
the same ``G.*`` builders a user writes with, which is the point: what the core cannot
express, the stdlib cannot smuggle in.

Three layers sit here. :mod:`~grassmann.stdlib.exterior` is the whole exterior algebra —
the antisymmetrizer, the exterior powers as first-class ``Img`` spaces, the wedge and
interior products, the Hodge star and the exterior derivative — derived from tensor
powers and subspace movement alone. :mod:`~grassmann.stdlib.linalg` holds the short
compositions worth a name, such as the inner product, the squared norm, the orthogonal
projector onto an image, and the least-squares solve. :mod:`~grassmann.stdlib.scalar`
supplies the elementwise nonlinearities a matrix-calculus front end needs, each an
``External`` node over a numpy kernel.
"""

from __future__ import annotations

from .exterior import alt, exterior_power, interior, tensor_power, wedge
from .linalg import dot, least_squares, norm_sq, orthogonal_projection
from .scalar import (
    absval,
    acos,
    asin,
    atan,
    cos,
    cosh,
    div,
    exp,
    hadamard,
    log,
    ones,
    reciprocal,
    sigmoid,
    sign,
    sin,
    sinh,
    softplus,
    sqrt,
    square,
    tan,
    tanh,
)

__all__ = [
    # exterior algebra
    "alt",
    "exterior_power",
    "tensor_power",
    "interior",
    "wedge",
    # linear-algebra conveniences
    "dot",
    "norm_sq",
    "orthogonal_projection",
    "least_squares",
    # nonlinear scalar ops
    "sqrt",
    "absval",
    "sign",
    "div",
    "reciprocal",
    "exp",
    "log",
    "square",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "atan",
    "asin",
    "acos",
    "sigmoid",
    "softplus",
    "hadamard",
    "ones",
]
