"""The standard library: derived operations built on the embedding.

These are coordinate-free linear-algebra operations built entirely on the
``G.*`` embedding surface, not as core IR nodes. Exterior algebra in particular
is a test of the core's expressiveness: ``wedge`` and ``alt`` are composed
functions here, never primitive nodes.

Public surface
--------------
- :func:`alt` — the ``k``-linear antisymmetrizer ``Alt : V⊗ᵏ ⊸ V⊗ᵏ``.
- :func:`exterior_power` — the ``k``-th exterior power ``Λᵏ(V) = Img(alt(V, k))``.
- :func:`wedge` — the derived wedge product ``∧ : Λᵏ × Λˡ → Λ^{k+l}``.
- :func:`tensor_power` — the ``k``-th tensor power ``V⊗ᵏ``.
- linear-algebra conveniences: :func:`dot`, :func:`norm_sq`,
  :func:`orthogonal_projection`, :func:`least_squares`.
- scalar nonlinearities: :func:`sqrt`, :func:`absval`, :func:`sign`,
  :func:`div`, :func:`reciprocal`.
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
