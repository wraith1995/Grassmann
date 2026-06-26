"""IW5 — the Grassmann stdlib (derived operations on top of the embedding).

A thin library of *derived* coordinate-free LA operations built **entirely** on the
``grassmann.embed`` (``G.*``) surface — explicitly **not** core IR. Exterior algebra is an
expressiveness test for the core (`50 §0`), so ``wedge`` / ``alt`` live here as composed
functions, never as primitive nodes.

Public surface
--------------
- :func:`alt` — the ``k``-linear antisymmetrizer ``Alt : V⊗ᵏ ⊸ V⊗ᵏ`` (`26 §2`, F2).
- :func:`exterior_power` — ``Λᵏ(V) = Img(alt(V, k))`` (`26 §2`).
- :func:`wedge` — the derived ``∧ : Λᵏ × Λˡ → Λ^{k+l}`` (`26 §2`, F5).
- :func:`tensor_power` — ``V⊗ᵏ`` (helper).
- linear-algebra conveniences: :func:`dot`, :func:`norm_sq`,
  :func:`orthogonal_projection`, :func:`least_squares`.
- basic nonlinear scalar ops (the old ``BaseNonLinear``): :func:`sqrt`, :func:`absval`,
  :func:`sign`, :func:`div`, :func:`reciprocal`.
"""

from __future__ import annotations

from .exterior import alt, exterior_power, tensor_power, wedge
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
    log,
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
]
