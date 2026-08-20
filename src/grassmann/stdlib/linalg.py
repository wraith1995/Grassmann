"""Small linear-algebra conveniences: derived ops built on the embed surface, not core IR.

Thin wrappers over the :mod:`grassmann.embed` surface for the most common derived
linear-algebra idioms. Like :mod:`grassmann.stdlib.exterior`, nothing here is a core
node — each function composes existing core ops (:class:`~grassmann.core.term.Inner` /
:class:`~grassmann.core.term.Eval`, :class:`~grassmann.core.term.Pinv`,
:class:`~grassmann.core.term.Compose`). Real-valued.
"""

from __future__ import annotations

import grassmann.embed as G
from grassmann.embed.ops import Expr

__all__ = ["dot", "norm_sq", "orthogonal_projection", "least_squares"]


def dot(x: Expr, y: Expr, space: Expr | None = None) -> Expr:
    """Take the inner product ``⟨x, y⟩`` of two vectors, contracting to a ``Scalar``.

    :class:`~grassmann.core.term.Inner` on ``s`` is the metric bilinear form — an element
    of ``s* ⊗ s*`` with two covariant slots — so it is applied to the two vectors directly
    (an :class:`~grassmann.core.term.Eval` at slots ``[0, 1]``), fully contracting to a
    ``Scalar``.

    Parameters
    ----------
    x, y
        The two operands; both must live in the metric space ``s``.
    space
        The metric space ``s``. Defaults to ``x``'s declared space; pass it explicitly
        when ``x``'s space cannot be read off the node (e.g. a derived term, not a
        :class:`~grassmann.core.term.Var`).
    """
    s = space if space is not None else _space_of(x)
    return G.inner(s)(x, y)


def norm_sq(x: Expr, space: Expr | None = None) -> Expr:
    """``‖x‖² = ⟨x, x⟩`` — a ``Scalar``."""
    return dot(x, x, space=space)


def orthogonal_projection(f: Expr) -> Expr:
    """Build the orthogonal projector ``P_{img f} = f ∘ f† : W ⊸ W`` onto ``img f``.

    ``f†`` is the Moore–Penrose pseudo-inverse (:func:`grassmann.embed.pinv`); ``f ∘ f†``
    is the orthogonal projection onto the image :func:`grassmann.embed.img` of ``f``
    (idempotent, self-adjoint in the metric).
    """
    return f @ f.pinv


def least_squares(f: Expr, v: Expr) -> Expr:
    """``f† v`` — the minimum-norm least-squares solution of ``f x = v``."""
    return f.pinv(v)


def _space_of(x: Expr) -> Expr:
    from grassmann.core import space as _s

    node = x.node if isinstance(x, Expr) else x
    s = getattr(node, "space", None)
    if s is not None and _s.is_space(s):
        return Expr(s)
    raise ValueError(
        "linalg: could not read the operand's space; pass space=... explicitly"
    )
