"""The short linear-algebra compositions worth a name of their own.

Each function here is one or two core operations put together —
:class:`~grassmann.core.term.Inner` applied through an
:class:`~grassmann.core.term.Eval`, a :class:`~grassmann.core.term.Pinv` composed with
its map — spelled out once so callers write the idiom rather than the composition. Like
the rest of the stdlib, none of it is a core node, and all of it is real-valued.
"""

from __future__ import annotations

import grassmann.embed as G
from grassmann.embed.ops import Expr

__all__ = ["dot", "norm_sq", "orthogonal_projection", "least_squares"]


def dot(x: Expr, y: Expr, space: Expr | None = None) -> Expr:
    """Take the inner product ``⟨x, y⟩`` of two vectors, contracting to a ``Scalar``.

    :class:`~grassmann.core.term.Inner` on ``s`` is the metric as a value: a bilinear form
    in ``s* ⊗ s*``, with two covariant slots. Pairing two vectors is therefore just
    applying it to both at once — an :class:`~grassmann.core.term.Eval` at slots
    ``[0, 1]`` — which contracts both slots and leaves a ``Scalar``.

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
    """Give the squared norm ``‖x‖² = ⟨x, x⟩`` of ``x``, a ``Scalar``.

    Squared, so it stays polynomial in the coordinates and the lowering keeps exact
    rational cells; take :func:`grassmann.stdlib.scalar.sqrt` of it for the norm itself.
    """
    return dot(x, x, space=space)


def orthogonal_projection(f: Expr) -> Expr:
    """Build the orthogonal projector ``P = f ∘ f⁺ : W ⊸ W`` onto the image of ``f``.

    Composing ``f`` with its pseudo-inverse (:func:`grassmann.embed.pinv`) sends a vector
    of the codomain to the closest vector that ``f`` can reach. The result is idempotent
    and self-adjoint in the metric, which is what makes it the orthogonal projection onto
    :func:`grassmann.embed.img` of ``f`` rather than some other projection along it.
    """
    return f @ f.pinv


def least_squares(f: Expr, v: Expr) -> Expr:
    """Solve ``f x = v`` in the least-squares sense, returning ``f⁺ v``.

    When ``f`` is not invertible this is the minimum-norm vector among those minimizing
    ``‖f x − v‖``, so it answers for an overdetermined and an underdetermined system
    alike.
    """
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
