"""How grassmann's own statement ops render as standalone numpy source.

A lowered program can be emitted as plain numpy by :func:`polyarray.to_numpy_source`, and a
consumer inlines that source rather than shipping grassmann itself. Most of what a lowered
program contains is a polyarray builtin and renders without help — the transposes, reshapes,
stacks, dimension arithmetic, projections, embeddings and factorizations all live there. A
few ops are grassmann's own, and polyarray cannot know about them; this module supplies their
renderings through the ``op_renderers`` extension point, which keeps the dependency pointing
one way, from grassmann to polyarray::

    import polyarray as pa
    from grassmann.lower.numpy_renderers import op_renderers
    src = pa.to_numpy_source(program, func_name="f", op_renderers=op_renderers())
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .represent import _ExternalOp, _QrColsOp
    from .space_basis import _QrSignConventionOp

# A renderer receives an op together with its arguments already rendered as source, and
# returns the single python expression that computes it.
Renderer = Callable[..., str]

# The python-expression template for each foreign scalar operator, over arguments already
# coerced to float.
_UNARY = {
    "sqrt": "np.sqrt({x})", "abs": "np.abs({x})", "sign": "np.sign({x})",
    "exp": "np.exp({x})", "log": "np.log({x})", "square": "np.square({x})",
    "sin": "np.sin({x})", "cos": "np.cos({x})", "tan": "np.tan({x})",
    "sinh": "np.sinh({x})", "cosh": "np.cosh({x})", "tanh": "np.tanh({x})",
    "asin": "np.arcsin({x})", "acos": "np.arccos({x})", "atan": "np.arctan({x})",
    "sigmoid": "(1.0 / (1.0 + np.exp(-{x})))", "softplus": "np.logaddexp(0.0, {x})",
    "reciprocal": "(1.0 / {x})", "ones": "np.ones_like({x})",
}
_BINARY = {"div": "({x} / {y})", "hadamard": "({x} * {y})"}


def _external_op(op: _ExternalOp, a: list[str]) -> str:
    """Render a foreign scalar operator as the elementwise numpy call it denotes.

    An operator taking no arguments denotes a constant instead, so it is evaluated here at
    emit time and rendered as its literal value.
    """
    if not a:  # an argument-less External is a constant, e.g. a Hodge volume-form basis vector
        import numpy as np
        val = np.asarray(op.fn(), dtype=float)
        return f"np.array({val.tolist()!r}, dtype=float)"
    xs = [f"np.asarray({e}, dtype=float)" for e in a]
    if op.name in _BINARY:
        return _BINARY[op.name].format(x=xs[0], y=xs[1])
    if op.name in _UNARY:
        return _UNARY[op.name].format(x=xs[0])
    raise NotImplementedError(f"numpy_renderers: no rendering for _ExternalOp {op.name!r}")


# -- Subspace realization ops (orthonormal frames) ---------------------------

def _qr_cols(op: _QrColsOp, a: list[str]) -> str:
    """Render a QR's orthonormal columns, signed so that ``R``'s diagonal is positive.

    The sign fix is what makes the frame reproducible: numpy's ``qr`` leaves the column signs
    free, and a zero on the diagonal is read as ``+1``.
    """
    return ("(lambda _A: (lambda _q, _r: _q * np.where(np.sign(np.diag(_r)) == 0, 1.0, "
            f"np.sign(np.diag(_r))))(*np.linalg.qr(_A)))(np.asarray({a[0]}, dtype=float))")


def _qr_sign_convention(op: _QrSignConventionOp, a: list[str]) -> str:
    """Render the positive-diagonal sign fix ``(Q, R) -> (Q·s, s·R)`` with ``s = sign(diag R)``."""
    # The op carries its own emit hook, which is the single source of truth for this rendering.
    # This entry exists for a caller that threads ``op_renderers=`` explicitly; without one,
    # ``to_numpy_source`` finds the hook on the op unaided.
    return op.__numpy_source__(a)


def op_renderers() -> dict[str, Renderer]:
    """Give the map from op class name to renderer for grassmann's own statement ops.

    Pass it as ``polyarray.to_numpy_source(..., op_renderers=op_renderers())``. Only the
    grassmann-specific ops appear, since everything else in a lowered program is a polyarray
    builtin and renders on its own.

    The sign-convention op is a partial exception: it carries its own emit hook, so a program
    containing it emits correctly with no ``op_renderers=`` at all — convenient for an ad-hoc
    dump. Its entry here delegates to that same hook, and an explicit mapping still takes
    precedence over the hook.
    """
    return {
        "_ExternalOp": _external_op,
        # FFS / geometry realization
        "_QrColsOp": _qr_cols,
        "_QrSignConventionOp": _qr_sign_convention,
    }
