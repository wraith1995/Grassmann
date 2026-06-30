"""numpy-source renderers for grassmann's own lowering ``Stmt`` ops.

:func:`polyarray.to_numpy_source` emits standalone numpy for a ``Program``, but a Grassmann
program also contains a few grassmann-specific ops (``_ConstOp`` / ``_AxisLenOp`` /
``_ScaleByOp`` / ``_ExternalOp``) that polyarray can't know about. This module supplies their
numpy renderings via the ``op_renderers`` extension point, keeping the dependency direction
grassmann → polyarray (polyarray stays oblivious to grassmann's ops).

Usage::

    import polyarray as pa
    from grassmann.lower.numpy_renderers import op_renderers
    src = pa.to_numpy_source(program, func_name="f", op_renderers=op_renderers())
"""

from __future__ import annotations

import numpy as np

# _ExternalOp.name -> a python-expression template over its (float-coerced) args.
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


def _const_op(op, a):
    val = np.frombuffer(op.data_bytes, dtype=op.dtype).reshape(op.shape)
    return f"np.array({val.tolist()!r}, dtype=np.{op.dtype})"


def _axis_len_op(op, a):
    return f"np.asarray(int(np.asarray({a[0]}).shape[{op.axis}]))"


def _scale_by_op(op, a):
    return f"(np.asarray({a[1]}, dtype=float) * np.asarray({a[0]}, dtype=float))"


def _external_op(op, a):
    xs = [f"np.asarray({e}, dtype=float)" for e in a]
    if op.name in _BINARY:
        return _BINARY[op.name].format(x=xs[0], y=xs[1])
    if op.name in _UNARY:
        return _UNARY[op.name].format(x=xs[0])
    raise NotImplementedError(f"numpy_renderers: no rendering for _ExternalOp {op.name!r}")


def op_renderers() -> dict:
    """The ``{op_class_name: (op, arg_exprs) -> python_expr}`` map for grassmann's Stmt ops,
    to pass as ``polyarray.to_numpy_source(..., op_renderers=op_renderers())``."""
    return {
        "_ConstOp": _const_op,
        "_AxisLenOp": _axis_len_op,
        "_ScaleByOp": _scale_by_op,
        "_ExternalOp": _external_op,
    }
