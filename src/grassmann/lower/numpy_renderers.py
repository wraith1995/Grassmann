"""numpy-source renderers for grassmann's own lowering ``Stmt`` ops.

:func:`polyarray.to_numpy_source` emits standalone numpy for a ``Program``, but a Grassmann
program also contains a few grassmann-specific ops (``_ExternalOp`` / ``_QrColsOp`` …) that
polyarray can't know about. This module supplies their numpy renderings via the
``op_renderers`` extension point, keeping the dependency direction grassmann → polyarray
(polyarray stays oblivious to grassmann's ops).

The generic array / linalg ops (transpose / reshape / stacks / dim-arithmetic / eye / const /
project / embed / kron / inv-transpose / compose / rank / SPD-sqrt / metric-orthonormal / …)
were relocated INTO polyarray as first-class builtins, so their renderings now live in
``polyarray.numpy_source._builtin_renderers`` and are NOT re-supplied here.

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


def _external_op(op, a):
    if not a:  # an argument-less External is a CONSTANT (e.g. a Hodge volume-form basis vector)
        import numpy as np
        val = np.asarray(op.fn(), dtype=float)
        return f"np.array({val.tolist()!r}, dtype=float)"
    xs = [f"np.asarray({e}, dtype=float)" for e in a]
    if op.name in _BINARY:
        return _BINARY[op.name].format(x=xs[0], y=xs[1])
    if op.name in _UNARY:
        return _UNARY[op.name].format(x=xs[0])
    raise NotImplementedError(f"numpy_renderers: no rendering for _ExternalOp {op.name!r}")


# -- FFS / geometry realization ops (orthonormal frames) ---------------------

def _qr_cols(op, a):           # orthonormal columns, positive-R-diagonal sign convention
    return ("(lambda _A: (lambda _q, _r: _q * np.where(np.sign(np.diag(_r)) == 0, 1.0, "
            f"np.sign(np.diag(_r))))(*np.linalg.qr(_A)))(np.asarray({a[0]}, dtype=float))")


def _qr_sign_convention(op, a):  # _QrSignConventionOp(Q, R) -> (Q·s, s·R), s = sign(diag R) (0→1)
    # Delegate to the op's own auto-discovery hook — single source of truth. This entry stays for
    # explicit ``op_renderers=`` threading; ``to_numpy_source`` finds the hook on its own.
    return op.__numpy_source__(a)


def op_renderers() -> dict:
    """The ``{op_class_name: (op, arg_exprs) -> python_expr}`` map for grassmann's remaining
    Stmt ops (the generic array / linalg ops now render as polyarray builtins), to pass as
    ``polyarray.to_numpy_source(..., op_renderers=op_renderers())``.

    ``_QrSignConventionOp`` also carries its own ``__numpy_source__`` hook, which
    ``to_numpy_source`` discovers unaided — so a program containing it emits with no
    ``op_renderers=`` at all (an observability dump, say). Its entry here delegates to that hook,
    and an explicit mapping still takes precedence over it.
    """
    return {
        "_ExternalOp": _external_op,
        # FFS / geometry realization
        "_QrColsOp": _qr_cols,
        "_QrSignConventionOp": _qr_sign_convention,
    }
