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


def _scale_op(op, a):
    return f"({op.factor!r} * np.asarray({a[0]}, dtype=float))"


def _add_op(op, a):
    return "(" + " + ".join(f"np.asarray({e}, dtype=float)" for e in a) + ")"


def _concat_op(op, a):
    return ("np.concatenate(["
            + ", ".join(f"np.asarray({e}, dtype=float).reshape(-1)" for e in a) + "])")


def _transpose_op(op, a):
    return f"np.asarray({a[0]}, dtype=float).T"


def _axis_len_op(op, a):
    return f"np.asarray(int(np.asarray({a[0]}).shape[{op.axis}]))"


def _scale_by_op(op, a):
    return f"(np.asarray({a[1]}, dtype=float) * np.asarray({a[0]}, dtype=float))"


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


# -- FFS / geometry realization ops (subspace frames, project/embed) ---------

def _passthrough(op, a):       # _IdentityMatrixOp / _IdentityArrOp — value passthrough
    return f"np.asarray({a[0]}, dtype=float)"


def _qr_cols(op, a):           # orthonormal columns, positive-R-diagonal sign convention
    return ("(lambda _A: (lambda _q, _r: _q * np.where(np.sign(np.diag(_r)) == 0, 1.0, "
            f"np.sign(np.diag(_r))))(*np.linalg.qr(_A)))(np.asarray({a[0]}, dtype=float))")


def _project_op(op, a):        # Pᵀ @ v  (ambient coords -> sub-basis coords)
    return f"(np.asarray({a[0]}, dtype=float).T @ np.asarray({a[1]}, dtype=float).reshape(-1))"


def _embed_op(op, a):          # P @ vsub  (sub-coords -> ambient), reshaped
    base = f"(np.asarray({a[0]}, dtype=float) @ np.asarray({a[1]}, dtype=float))"
    return base + (f".reshape({tuple(op.shape)})" if getattr(op, "shape", ()) else "")


def _dyn_eye(op, a):           # runtime identity sized by a ref array's axis (default column)
    return f"np.eye(int(np.asarray({a[0]}).shape[{getattr(op, 'axis', 1)}]))"


def _prod_shape(op, a):        # static · ∏ refs[i].shape[axes[i]] — flattened tensor-product dim
    inner = ", ".join(f"int(np.asarray({a[i]}).shape[{ax}])" for i, ax in enumerate(op.axes))
    return f"np.asarray(int({getattr(op, 'static', 1)} * int(np.prod([{inner}] or [1]))))"


def _sum_shape(op, a):         # static + Σ refs[i].shape[axes[i]] — flattened direct-sum dim
    inner = " + ".join(f"int(np.asarray({a[i]}).shape[{ax}])" for i, ax in enumerate(op.axes))
    return f"np.asarray(int({getattr(op, 'static', 0)} + ({inner or '0'})))"


def _dyn_eye_tensor(op, a):    # np.eye(∏dᵢ).reshape(∏dᵢ, *dims) — matrix/tensor vmap seed identity
    dims = ", ".join(f"int(np.asarray({a[i]}).shape[{ax}])" for i, ax in enumerate(op.axes))
    return (f"(lambda _d: np.eye(int(np.prod(_d))).reshape(int(np.prod(_d)), *_d))([{dims}])")


def _dyn_zeros(op, a):         # np.zeros((d₀, d₁, …)) — a symbolic ℝⁿ zero sized off input shapes
    dims = ", ".join(f"int(np.asarray({a[i]}).shape[{ax}])" for i, ax in enumerate(op.axes))
    return f"np.zeros(({dims},))"


def _first_cols(op, a):        # _FirstColsOp(A, rank) -> A[:, :rank]  (rank orthonormal cols)
    return f"np.asarray({a[0]}, dtype=float)[:, :int({a[1]})]"


def _last_cols(op, a):         # _LastColsOp(A, rank) -> A[:, rank:]  (the complementary cols)
    return f"np.asarray({a[0]}, dtype=float)[:, int({a[1]}):]"


def _inv_transpose(op, a):     # _InvTransposeOp(A) -> inv(A).T
    return f"np.linalg.inv(np.asarray({a[0]}, dtype=float)).T"


# -- runtime dimension / rank arithmetic (0-d int producers sizing dynamic FFS axes) ---------

def _prod_dim(op, a):          # ∏ of operands' axis lengths
    return f"np.asarray(int(np.prod([np.asarray(_m).shape[{op.axis}] for _m in [{', '.join(a)}]])))"


def _sum_dim(op, a):           # Σ of operands' axis lengths
    return f"np.asarray(int(sum(np.asarray(_m).shape[{op.axis}] for _m in [{', '.join(a)}])))"


def _scale_axis_dim(op, a):    # n · (axis length), static n
    return f"np.asarray(int({op.n} * np.asarray({a[0]}).shape[{op.axis}]))"


def _mul_axis_dim(op, a):      # int(n) · (axis length), runtime n
    return f"np.asarray(int(int({a[0]}) * np.asarray({a[1]}).shape[{op.axis}]))"


def _rank_op(op, a):           # numerical rank of A
    tol = "None" if getattr(op, "tol", None) is None else repr(op.tol)
    return f"np.asarray(int(np.linalg.matrix_rank(np.asarray({a[0]}, dtype=float), tol={tol})))"


def _comp_rank(op, a):         # ambient − rank (the FFS complement dimension)
    return f"np.asarray(int({op.ambient} - int({a[0]})))"


def _eye_op(op, a):            # static identity np.eye(n)
    return f"np.eye({op.n})"


# -- FFS assembly ops (stacks / kron / metric orthonormalization) ----------------------------

def _kron(op, a):              # chained Kronecker product
    return f"__import__('functools').reduce(np.kron, [{', '.join(a)}])"


def _kron_free(op, a):         # Kron of two maps with trailing free (basis/seed) axes (block ⊗)
    nf, ng = op.nf_free, op.ng_free
    return (
        f"(lambda _F, _G: (_F.reshape(_F.shape[0], 1, _F.shape[1], 1, *_F.shape[2:], *([1]*{ng})) "
        f"* _G.reshape(1, _G.shape[0], 1, _G.shape[1], *([1]*{nf}), *_G.shape[2:])).reshape("
        f"_F.shape[0]*_G.shape[0], _F.shape[1]*_G.shape[1], *_F.shape[2:], *_G.shape[2:]))"
        f"(np.asarray({a[0]}, dtype=float), np.asarray({a[1]}, dtype=float))")


def _hstack(op, a):            # _HStackOp(*mats) -> np.hstack
    return f"np.hstack([np.asarray(_m, dtype=float) for _m in [{', '.join(a)}]])"


def _col_stack(op, a):         # _ColStackOp(*cols) -> stack flattened cols as columns
    return (f"np.stack([np.asarray(_c, dtype=float).reshape(-1) for _c in "
            f"[{', '.join(a)}]], axis=1)")


def _compose_via_std(op, a):   # _ComposeViaStdOp(R_to, R_from) -> solve(R_to, R_from)
    return f"np.linalg.solve(np.asarray({a[0]}, dtype=float), np.asarray({a[1]}, dtype=float))"


def _metric_orthonormal(op, a):  # _MetricOrthonormalOp(A, G) -> A @ inv(chol(Aᵀ G A))ᵀ
    return (f"(lambda _a, _g: _a @ np.linalg.inv(np.linalg.cholesky(_a.T @ _g @ _a)).T)"
            f"(np.asarray({a[0]}, dtype=float), np.asarray({a[1]}, dtype=float))")


def _sqrt_spd(op, a):          # _SqrtSpdOp(G) -> the SPD square root via eigh
    return (f"(lambda _w, _V: (_V * np.sqrt(_w)) @ _V.T)(*np.linalg.eigh("
            f"0.5 * (np.asarray({a[0]}, dtype=float) + np.asarray({a[0]}, dtype=float).T)))")


def _qr_sign_convention(op, a):  # _QrSignConventionOp(Q, R) -> (Q·s, s·R), s = sign(diag R) (0→1)
    return (f"(lambda _Q, _R: (lambda _s: (_Q * _s, _s[:, None] * _R))"
            f"(np.where(np.sign(np.diag(_R)) == 0, 1.0, np.sign(np.diag(_R)))))"
            f"(np.asarray({a[0]}, dtype=float), np.asarray({a[1]}, dtype=float))")


def _identity_mat(op, a):      # _IdentityMatOp(n) — a fixed n×n identity (no args)
    return f"np.eye({op.n})"


def _reshape_op(op, a):
    return f"np.asarray({a[0]}, dtype=float).reshape({tuple(op.shape)})"


def op_renderers() -> dict:
    """The ``{op_class_name: (op, arg_exprs) -> python_expr}`` map for grassmann's Stmt ops,
    to pass as ``polyarray.to_numpy_source(..., op_renderers=op_renderers())``."""
    return {
        "_ConstOp": _const_op,
        "_AxisLenOp": _axis_len_op,
        "_ScaleByOp": _scale_by_op,
        "_ScaleOp": _scale_op,
        "_AddOp": _add_op,
        "_ConcatOp": _concat_op,
        "_TransposeOp": _transpose_op,
        "_ExternalOp": _external_op,
        # FFS / geometry realization
        "_IdentityMatrixOp": _passthrough,
        "_IdentityArrOp": _passthrough,
        "_QrColsOp": _qr_cols,
        "_FirstColsOp": _first_cols,
        "_LastColsOp": _last_cols,
        "_InvTransposeOp": _inv_transpose,
        "_ProdDimOp": _prod_dim,
        "_SumDimOp": _sum_dim,
        "_ScaleAxisDimOp": _scale_axis_dim,
        "_MulAxisDimOp": _mul_axis_dim,
        "_RankOp": _rank_op,
        "_CompRankOp": _comp_rank,
        "_EyeOp": _eye_op,
        "_KronOp": _kron,
        "_KronFreeOp": _kron_free,
        "_HStackOp": _hstack,
        "_ColStackOp": _col_stack,
        "_ComposeViaStdOp": _compose_via_std,
        "_MetricOrthonormalOp": _metric_orthonormal,
        "_SqrtSpdOp": _sqrt_spd,
        "_QrSignConventionOp": _qr_sign_convention,
        "_ProjectOp": _project_op,
        "_EmbedOp": _embed_op,
        "_DynEyeOp": _dyn_eye,
        "_ProdShapeOp": _prod_shape,
        "_SumShapeOp": _sum_shape,
        "_DynEyeTensorOp": _dyn_eye_tensor,
        "_DynZerosOp": _dyn_zeros,
        "_IdentityMatOp": _identity_mat,
        "_ReshapeOp": _reshape_op,
    }
