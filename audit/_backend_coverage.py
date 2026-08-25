"""Frozen snapshot of the array-IR backend lowering vocabulary (see check_lowering).

grassmann emits array-IR ops (`Stmt.fn` op classes) that are lowered to executable code by
two backends living in the array-IR layer BELOW grassmann:

  * **pyab**  — the builder backend: a builtin op vocabulary, a type-keyed relocated
    array/linalg lowering table, and the op-carried ``fn.__pyab_lower__`` hook.
  * **numpy** — the source backend: a builtin renderer table plus the op-carried
    ``fn.__numpy_source__`` hook (and grassmann's own front-end renderers).

Most ops grassmann emits are DEFINED in that array-IR layer, so their lowerings are not
visible from a standalone grassmann checkout. This module records, as two frozen sets, the
op class names the two backends cover — the vocabulary a co-located array-IR checkout would
otherwise contribute. grassmann's OWN renderers and ``__pyab_lower__`` / ``__numpy_source__``
hooks are still read live from ``src/`` and unioned on top (see check_lowering._tables), so a
renderer grassmann adds is reflected immediately; only the downstream backend vocabulary is
snapshotted here.

Refresh (only when grassmann starts emitting a NEW downstream op, or the backend adds one):
regenerate from a full multi-repo checkout by extracting the pyab builtin/relocated-table
keys, the numpy builtin-renderer keys, and every downstream ``__pyab_lower__`` /
``__numpy_source__`` hook class name.
"""
from __future__ import annotations

# pyab coverage: builtin op vocabulary + relocated array/linalg table keys + downstream
# ``__pyab_lower__`` hook classes.
BACKEND_PYAB_COVERAGE: frozenset[str] = frozenset({
    "AbsOp", "AddOp", "AssertOp", "AxisLenOp", "BlockDiagOp", "BlockRepeatOp", "CallOp",
    "ColStackOp", "CompRankOp", "ComposeViaStdOp", "ConcatOp", "ConstOp", "DetOp",
    "DynBlockRepeatOp", "DynEyeOp", "DynEyeTensorOp", "DynReshapeOp", "DynZerosOp",
    "EinsumOp", "EinsumStmtOp", "EmbedOp", "EyeOp", "FirstColsOp", "GSvdFullOp", "GSvdOp",
    "HStackOp", "IdentityOp", "InvOp", "InvTransposeOp", "KronFreeOp", "KronOp",
    "LastColsOp", "MetricOrthonormalOp", "MoveaxisOp", "MulAxisDimOp",
    "NormalSignFromVerticesOp", "PinvOp", "ProdDimOp", "ProdShapeOp", "Program",
    "ProjectOp", "QrOp", "QrSignFixOp", "RankOp", "ReshapeOp", "ScaleAxisDimOp",
    "ScaleByOp", "ScaleOp", "SignOp", "SinvFullOp", "SolveOp", "SqrtOp", "SqrtSpdOp",
    "SumDimOp", "SumShapeOp", "SvdOp", "SwitchOp", "TensordotOp", "TransposeOp", "WhileOp",
})

# numpy coverage: builtin renderer-table keys + downstream ``__numpy_source__`` hook classes.
BACKEND_NUMPY_COVERAGE: frozenset[str] = frozenset({
    "AbsOp", "AddOp", "AssertOp", "AxisLenOp", "BlockDiagOp", "BlockRepeatOp", "CallOp",
    "ColStackOp", "CompRankOp", "ComposeViaStdOp", "ConcatOp", "ConstOp", "DetOp",
    "DynBlockRepeatOp", "DynEyeOp", "DynEyeTensorOp", "DynReshapeOp", "DynZerosOp",
    "EinsumOp", "EinsumStmtOp", "EmbedOp", "EyeOp", "FirstColsOp", "GSvdFullOp",
    "HStackOp", "IdentityOp", "InvOp", "InvTransposeOp", "KronFreeOp", "KronOp",
    "LastColsOp", "MetricOrthonormalOp", "MoveaxisOp", "MulAxisDimOp", "PinvOp",
    "ProdDimOp", "ProdShapeOp", "Program", "ProjectOp", "QrOp", "QrSignFixOp", "RankOp",
    "ReshapeOp", "ScaleAxisDimOp", "ScaleByOp", "ScaleOp", "SignOp", "SinvFullOp",
    "SolveOp", "SqrtOp", "SqrtSpdOp", "SumDimOp", "SumShapeOp", "SvdOp", "SwitchOp",
    "TensordotOp", "TransposeOp",
})
