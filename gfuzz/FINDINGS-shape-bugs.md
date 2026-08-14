# Findings: two shape-divergence bugs in grassmann → polyarray lowering

Investigation only — no source edited. All paths are repo-relative to `grassmann/`
unless noted. Every claim below was checked by running code.

## TL;DR

The suspected root cause is **confirmed**: a grassmann `Space` does **not** have a
single canonical numeric array shape at the lowering boundary. `_shape_of_space`
(the convention a leaf `Var` uses) and the reducing nodes `_rep_oprod` /
`_rep_oplus` compute an operand's shape by **different, mutually inconsistent
conventions** whenever a tensor operand is *multi-axis* — which happens exactly
when an `OProd` factor is itself an `OProd` (a nested tensor) or carries a unit
`Scalar` (`𝟙`) factor.

- **Bug C** (OProd / unit factor): `_rep_oprod` produces **more axes** than
  `_shape_of_space` assigns the same space, so an `OProdV` node and a sibling
  `Var` of the identical space realize to different ranks → `Sum` shape assert.
- **Bug B** (OPlus / direct sum): `_rep_oplus` sizes the concatenation from
  **axis 0 only** of each summand, undercounting a multi-axis summand → the
  declared `ConcatOp` output shape disagrees with the (correct) flattened data.

They are **two distinct defects in two distinct functions**, sharing one root
shape: *code that assumes a compound-space operand realizes to a single axis (or a
specific axis count) when it can be multi-axis*. `drop_unit_factors` is essentially
**never applied at the numeric boundary** (its only mention in `lower/represent.py`
is a comment at line 4140), so the two conventions are never reconciled.

---

## Bug C — OProd with a unit `Scalar` tensor factor

### Minimal standalone reproducer (no gfuzz)

```python
import numpy as np, grassmann as G
from grassmann.core import space as S, term as T
R3 = S.base('R3', 3)
z1 = T.var('z1', S.oprod(R3, S.Scalar()))                       # (R3 ⊗ 𝟙)
v2 = T.var('v2', S.Scalar())                                    # 𝟙
z2 = T.var('z2', S.oprod(S.oprod(R3, S.Scalar()), S.Scalar()))  # (R3 ⊗ 𝟙) ⊗ 𝟙
F  = T.sum_(T.OProdV((z1, v2)), z2)     # both operands live in (R3 ⊗ 𝟙) ⊗ 𝟙
prog = G.compile(F)
feed = {i.name: np.ones(tuple(int(s) for s in i.shape)) for i in prog.inputs}
prog.run(feed)      # AssertionError: assert:sum-shapes[shape_eq] (3,1,1) != (3,1)
```

Verified directly that the two `Sum` operands realize to different ranks for the
**same** space `Y = (R3 ⊗ 𝟙) ⊗ 𝟙`:

| node | realized shape |
|---|---|
| `T.OProdV((z1, v2))` (terminal `OutSpec`) | `(3, 1, 1)` |
| `T.var('z2', Y)` (declared input) | `(3, 1)` |

### Root cause

Two shape conventions for `OProd` that disagree on **axis count** when a factor is
itself compound:

**(a) `_shape_of_space` — one axis per TOP-LEVEL factor** (`lower/represent.py:7147-7159`):

```python
case S.OProd(spaces):
    return tuple(_static_dim(s) for s in spaces)   # 7150-7151
```

Each factor is collapsed to a single integer `_static_dim(s)` (its *total*
flattened dim). The dynamic mirror `_shape_of_space_dyn` does the same
(`7186-7187`, `_dyn_dim` per factor), which is what a leaf `Var` uses
(`_rep_var` → `_shape_of_space_dyn`, ~line 845). So:

- `_shape_of_space((R3⊗𝟙) ⊗ 𝟙)` = `(_static_dim(R3⊗𝟙)=3, _static_dim(𝟙)=1)` = `(3,1)`.

This convention is **not a fixpoint**: `_shape_of_space(R3 ⊗ 𝟙)` = `(3,1)` (the
unit `𝟙` factor becomes a size-1 axis), whereas concatenating the per-factor
shapes `_shape_of_space(R3) ++ _shape_of_space(𝟙)` = `(3,) ++ ()` = `(3,)`.

**(b) `_rep_oprod` — concatenate each operand's FULL realized axes**
(`lower/represent.py:1288-1362`, loop at `1323-1337`):

```python
for v, r in zip(vectors, reps):
    ashape = _shape_of(r.array)
    n_space = len(ashape) - len(r.free_labels)
    ...
    space_dims.extend(ashape[:n_space])            # 1333: keep ALL of the operand's space axes
    if n_space == 0 and isinstance(_synth(v), S.Scalar):
        canonical_dims.append(1)                   # 1335: a 0-axis Scalar operand gets a singleton
    else:
        canonical_dims.extend(ashape[:n_space])
...
out_shape = tuple(space_dims) + free_dims           # 1341
# then a ReshapeOp to canonical_dims if a Scalar factor lacked its axis (1346-1351)
```

For `OProdV((z1, v2))`:
- `z1` (a `Var` of `R3⊗𝟙`) is realized at `_shape_of_space(R3⊗𝟙)` = `(3,1)` → 2 axes → contributes `[3,1]`.
- `v2` (a `Var` of `𝟙`) is realized at `()` → 0 axes → contributes canonical `[1]`.
- ⇒ `canonical_dims = [3,1,1]`, final shape `(3,1,1)`.

The extra axis is `z1`'s **inner** `𝟙` axis. `_shape_of_space(Y)` collapsed the
whole factor `R3⊗𝟙` into one `_static_dim = 3` axis, but `_rep_oprod` splices in
`z1`'s realized `(3,1)` verbatim. The two conventions agree **only** when every
factor is single-axis (a `base`/`dual`/`scalar` leaf); a nested `OProd` factor —
most easily produced by a unit `Scalar` factor — breaks the agreement.

The crash surfaces in `_rep_sum` (`lower/represent.py:1191-1226`, `_assert_shape_eq`
at 1222) because the two `Sum` operands now have different ranks. `Sum` is only the
*messenger*; the defect is the shape `_rep_oprod` assigned.

---

## Bug B — OPlus (direct sum) of a multi-axis summand

### Minimal standalone reproducer (no gfuzz)

```python
import numpy as np, grassmann as G
from grassmann.core import space as S, term as T
R2d = S.dual(S.base('R2', 2))      # (R2)*
R3  = S.base('R3', 3)
A = T.var('A', S.oprod(R2d, R3))   # (R2)* ⊗ R3   -> realizes (2,3)
B = T.var('B', R2d)                # (R2)*        -> realizes (2,)
F = T.OPlusV((A, B))               # direct sum
prog = G.compile(F)
feed = {i.name: np.ones(tuple(int(s) for s in i.shape)) for i in prog.inputs}
prog.run(feed)     # ValueError: bulk Stmt output 'oplus': expected shape (4,); got (8,)
```

This is the reduced core of the fuzzer's trial#330
(`../.venv/bin/python -m gfuzz --seed 0 --only 330 --no-shrink`), whose crashing
subterm is `((v2+v3)+tanh(v4)) ⊕ (…·v5+tanh(v6))` with `v2..v4 : (ℝ²)*⊗ℝ³` and
`v5,v6 : (ℝ²)*` — an `OPlus` of a `(2,3)` summand and a `(2,)` summand.

### Root cause

`_rep_oplus` (`lower/represent.py:1253-1285`), the length computation at `1267-1273`:

```python
_arrs = [r.array for r in reps]
_dims = [(_shape_of(a)[0] if _shape_of(a) else 1) for a in _arrs]   # 1270: AXIS 0 ONLY
_sum  = _combined_dim(ctx, _dims, "sum", _arrs, 0, label="oplus")   # 1271: ref_axis=0
[out] = ctx.prog.emit_stmt(
    pa.ConcatOp(), _arrs, [pa.OutSpec("oplus", (_sum,))], note="oplus:concat")  # 1272-1273
```

`ConcatOp` **flattens every operand** (`reshape(-1)`) before concatenating —
confirmed in `polyarray/src/polyarray/batch.py:196-202` (comment 184-186). So for
`A=(2,3)` and `B=(2,)` the actual concat length is `6 + 2 = 8`, which correctly
equals `_shape_of_space(OPlus(...)) = (8,)` (`_shape_of_space` OPlus arm,
`7152-7153`: `sum(_static_dim(s))`, with `_static_dim((R2)*⊗R3) = 6`).

But `_dims` reads only `_shape_of(a)[0]` — the **first axis**: `2` for `A`, `2` for
`B` → `_sum = 4`. The declared `OutSpec("oplus", (4,))` then disagrees with the
(mathematically correct) 8-element concat, and the bulk shape check raises
`expected (4,); got (8,)`.

Confirmed numerically: `_static_dim(A) = 6`, `_shape_of(A)[0] = 2`, canonical
`_shape_of_space(OPlus) = (8,)`. The **data would be correct**; only the declared
length is wrong. The dynamic path is wrong the same way —
`_combined_dim(..., ref_axis=0)` would emit a `SumDimOp` reading axis 0 only.

---

## Are B and C the same defect?

**No — two distinct defects, one shared root shape.**

- Distinct functions and mechanisms:
  - **C** is in `_rep_oprod`: it produces an array whose **rank** differs from the
    space's canonical `_shape_of_space` rank (extra axes from a compound/unit
    factor). The *data* is mis-shaped.
  - **B** is in `_rep_oplus`: the `ConcatOp` *data* is correct (matches the
    canonical `(8,)`), but the **declared length** is undercounted because it reads
    axis 0 instead of the flattened total.
- Shared root: both assume an `OProd`-typed operand of a compound node is
  single-axis. Both are instances of the same missing invariant — *a space has one
  canonical numeric shape and every realizer must match it* — and both are
  triggered by the same structural feature (a multi-axis `OProd`, i.e. nested
  `OProd` or a unit `Scalar` factor).

The unit `Scalar` (`𝟙`, `drop_unit_factors`) hypothesis is **confirmed for C** (the
extra axis is literally an undropped `𝟙` axis) and is **the same family for B**
(a multi-axis tensor summand), though B's specific trigger is a genuine `⊗` product
`(R2)*⊗R3`, not a unit factor. `drop_unit_factors` is never invoked at the numeric
boundary, so nothing collapses `R3⊗𝟙 → R3` before shaping.

---

## Fix proposals (NOT implemented)

All fixes belong in **grassmann** (`lower/represent.py`); polyarray's `ConcatOp`
flatten semantics and shape checks are correct and should stay.

### Bug B — `_rep_oplus` (`lower/represent.py:1270-1271`)

Size each summand's contribution by its **flattened total** (the size `ConcatOp`
actually contributes), not axis 0. Statically that is the product of all axes:

```python
# instead of _shape_of(a)[0]:
_dims = [int(np.prod(_shape_of(a))) if _shape_of(a) else 1 for a in _arrs]
```

For the dynamic/`DimAtom` case, combine over **all** axes of each operand with a
"prod" reduction rather than passing `ref_axis=0` to `_combined_dim`. Cleaner
equivalent: size each summand from its space via `_static_dim`/`_dyn_dim(_synth(v))`
(which already give the flattened total and match `_shape_of_space`'s OPlus arm), so
`_rep_oplus` and `_shape_of_space` are sized by the *same* function. Result: declared
shape `(8,)`, matching both the concat data and the canonical space shape. Low risk —
produced data does not change.

### Bug C — `_rep_oprod` (`lower/represent.py:1323-1351`) vs `_shape_of_space` (`7150-7151`)

Reconcile the two conventions to **one canonical shape**. Two options:

1. **Make the realizer match `_shape_of_space` (recommended, minimal blast
   radius).** After building the outer-product array, `ReshapeOp` it to exactly
   `_shape_of_space(space)` (`space = _synth(e)`), i.e. collapse each top-level
   factor's realized axes down to the single `_static_dim` axis `_shape_of_space`
   assigns it. Guarantees an `OProdV` and a sibling `Var` of the same space are
   shape-compatible. Preserve `free_labels` (`LLam`-bound axes) after the space axes.

2. **Canonicalize the space via a recursive `drop_unit_factors` at the numeric
   boundary**, applied consistently in `_shape_of_space`, `_shape_of_space_dyn`,
   `_rep_oprod`, and `_rep_var`, so `R3⊗𝟙 → R3` (and nested products flatten) before
   any shape is computed. More principled, higher blast radius: `drop_unit_factors`
   today is non-recursive/top-level only, and lanes such as Λᵏ `OProd((𝟙, Img(alt∘J)))`
   deliberately rely on the current un-dropped shape, so it must be introduced behind
   the canonical-shape invariant below.

Option 1 fixes the crash locally; option 2 addresses the underlying "one canonical
shape" gap. They compose (do 1 now, 2 as the systemic follow-up).

---

## Invariant worth asserting (catches this whole class)

> **A grassmann `Space` has exactly one canonical numeric array shape,
> `_shape_of_space(space)` (dynamic mirror `_shape_of_space_dyn`), and every
> `_rep_*` node that realizes a value of that space MUST produce an array whose
> shape equals it.**

Add a debug/assert hook at the `Rep`-construction boundary (or in `_rep`/`_rep_node`,
`lower/represent.py:578` / `708`): for any node whose `_synth` is a *statically*
shaped space, assert `_shape_of(rep.array)[:n_space] == _shape_of_space(_synth(e))`
(accounting for trailing `free_labels` axes). This would have caught **both** bugs at
the producing node (`_rep_oprod` for C, `_rep_oplus` for B) instead of downstream in
`_rep_sum`/`ConcatOp`.

Companion fuzz oracle: for every generated well-typed term `t`,
`_shape_of(compile(t) terminal) == _shape_of_space(synth_space(t))` (up to free axes).
gfuzz already generates the offending terms.

---

## Evidence log (commands run under `grassmann/`, venv `../.venv/bin/python`)

- Bug C repro → `AssertionError assert:sum-shapes[shape_eq] (3,1,1) != (3,1)`.
- Bug B via gfuzz `--seed 0 --only 330` and the minimal standalone → both
  `ValueError: bulk Stmt output 'oplus': expected shape (4,); got (8,)`.
- `_shape_of_space((R3⊗𝟙)⊗𝟙) = (3,1)`; `T.OProdV((z1,v2))` terminal `OutSpec = (3,1,1)`;
  `Var z2` declared input `= (3,1)`.
- `_shape_of_space(OPlus((R2)*⊗R3, (R2)*)) = (8,)`; `_static_dim((R2)*⊗R3) = 6`;
  `_shape_of((R2)*⊗R3)[0] = 2`.
- `ConcatOp` flattens each operand (`polyarray/src/polyarray/batch.py:196-202`,
  comment 184-186).
- `drop_unit_factors` defined `core/space.py:617`; only textual mention in
  `lower/represent.py` is the comment at line 4140 — not applied at the numeric
  boundary.

---

## Dynamic-shape mode — n-generic lowering gaps (found later, NOT fixed)

The `gfuzz --dynamic` mode builds terms over symbolic `ℝⁿ` bases (`OPlusRep(DimVar, Scalar)`) so
the compiled program is dimension-generic. Most nodes cleanly bail on symbolic dims
(`RepresentError: … has a runtime/symbolic dimension` — an expected skip). But a few nodes crash
*uncleanly* (an `AssertionError`/`ValueError`, not a clean refusal) on terms that are
**bidirectionally well-typed** (both `synth_space` and `infer.check` accept them) — i.e. the type
checker says the term is fine but the n-generic realizer mis-sizes a tensor axis. These are the
dynamic analog of bugs B/C (the same tensor-axis-realization family), reported for triage:

- **`Kron` applied over symbolic dims** — `kron(A, B)(u ⊗ v)` raises
  `AssertionError: assert:eval-dim@0[rank_eq] 4 != 2` (a fused product axis `2·2=4` contracted
  against a single `2`). Bidirectionally typed. (Minimal core: a `kron` of two symbolic-dim maps
  applied to an `OProdV`.)
- **Double `permute` of a symbolic tensor product** — `permute(permute(remetric(v, g) ⊗ w, [1,0]),
  [1,0])` (which should be the identity) raises
  `ValueError: Einstein sum subscript a does not contain the correct number of indices`. The
  permute→einsum for a multi-axis operand with a symbolic axis emits a malformed spec.

Both live in `lower/represent.py`'s n-generic path (the `_rep_oprod`/`Kron`/`Permute` einsum
construction with `DimAtom` axes). They are lower-priority than B/C (dynamic lowering is
explicitly only partially wired), and are left for the maintainer — the fuzzer reports, it does not
fix. Reproduce: `python -m gfuzz --dynamic --seed 1` (shrinks to the terms above).

---

## `diff` of `det`/`logdet` produces an ill-typed Jacobian (found by mining the skips)

Discovered by auditing the *skip* population — the lesson from the shape hook is that a real bug
can hide behind an "acceptable bail". The `ad_vs_fd` oracle was classifying this as an
`InferError` skip; it is a genuine `diff` soundness bug.

**Symptom.** For a well-typed `F` containing `det(M)` / `logdet(M)` where `M` is an x-dependent
*square endomorphism that is not asserted `bij`* (e.g. `M = M0{bij} + bump(x)` — a `Sum`, which
`synth` accepts for `det`/`logdet` since a determinant is defined for any square matrix),
`diff(F, x)` returns a Jacobian whose body **does not type-check**:
`InferError: Solve requires the map to be asserted 'bij'`.

**Minimal reproducer.**
```python
V = S.base('V', 3); x = T.var('x', V)
Mb = T.var('M', S.oprod(S.dual(V), V), props=bij)
Mx = T.sum_(Mb, T.scale(0.05, T.oprod_v(T.Eval(T.flat(V),(0,),(x,)), x)))  # x-dependent, NOT bij
synth_space(T.LogDet(Mx))         # OK — logdet of a square endomorphism type-checks
synth_space(diff(T.LogDet(Mx), x).body)   # InferError: Solve requires bij
```

**Root cause.** `transform/diff.py::_dd_logdet_kernel` (~line 1061) builds the derivative
`tr(f⁻¹·df)` as `Trace(Compose(Solve(f), (0,), (df,)), 1, 0)`. `Solve(f)` requires `f` provably
`bij`, but `f` (the det/logdet argument) need not be. `det`'s rule (`_dd_det`) uses the same
kernel; `trace`'s derivative doesn't need the inverse, so it is unaffected.

**Fix direction (a maintainer call — touches the invertibility assumption).** Jacobi's formula
`d det(X) = det(X)·tr(X⁻¹ dX)` presupposes `X` invertible; det/logdet are only differentiable at an
invertible point. So `diff` should construct the `Solve` under a `bij` assumption on `f` (assert
`bij` on the operand for the derivative), rather than requiring the *original* term to carry it.
Alternatively `synth` could require `logdet`'s argument to be invertible. Reported, not fixed.

The `ad_vs_fd` oracle now reports "diff produced an ill-typed Jacobian" as a **bug** (any inference
error when lowering `J·h`, given `F` itself lowered) — so this class can no longer hide behind the
`InferError` bail.
