# polyarray additions for Phase-4 lowering — implementation spec

> 2026-06-16. Precise change-list to make polyarray a target for Grassmann lowering
> (`40`/`41`). All **additive, behind the frozen public API**; existing op behavior
> unchanged; chartlib must stay green. Grounded in the real code (frozen-dataclass ops
> with `__call__`; multi-output via tuple-return scattered in `_run_stmt`;
> `OutSpec(name, shape:tuple[int,...])`; `emit_stmt(fn,in_refs,out_specs,note,bulk)`;
> `SymbolicBudget` presets).

Suggested staging: **Stage A** (ops + budget + assert — easy, no core changes) →
**Stage B** (dynamic shapes — the substantial, riskier change).

---

## Stage A — new Ops + budget (no core-path changes)

### A1. `SvdOp`  (`ir.py`, beside `PinvOp`)
Frozen dataclass; multi-output. Returns `(U, S, V, rank)`; `rank` is the
numerical rank at tolerance `rcond` (default like `np.linalg.matrix_rank`).
```python
@dataclass(frozen=True)
class SvdOp:
    rcond: float | None = None          # threshold for the rank cut
    full_matrices: bool = False
    def __call__(self, A):
        A = np.asarray(A, dtype=float)
        U, S, Vh = np.linalg.svd(A, full_matrices=self.full_matrices)
        tol = (S.max() * max(A.shape) * np.finfo(float).eps) if self.rcond is None \
              else self.rcond * S.max()
        rank = int((S > tol).sum())
        return U, S, Vh, np.asarray(rank)     # 4-tuple → 4 OutSpecs
```
`rank` (the 0-d int) becomes the runtime **`δ_f`** (Stage B `DimAtom`).

### A2. `QrOp`
```python
@dataclass(frozen=True)
class QrOp:
    mode: str = "reduced"
    def __call__(self, A):
        Q, R = np.linalg.qr(np.asarray(A, dtype=float), mode=self.mode)
        return Q, R
```

### A2b. `GSvdOp`  (generalized SVD; metric-aware) — NEW per user decision
Frozen dataclass; multi-output. Computes the GSVD of `A` w.r.t. SPD metrics `M_V`, `M_W`
by whitening with their Cholesky factors then de-whitening — a **named op** so the
metric-general FFS/pinv path (`50 A5`) is explicit, not open-coded in the `Program`.
Returns `(U, UI, V, VI, S, rank)`, factors orthonormal **in the respective metrics**.
```python
@dataclass(frozen=True)
class GSvdOp:
    rcond: float | None = None
    def __call__(self, A, M_V, M_W):
        A  = np.asarray(A, dtype=float)
        Lw = np.linalg.cholesky(np.asarray(M_W, float))   # M_W = Lw Lwᵀ
        Rw = np.linalg.cholesky(np.asarray(M_V, float))   # M_V = Rw Rwᵀ
        Aw = Lw.T @ A @ np.linalg.inv(Rw.T)               # whiten both spaces
        Ut, S, Vth = np.linalg.svd(Aw, full_matrices=True)
        tol = (S.max() * max(Aw.shape) * np.finfo(float).eps) if self.rcond is None \
              else self.rcond * S.max()
        rank = 0 if S.size == 0 else int((S > tol).sum())
        # de-whiten factors back to the metrics: [U|UI]=Lw⁻ᵀŨ, [V|VI]=Rw⁻ᵀṼ
        U_full = np.linalg.solve(Lw.T, Ut)                # ([U|UI])ᵀ M_W ([U|UI]) = I
        V_full = np.linalg.solve(Rw.T, Vth.T)             # ([V|VI])ᵀ M_V ([V|VI]) = I
        U, UI = U_full[:, :rank], U_full[:, rank:]        # image | coker
        V, VI = V_full[:, :rank], V_full[:, rank:]        # coimg | ker
        return U, UI, V, VI, S, np.asarray(rank)          # 6-tuple → 6 OutSpecs
```
**Convention (as implemented).** `A` is the `|W|×|V|` map matrix; `M_V` (`|V|²`) is the
domain metric, `M_W` (`|W|²`) the codomain metric (both SPD). Outputs, FFS-aligned
(`50 §A1`/`§A5`): `U` = image basis in `W` (`Uᵀ M_W U = I`); `UI` = **coker** (the
`M_W`-⊥ complement, `A* UI ≈ 0` for the metric adjoint `A* = M_V⁻¹ Aᵀ M_W`); `V` = coimg
basis in `V` (`Vᵀ M_V V = I`); `VI` = **ker** (`A·VI ≈ 0`). The leading `rank` columns are
image/coimg; the complement is coker/ker. The full metric-orthonormal factors are
`[U|UI]` and `[V|VI]`. **Reconstruction identity** (the trailing `M_V` makes the
coordinates contravariant via the dual pairing):
```
A = [U|UI] · Sfull · [V|VI]ᵀ · M_V    =    U · diag(S[:rank]) · Vᵀ · M_V
```
With `M_V = M_W = I` the Cholesky factors are `I`, so `U=Ũ`, `V=Ṽ`, the identity collapses
to `A = U diag(S) Vᵀ` (so `[V|VI]ᵀ` is `SvdOp`'s `Vh`) and `S`/`rank` match `SvdOp` exactly
— the orientation chosen so `GSvdOp` reduces to `SvdOp`. Kept **distinct** for the paper's
named-boundary discipline (`01 §1.4`). `rank` feeds `DimAtom` exactly like `SvdOp`
(Stage B).

> **Injective / `IsoSpace` case — no new op.** The existing **`QrOp`** (A2) realizes the
> image basis of an asserted-`inj` map (`50 A1`): `Q` is the orthonormal range basis with
> static `rank = |V|`. This recovers the dropped `IsoSpace` functionality (`20`); `inj` ⇒
> `QrOp`, not `SvdOp`.

### A3. `AssertOp`  (passthrough check)
A `Stmt` that validates a predicate on its bound inputs and **passes the first input
through** (so downstream data-depends on it → ordering preserved). Kinds cover the
D-scope checks.
```python
@dataclass(frozen=True)
class AssertOp:
    kind: str            # "shape_eq" | "rank_eq" | "spd" | "square_full_rank"
    msg: str = ""
    def __call__(self, x, *rest):
        x = np.asarray(x)
        if   self.kind == "shape_eq":         _check(x.shape == np.asarray(rest[0]).shape, ...)
        elif self.kind == "rank_eq":          _check(int(rest[0]) == int(rest[1]), ...)   # δ vs asserted
        elif self.kind == "spd":              _check(_is_spd(x), ...)
        elif self.kind == "square_full_rank": _check(x.shape[0]==x.shape[1] and _full_rank(x), ...)
        return x        # passthrough
```
(`_check` raises `AssertionError(self.msg + detail)`.) One output `OutSpec` mirroring
`x`'s shape.

### A4. `SymbolicBudget.force_stmts()`  (the "no symbolic budget" preset)
The opposite of `build_big_symbols`: drive **every modeled op to a `Stmt`** (no
closed-form rational), so Grassmann's symbolic inputs flow through as deferred Stmts.
```python
@classmethod
def force_stmts(cls, **overrides):
    return cls(naive_inverse_max_size=0, inverse_max_degree=0,
               einsum_bag_threshold=1, freeze=True, **overrides)
```
- `naive_inverse_max_size=0`/`inverse_max_degree=0` ⇒ every `det`/`inverse` emits a Stmt.
- `einsum_bag_threshold=1` ⇒ every multi-operand symbolic einsum offloads to a Stmt.
- `freeze=True` ⇒ cap any residual cell to an atom.
(Grassmann mostly emits Stmts directly via `representDown`; this preset covers any
`SymArray`-method path uniformly.)

### A5. wiring
- Export `SvdOp, QrOp, GSvdOp, AssertOp` (and `force_stmts`) from `__init__.py.__all__`.
- No `Program.run` change — `_run_stmt` already calls `fn(*resolved)` and scatters a
  returned tuple over `stmt.out`. Multi-output = pass N `OutSpec`s to `emit_stmt`
  (`SvdOp` → 4, `GSvdOp` → 6).

**Stage-A tests** (`tests/`): SvdOp rank on a known-rank matrix + `U@diag(S)@Vh ≈ A`;
QrOp `Q` orthonormal & `Q@R ≈ A`; **GSvdOp on identity metrics ≈ SvdOp**, and on
non-identity SPD metrics the factors are orthonormal in those metrics; AssertOp
passes/raises per kind; `force_stmts()` makes a `det`/`inverse`/multi-einsum emit a `Stmt`
(assert a statement was appended).

---

## Stage B — dynamic shapes (the substantial change)

**Goal:** an array axis may be a **runtime integer produced by a Stmt** (the SVD rank
`δ_f`), so FFS/projected/image arrays are sized at run time.

### B1. `DimAtom` — a runtime dimension value
A new lightweight value: an integer **produced by a Stmt output** (not pre-declared,
unlike `IntAtom`) and usable as a shape entry.
```python
@dataclass(frozen=True)
class DimAtom:
    name: str            # provenance, e.g. "rank:Alt"
    source: tuple[int,int]   # (stmt_idx, output_index) that produces it
```
Created when an `SvdOp`'s `rank` output is consumed as a dimension; bound to a concrete
`int` at run time (B4).

### B2. `OutSpec.shape` accepts runtime dims
```python
shape: tuple[int | DimAtom, ...] = ()        # was tuple[int, ...]
```
A shape is **dynamic** if any entry is a `DimAtom`.

### B3. `emit_stmt` — force bulk for dynamic shapes
Per-cell atom allocation can't enumerate a `DimAtom`-sized axis. So: **if any
`out_spec.shape` is dynamic ⇒ that output is `bulk=True`** (a whole-tensor handle, no
per-cell atoms); its `SymArray._bulk` carries the (symbolic) shape. This matches
Grassmann's numeric lowering (dynamic-shaped values are never per-cell rational).

### B4. `Program.run` — bind & resolve runtime dims
- When a Stmt produces a value bound to a `DimAtom` output, record `δ := int(value)` in
  `bindings` (a `dim_bindings: dict[DimAtom,int]`).
- Resolving a later dynamic `OutSpec.shape`: substitute each `DimAtom` with its bound
  `int` ⇒ a concrete shape. Concrete arrays flow into numpy ops as usual.
- **Shape-agnostic ops need nothing** — `EinsumStmt`/`Tensordot`/linalg use axis labels /
  the actual array sizes at run time. Only *build-time* shape bookkeeping must tolerate
  `DimAtom`s.

### B5. guard the static path
Everywhere build-time code reads a shape (`np.ndindex`, cell counts, `_cell_size`,
`shape` math): **gate on "shape is dynamic"** and skip per-cell work (route to bulk).
Static-shape programs hit none of this ⇒ behavior byte-identical. Add a `is_dynamic`
helper.

### B6. wiring
- Export `DimAtom`.
- `SymArray.shape` may return a tuple with `DimAtom`s (build time); document it.

**Stage-B tests:** a `Program` where `SvdOp`'s rank sizes a downstream slice; **run on
two inputs of different numerical rank ⇒ two different output shapes**; an `EinsumStmt`
contracting a `δ`-sized axis runs on both; the static-shape suite + chartlib stay green.

---

## Faithfulness / risk

- **Stage A** is purely additive (new frozen ops + a budget preset) — zero risk to
  existing paths; chartlib untouched.
- **Stage B** touches core build-time shape handling — the only real risk. Mitigate with
  the B5 gate (dynamic-only branches), so static programs are byte-identical, and keep
  the rational/per-cell lane out of dynamic shapes (bulk + numeric only).
- Update `polyarray/PUBLIC_API.md` and run the full polyarray + chartlib suites.

## Status
**IMPLEMENTED & GREEN** on branch `grassmann-lowering-additions`: `SvdOp`, `QrOp`,
`AssertOp`, `SymbolicBudget.force_stmts()` (Stage A) + `DimAtom` dynamic shapes (Stage B).
38/38 polyarray tests + chartLib parity green; not pushed (user pushes manually).
**IMPLEMENTED & GREEN (after that pass, per the GSVD decision):** **`GSvdOp`** (A2b) — a
small Stage-A-style additive op; same wiring/test pattern, exported from `__init__.__all__`,
documented in `PUBLIC_API.md`. 4 new tests (identity-metrics ≈ `SvdOp`; metric
orthonormality + reconstruction; known-rank block split incl. ker/coker; 6-output `Program`
scatter); full polyarray suite **42/42 green**, not pushed (user pushes manually). (The
`inj`/IsoSpace case needs *no* new op — it reuses the already-shipped `QrOp`.)
