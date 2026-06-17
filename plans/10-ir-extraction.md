# WS1 — `polyarray`: extract the shared array IR (used by Grassmann *and* chartLib)

> Plan v2 (2026-06-15). Grounded in a direct read of
> `~/projects/fem/chartLib/src/chartlib/_symbolic/`. See `00-landscape.md` §2.
>
> **Name:** the package is importable as **`polyarray`** (the pythonic rendering of
> "PolyArrayPyIR" — short, lowercase, no `_`, drops the redundant `Py`/`IR`). Dist
> name and project dir: `polyarray`.

## What changed from v0 (per direction)

- This is **not** a one-way vendor copy. We **extract** the symbolic array IR out
  of chartLib into a **standalone shared package, `polyarray`**, and **both
  Grassmann and chartLib depend on it.** chartLib is refactored to import the core
  from `polyarray`; its sampler/geometry code stays in chartLib. There is no
  "drift" to track — `polyarray` becomes the single source of truth.
- **Keep all polynomial backends**, including the **Cython C++ backends**
  (`_poly_native_cpp_double`, `_poly_native_cpp_mpf`) and their build glue.
- **No new options.** This is a faithful extraction: same Ops, same `SymArray`
  surface, same provenance, same backend-selection logic. We rename the package and
  move files; we do not add, re-skin, or trim. (Supersedes v0 §4 "deviations" and
  Open Q1.)

## Goal

Stand up `polyarray` as an independent, installable package containing the
symbolic-numeric array IR (`Program`/`Stmt`/`SymArray`/`RationalFunction` + all
poly backends), with **zero dependency on chartLib**; repoint chartLib's
`_symbolic` core at it; and make it the target both chartLib and Grassmann emit
against.

## 1. The extraction boundary (verified against source)

Dependency DAG inside `_symbolic/` (leaf → root), with line counts:

```
  int_atom.py (33) ─────────────┐
  poly_native_value.py (164) ───┤
        └─> poly_native_py.py (561) ─┐
                                     ├─> poly_backend.py (637) ─> rational.py (1041) ─> ir.py (2635)
  poly_backend.py also lazily loads: _poly_native_cpp_double.pyx, _poly_native_cpp_mpf.pyx
```

**MOVE into `polyarray` (the whole IR core + every backend):**
| file | role |
|---|---|
| `ir.py` | `Program`, `Stmt`, `SymArray`, refs, Ops, `SymbolicBudget`, `Provenance`, `SymbolEnv`, `SymInput`, `Program.run` |
| `rational.py` | `RationalFunction`, `bareiss_det`, `cofactor_inverse`, ring helpers |
| `poly_backend.py` | `Poly`/`Ring` protocol + backend selection (sympy / native_py / native_value / cython cpp) |
| `poly_native_py.py` | pure-Python polynomial backend |
| `poly_native_value.py` | value-handle backend support |
| `int_atom.py` | `IntAtom` integer selectors |
| `forward.py` | **IR passes:** `analyze` (cost report) + `partial_eval` (post-build collapse) + `iter_programs`. Imports only `.ir`/`.rational` — see §4a |
| `_poly_native_cpp_double.pyx`, `_poly_native_cpp_mpf.pyx` | C++ polynomial backends |
| `setup_cython.py` (re-pathed) + a `make cython` target | Cython build glue |

**STAYS in chartLib (sampler/geometry — refactored to import the core):**
`interpreter.py` (the `SymbolicInterpreter`), `per_point.py`, `integrate.py`,
and `_symbolic/__init__.py` (which re-exports for chartLib's API).

**Import edges (verified):** `ir.py` imports only `numpy`, `sympy`, `.int_atom`,
`.rational` — clean. `SymbolicBudget` is defined in `ir.py:224` (chartLib's
`interpreter.py` only re-exports it). The only chartLib-specific import surface is
the stock `_symbolic/__init__.py` (the sampler + `TYPE_CHECKING` geometry/topology),
which stays in chartLib. So nothing in the MOVE set references chartLib.

**Cython build (verified):** `setup_cython.py` globs `_poly_native_cpp_*.pyx`,
builds `.so` in place next to the `.pyx`; deps = `numpy`, `Cython`, `sympy`; no
`.pxd`/headers. Moves cleanly; re-path `_PYX_DIR` to the new package and keep a
`make cython` target.

## 2. Package layout & how both consumers use it

`polyarray` is a sibling project under `~/projects/fem/` (like `chartLib`,
`grassmann`, `PyArrayBackend`):

```
~/projects/fem/polyarray/
  pyproject.toml            # deps: numpy, sympy; build: setuptools+Cython, cibuildwheel
  Makefile                  # `make cython` -> build_ext --inplace (local dev)
  setup_cython.py           # re-pathed glob build for the .pyx backends
  src/polyarray/
    __init__.py             # public surface (see §3)
    ir.py rational.py poly_backend.py
    poly_native_py.py poly_native_value.py int_atom.py
    forward.py              # analyze + partial_eval (IR passes)
    _poly_native_cpp_double.pyx  _poly_native_cpp_mpf.pyx
  tests/                    # golden programs + backend matrix (see §7)
```

- **chartLib** depends on `polyarray`; `chartlib/_symbolic/` keeps only
  `interpreter.py`, `per_point.py`, `integrate.py`, and an `__init__.py` that
  imports the core from `polyarray` and re-exports the same names it exports today
  (back-compatible public API for chartLib users).
- **Grassmann** depends on `polyarray`; the Phase-4 lowering emits `Program`s
  against it.

**Packaging — ship wheels.** Build binary wheels with the Cython extensions
precompiled via **cibuildwheel** (manylinux x86_64/aarch64 + macOS arm64/x86_64),
so `pip install polyarray` needs no local C++ toolchain. The pure-Python backends
(sympy / native_py / native_value) remain a working fallback if a wheel is absent;
`make cython` stays for local dev/editable installs.

## 3. The public target API (what chartLib re-exports and Grassmann emits against)

Same surface chartLib exposes today, now homed in `polyarray`:

- **Construction:** `Program`, `program.emit_stmt(fn, in_refs, out_specs, note=…, bulk=…)`,
  `SymInput`, `allocate_input`, `SymbolEnv`, `Provenance`.
- **Values:** `SymArray` (ops: `matmul`, `matvec`, `transpose`, `det`, `inverse`,
  `pinv`, `solve`, `sqrt`, `abs`, `sign`, slicing, `evaluate`), `RationalFunction`.
- **Ops (the Stmt vocabulary):** `DetOp`, `InvOp`, `PinvOp`, `SolveOp`, `SqrtOp`,
  `AbsOp`, `SignOp`, `TensordotOp`, `MoveaxisOp`, `EinsumOp`, `EinsumStmtOp`,
  `SwitchOp`, `CallOp`, `WhileOp`, `IdentityOp`, `IntAtom`.
- **Budget / lanes:** `SymbolicBudget` (+ `.legacy()`, `.build_big_symbols()`).
- **Backends:** the existing `poly_backend` selection logic, unchanged — sympy /
  native_py / native_value / cython cpp.
- **Execution:** `Program.run(values) -> dict[str, ndarray]`,
  `Program.build_runtime_bindings`, `Program.copy`.

Action: `PUBLIC_API.md` enumerating committed names + signatures. Faithful to
chartLib's current exports (no additions).

## 3a. Execution model — *how it runs* (and it does, today)

The IR is **executable now**: running chartLib's IR module tests gives
`36 passed in 0.54s`. The executor is `Program.run`, a **pure-Python tree-walking
interpreter** — **no jax / torch / numba anywhere in the IR** (verified by grep).
`ir.py:2113` `Program.run(values)`:

1. **Bind inputs** (`_bindings_from_values`, `ir.py:2144`): flatten each declared
   input array into a `{atom_name: float}` dict, validate shapes, coerce `IntAtom`s.
2. **Step statements in order** (`_run_stmt`, `ir.py:2204`): for each `Stmt`,
   resolve its input refs (`_resolve_ref`), then call `stmt.fn` — which is either
   a numpy op (`DetOp`→`np.linalg.det`, `EinsumStmtOp`→`np.einsum`, `SolveOp`, …),
   a **sub-`Program`** (recursed via `stmt.fn.run(...)`), or a Python callable —
   and scatter the results back into `bindings`.
3. **Evaluate outputs** (`_eval_symarray`/`_evaluate_cells`, `ir.py:2300`):
   substitute the `bindings` into each named output's `RationalFunction` cells.

So the **three lanes** resolve at run time as: *numeric* cells are already floats;
*rational* cells are closed-form `RationalFunction`s evaluated by substitution;
*deferred* work is exactly the `Stmt`s stepped in (2). The whole thing is data —
build a `Program`, call `.run({...})`, get `{name: ndarray}`.

For Grassmann this means: **the reference target is runnable from day one** — Phase-1
acceptance ("matrix program → numpy") just builds a `Program` and calls `.run`. A
faster jax/codegen backend (FEMOracle's `stmtize`→`execute` idea) is optional, later
Phase-5 work; it is **not** required to execute.

## 3b. Simplification & partial-evaluation passes — yes, they move here

The IR simplification / partial-symbolic-evaluation work you asked about **lives in
the extractable core and moves into `polyarray`** — making `polyarray` the single
home where both chartLib and Grassmann get these improvements. Inventory:

- **In `forward.py` (the IR passes — now in the move-set):**
  - `analyze(program) -> IRReport` — post-build structural cost analysis (stmt
    counts, per-cell monomial mass, provenance histogram).
  - `partial_eval(program, max_cell_size) -> Program` — cost-driven, **exactness-
    preserving** collapse: any output cell over the monomial bound is captured as an
    `IdentityOp` atom and re-evaluated at run time. `max_cell_size=0` ⇒ collapse all
    symbolic outputs to numeric. This *is* the partial-symbolic-evaluation pass.
  - Verified clean: `forward.py` imports only `.ir` and `.rational` (numpy/sympy);
    its "interpreter/geometry" mentions are docstring text, not imports.
- **In `rational.py`:** structural-zero predicates `coeff_zero` / `simple_zero`;
  zero-numerator canonicalization (DCE) in `RationalFunction.__init__`; boundary
  `clean()` / `try_cancel`; sparsity-gated `bareiss_det` / `cofactor_inverse`.
- **In `ir.py`:** `freeze_array` / `freeze_array_bulk` / `unpack` / `IdentityOp`
  (capture-heavy-intermediate-as-atom), `cells_sparsity`, `_cell_size`, and the
  whole `SymbolicBudget` lane-choice machinery (`legacy()`,
  `build_big_symbols()`).
- **Out of scope:** `Program.fingerprint()` (a structural cache key referenced in
  chartLib comments but never implemented). We do **not** add it. Ops stay
  hashable/frozen as they already are; no caching layer.

Because these passes are in scope for the move, the shared package isn't just a
dumb IR — it's the IR **plus** its analysis/partial-eval layer, which is precisely
what Grassmann's lowering will lean on (and what chartLib already uses).

## 4. Faithful extraction — invariants

- **No behavior change.** Byte-for-byte move where possible; edits limited to
  import paths and the `setup_cython.py` `_PYX_DIR`. No new Ops, no provenance
  vocabulary changes, no `SymArray` surface changes, no new backend defaults.
- **chartLib stays green.** chartLib's existing `_symbolic` tests pass unchanged
  against the extracted package (they exercise the public surface we preserve).
- **Backend selection unchanged.** Whatever `poly_backend.py` does today to choose
  sympy/native_py/native_value/cpp is preserved verbatim.

## 5. Extraction procedure (extract-and-refactor)

1. **Snapshot:** record chartLib's git commit/hash + date in `polyarray/VENDORED.md`
   (provenance of the one-time extraction).
2. **Scaffold** the `polyarray` package (`pyproject.toml`, `src/polyarray/`,
   `Makefile`, `tests/`).
3. **Move** the MOVE-set files (incl. `forward.py`) into `src/polyarray/`; fix
   intra-package imports (already relative: `.int_atom`, `.rational`, `.ir`, …).
4. **Move + re-path** `setup_cython.py` (`_PYX_DIR` → new package dir); add the
   `make cython` target and the cibuildwheel config. Build and confirm the two
   `.so` load.
5. **Compile-check isolation:** `import polyarray` with chartLib *off* the path.
   Add a test asserting `chartlib` is not imported as a side effect.
6. **Refactor chartLib:** add `polyarray` as a dependency; in
   `chartlib/_symbolic/__init__.py`, `interpreter.py`, `per_point.py`, `integrate.py`
   (and any module that used `forward.analyze`/`partial_eval`), replace
   `from .ir import …` / `from .rational import …` / `from .forward import …` with
   `from polyarray import …`. Keep `chartlib/_symbolic`'s public re-exports
   identical. Delete the moved files (incl. `forward.py`) from chartLib.
7. **Run chartLib's full test suite** — must stay green (this is the acceptance gate
   for the refactor).
8. **Author `polyarray` golden tests** (§7) and `PUBLIC_API.md`.
9. **Point Grassmann** at `polyarray` (dependency only; lowering comes in Phase 4).

## 6. Backends — all of them, no new options

Preserve the full set and the existing selection logic:
- **sympy** ring (always available).
- **native_py** (pure-Python; no build).
- **native_value** (value-handle support).
- **cython cpp** (`_poly_native_cpp_double`, `_poly_native_cpp_mpf`) — shipped
  precompiled in the **wheels** (cibuildwheel); `make cython` for local/editable
  installs; lazily imported by `poly_backend` with the existing "extension not
  built" fallback to the pure-Python backends preserved.

The **rational lane is in scope** (it exercises the poly backends; required for the
backend test matrix). No change to defaults — whatever chartLib selects today is
what `polyarray` selects.

## 7. Acceptance / verification

- **Import isolation:** `import polyarray` succeeds with chartLib not importable;
  a test asserts no `chartlib` module loaded afterward.
- **Backend matrix:** the golden programs run on each available backend
  (sympy, native_py, native_value, and cpp from the wheel / `make cython`) with
  matching results.
- **Golden programs** (hand-built `Program`s via `Program.run`):
  - numeric lane: matmul + `det` + `solve` → matches numpy.
  - rational lane: symbolic 2×2 inverse → `RationalFunction` entries match by hand.
  - deferred/imperative lane: over-budget `det` emits `Stmt(DetOp)` and evaluates.
  - `EinsumStmtOp` / `Tensordot` contraction round-trips.
  - `SwitchOp` selects a branch by `IntAtom`.
- **chartLib regression:** chartLib's existing test suite passes against the
  extracted package (the real proof the extraction is faithful).
- **Lane determinism:** same program shape → same statement schema across runs.

## 8. Resolved & open

**Resolved:**
- Import/dist name = **`polyarray`**.
- **Ship wheels** (cibuildwheel, cpp precompiled) with a pure-Python fallback.
- **All backends** kept, **no new options**; `forward.py` (analyze + partial_eval)
  is **in** the package.
- **Migration sequencing:** land `polyarray` (package + golden tests + wheels)
  **first** as a self-contained change; then refactor chartLib to depend on it as a
  **second** change. Smaller blast radius, and `polyarray` is verifiable on its own
  before chartLib moves.
- **`Program.fingerprint()`: out of scope.** Not implementing it. (Ops stay
  hashable/frozen as they already are, but no fingerprint/caching layer is added.)

**Resolved (cont.):**
- **Grassmann emits against `polyarray` directly** — no thin wrapper layer. The
  Phase-4 lowering calls `Program` / `emit_stmt` / the Op classes / `SymArray`
  directly.

## 9. Status

**Stage 1 — DONE (2026-06-15).** `polyarray` stands alone at
`~/projects/fem/polyarray/`; `17 passed` (import isolation + all three lanes +
partial_eval round-trip); **all 5 backends green** (sympy, native_py, native_value,
native_cpp_double, native_cpp_mpf); Cython C++ `.so`s build via `make cython`;
cibuildwheel configured; `forward.py` (analyze + partial_eval) included; **no
`Program.fingerprint`**. The 9 source/`.pyx` files were copied byte-for-byte (only
`setup_cython.py` repathed); two infra files added for wheels (`setup.py`,
`.gitignore`). Public API → `polyarray/PUBLIC_API.md`; provenance → `VENDORED.md`.
chartLib **untouched** (HEAD `11d426a`; its IR tests still `36 passed`).

**Carried-forward notes (decide later, not blocking):**
- Backend selection env vars remain `CHARTLIB_POLY_BACKEND` / `CHARTLIB_POLY_COEFF`
  (preserved verbatim — renaming to `POLYARRAY_*` would be a behavior change).
- Valid `CHARTLIB_POLY_BACKEND` = `sympy | flint | native_py | native_cpp`;
  `native_value` is the value-handle layer (not a standalone backend); `COEFF` =
  `double | mpf | quad`.
- `flint` is an optional extra (`.[flint]`), sympy fallback when absent.
- Build/test env is Python 3.14; `requires-python = ">=3.13"`.

**Stage 2 — chartLib refactor: DONE & GREEN (2026-06-15).** On branch
`polyarray-extraction` (chartLib, off `main@11d426a`, commit `e9b3dd0`; default branch
untouched). `polyarray` added as an editable path dep (`[tool.uv.sources]`), Cython
`.so`s built via `make -C ../polyarray cython` (chartLib's `Makefile cython` target now
delegates there; chartLib's own `setup_cython.py` deleted). ~90 import sites across
`_symbolic/` + 12 src modules + ~30 test files repointed to `polyarray.*` (submodule
paths preserved so private helpers resolve); `chartlib._symbolic` public `__all__`
unchanged. The 9 moved files deleted from chartLib. **No extraction bugs** — every
name chartLib needed (incl. private helpers) was present in `polyarray`; `polyarray`
unmodified. Tests: fast lane `1045 passed`; full `3418 passed, 873 skipped` (with
`--timeout=120` per chartLib CLAUDE.md; one heavy parity combo only trips the default
30s cap under xdist contention, passes at higher timeout — not a regression).
**Merged:** fast-forwarded into chartLib `main` (now at `e9b3dd0`); local only —
`main` is ahead of `origin/main` by 1, not yet pushed. `polyarray` pushed to private
repo `github.com/wraith1995/polyarray` (`main` @ `972a2b5`).
