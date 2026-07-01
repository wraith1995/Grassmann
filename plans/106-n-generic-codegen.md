# Plan 106 — dimension-generic (n-generic) codegen

## Goal
The web tool's "Generate Python" currently substitutes every symbolic dim `n → 3` and unrolls
(`np.eye(3)`, `A[0,0]…A[2,2]`). It should emit **dimension-generic** numpy: `n` taken from the
input array shapes at runtime (`A.shape[0]`, `np.eye(n)`), working for any `n`.

## Root cause (traced)
- `api_code` does `dims = {d: 3}` + `instantiate` → hardcodes 3.
- Keeping `n` symbolic fails in `represent`: `_shape_of_space(OPlusRep(DimVar('n')))` → `_static_dim`
  throws `DynamicDim`. Even a bare `x : R^n` fails.
- The **vmap** that materializes a derivative map (`_rep_llam_vmap`) also needs a static `n` to build
  `np.eye(n)` and the binder shape.

## Mechanism to reuse
The FFS runtime-rank lane already threads a **runtime dimension** as a `pa.DimAtom`:
- `pa.DimAtom.from_input(name, input_name, axis)` — a dim read from `array.shape[axis]` at run time.
- `_shape_of_space_dyn(space, ctx)` returns `DimAtom`s for FFS axes; `_make_dynamic_ffs_input` mints
  input-sourced atoms + sibling `shape_eq`/`rank_eq` asserts.
- polyarray renders `("in", name, axis)`-sourced atoms as runtime shapes and defers input binding.

**Plan: make a symbolic `DimVar('n')` resolve to a canonical input-sourced `DimAtom`, so the whole
lowering flows a runtime dim instead of a static int** — the same lane the FFS δ uses.

## Steps (each guarded by the full suite, 2346 green)
1. **Foundation — vars.** `_Ctx.dimvar_atoms: {dimname: DimAtom}`. In `_rep_var`, a var whose space
   carries a `DimVar` axis mints the canonical atom from ITS shape the first time (source =
   `("in", var, axis)`), reuses it after (+ a `shape_eq` sibling assert). `_shape_of_space_dyn`
   returns the canonical atom for `DimVar` axes. Target: bare `x`, `x+x` lower + render `np.eye(n)`-free.
2. **Linear algebra.** `A*x`, `x'*x`, `A*B` — einsum/compose/inner over runtime axes (the metric
   `Inner` identity becomes `np.eye(n)` at runtime, not a literal).
3. **The derivative / vmap.** `_rep_llam_vmap` builds the binder eye from a runtime `DimAtom`
   (`np.eye(n)`), per the δᵢ-loop idea — materialize the map by sweeping `n` runtime directions.
   (η-reduced forms may still hit an `inner`-map llam, so this is needed either way.)
4. **Wire api_code.** Emit from the symbolic (non-instantiated) node when it lowers; keep the
   `dims={n:3}` fallback if a node still can't. Update `_prettify`.

## Status
- [ ] Step 1 — foundation (vars)
- [ ] Step 2 — linear algebra
- [ ] Step 3 — vmap/derivative
- [ ] Step 4 — api_code wiring
- Branch: `feat/mcalc-frontend`. Full suite must stay green at each step.
