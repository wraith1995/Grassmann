# 91 — Application conventions: normative spec + sync plan

> **STATUS: convention DECIDED, implementation DEFERRED to a cross-branch sync.**
> Other branches (`ad`, `main`/stable, …) are in flight; changing application semantics now
> would diverge them. This document **records the agreed convention** so every branch migrates
> to it together at the sync. Do **not** implement Phases 1–3 on a single branch in isolation.

## Convention (DECIDED — `Eval`'s contraction-view is canonical)

1. A map `f : A ⊸ B` (space `~A⊗B = Dual(A)⊗B`) is **stored `(|A|,|B|) = (dom,cod)`**.
2. `Eval` is **the** application primitive and is canonical: **a map's input is its `Dual`
   factor**; `Eval(f, slot, x)` contracts that slot. (`Eval(f,(0,),x)` reads as `Fᵀ·x` only
   because storage is `(dom,cod)` and that's numpy-matmul-relative — the contraction-view is the
   real definition. `Eval` does **not** change.)
3. **The one invariant every map-application op must satisfy:**

   > `XApply(f, v)  ≡  Eval(X(f), (0,), (v,))`  — a bundled apply equals applying the map via `Eval`.

   - `PinvApply` already conforms.
   - **`SolveApply` is the lone violator** today: it computes `solve(F,v)=F⁻¹·v` (cod-orientation)
     instead of `Eval(Solve(f),v)=solve(Fᵀ,v)=F⁻ᵀ·v`. At sync it is fixed to conform. Callers
     wanting the old `F⁻¹·v` write `SolveApply(Dagger(f), v)` / `Eval(Dagger(Solve(f)), v)`.
4. **`LLam` storage**: currently codomain-major `(cod,dom)` (FFS/pullback-seam convention,
   `plans/41 §0/§7`) while its synth space is `(dom,cod)`; its `Rep` even tags a `(cod,dom)`
   array with a `(dom,cod)` basis. Sync decision pending (Phase 2a vs 2b below).

All of the above is numerically verified (sub-agents, 16/16 asserts; `plans/90 §findings`).

## The problem ("this is silly")

A map `f : A ⊸ B` (space `~A⊗B = Dual(A)⊗B`) is **stored `(|A|,|B|) = (dom,cod)`**. Three ways
to "apply" a map currently disagree about which axis a vector flows into:

| op | what it computes | contracts | consistent with `Eval`? |
|----|------------------|-----------|--------------------------|
| `Eval(f,(0,),(x,))` | `Fᵀ·x` | the **dom** axis (slot 0 = the `Dual` factor) | — (this IS the canonical primitive) |
| `PinvApply(f,v)` | `(Fᵀ)⁺·v` = `Eval(Pinv(f),v)` | dom axis (Eval orientation) | ✅ yes |
| `SolveApply(f,v)` | `F⁻¹·v` = `solve(F,v)` | the **cod** axis (opposite!) | ❌ **no** |

So `SolveApply` is the **one outlier**: it inverts the *opposite* orientation from the one
`Eval`/`PinvApply` apply. Separately, **`LLam` lowers codomain-major `(cod,dom)`** while its
synthesized space is `(dom,cod)` — a documented FFS/pullback-seam convention (`plans/41 §0/§7`),
but its `Rep` tags a `(cod,dom)` array with a `(dom,cod)` basis (a latent axis-pairing hazard on
the non-default-body / concrete-MATCH path, `represent.py:3656`).

All facts verified by sub-agents (16/16 numeric asserts; cites in `plans/90 §findings`).

## The canonical convention (the one rule)

`Eval`'s contraction-view is canonical and principled: **a map's input is its `Dual` factor**;
`Eval(f, slot, x)` contracts that slot. The `fᵀ` look is only numpy-matmul-relative. Everything
that "applies a map" must satisfy the single invariant:

> **`XApply(f, v)  ≡  Eval(X(f), (0,), (v,))`**  — the bundled apply = apply the corresponding
> MAP via `Eval`.

`PinvApply` already obeys it; `SolveApply` does not. We do **not** touch `Eval` itself.

## Migration plan (execute at the cross-branch sync, not before)

### Phase 0 — Specify & add a conformance test (no behavior change)
- Pin the rule in `docs/terms/{eval,solve-apply,pinv-apply,llam}.md`.
- Add a **convention-conformance** property test: for every map-application op and a battery of
  random maps, assert `Apply(f,v) == Eval(map(f), v)` numerically. It **fails for `SolveApply`
  today** — that failing test is the acceptance gate for Phase 1.

### Phase 1 — Fix the `SolveApply` orientation (the outlier)
- `_rep_solve_apply` (`represent.py`): compute in the `Eval` orientation — `solve(Fᵀ, v)`
  (equivalently, lower `SolveApply(f,v)` by desugaring to `Eval(Solve(f),(0,),(v,))`).
- Update `_dd_solve_apply` (`transform/diff.py:363`) if it bakes the old orientation.
- Audit `f.solve(v)` callers (`embed/ops.py:280`) and every test using `solve_apply`/`SolveApply`;
  any that encoded the old `F⁻¹v` switch to the new `F⁻ᵀv` (or, to keep `F⁻¹v`, write
  `SolveApply(Dagger(f),v)` / `Eval(Dagger(Solve(f)),v)`).
- Green: conformance test + full suite.

### Phase 2 (optional, deeper) — `LLam` storage orientation
Two options:
- **(2a, light, recommended)** Keep `(cod,dom)` at the FFS/pullback seam (it's load-bearing for
  §7), but fix the `Rep` self-inconsistency (tag the array with the basis order that matches it)
  and add the missing axis-pairing guard/assert for the non-default-body / concrete-MATCH path.
- **(2b, full)** Make `_rep_llam` emit canonical `(dom,cod)` and transpose at the seam instead.
  Higher risk — touches the §7 pullback einsum and its consumers.

### Phase 3 — Rewriter dividend (closes #11's remainder)
Once the convention is uniform, the applied-form inverse identities and the
`Eval(f, Eval(h, x)) → Eval(Compose(f,h), x)` fold become unambiguous → wire them into the
rewriter (the pieces held back under the item-4 "where does Compose go" caution).

## Decisions
1. **Canonical orientation** — ✅ DECIDED: `Eval`'s contraction-view (input = `Dual` slot).
   (Rejected: flipping to natural `F@x` — too invasive.)
2. **`SolveApply`** — ✅ DECIDED in principle: redefine to satisfy the invariant
   (`solve(F,v) → solve(Fᵀ,v)`). Execute at sync.
3. **`LLam`** — ⏳ OPEN at sync: Phase 2a (light: fix `Rep` tag + axis-pairing guard, keep the
   `(cod,dom)` seam) vs 2b (full re-orientation). Lean 2a.

## Cross-branch sync

This convention must land **uniformly** across the active worktrees/branches so they don't
diverge on application semantics:

- `grassmann` (`transform-pullback-covariant-diff`), `grassmann-rw` (`rewrite-sop-normal-form`,
  where this doc lives), `grassmann-ad` (`ad`), `grassmann-stable` (`main`).
- At the sync: cherry-pick/port this doc to the shared base, run Phase 0 (the conformance test)
  on each branch to confirm they all exhibit the same `SolveApply` violation, then apply Phase 1
  once on the base and merge. Phase 2/3 follow.
- Until then: **rewriter rules added on `rewrite-sop-normal-form` rely only on `Eval`'s
  contraction-view and the verified map-level identities** — none depend on the (future)
  `SolveApply` fix, so this branch stays correct under the current semantics.

## Risk / blast radius
- Phase 1 is a **semantic change to `SolveApply`** (its result transposes). Few construction
  sites (`embed/ops.py`, `diff.py`) but **several tests** reference it — expect to update the ones
  that asserted the old `F⁻¹v` value. Lower-level lowering change, well-covered by the suite.
- Phase 2 touches the FFS/pullback seam; 2a is contained, 2b is not.
