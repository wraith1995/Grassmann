# 91 — Application conventions: normative spec + sync plan

> **STATUS: convention DECIDED; Phases 0/1/3 IMPLEMENTED on `rewrite-sop-normal-form`.**
> The fix is made **here** (this branch) and intentionally diverges from the other in-flight
> branches (`ad`, `main`/stable, …) — the merge will need a cross-branch sync (below). Phase 2
> (LLam storage) is assessed and deferred. This doc remains the spec to drive the sync.

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

## Migration plan

### Phase 0 — Specify & conformance test ✅ DONE
- Convention pinned in `docs/_authored/terms/solve-apply.md` (regenerated to `docs/terms/`).
- `tests/test_represent.py::test_application_convention_invariant` asserts `XApply(f,v) ==
  Eval(X(f),v)` for `SolveApply`/`PinvApply` — the acceptance gate.

### Phase 1 — Fix the `SolveApply` orientation ✅ DONE
- `_rep_solve_apply` now desugars to `_rep(Eval(Solve(f),(0,),(v,)))` = `solve(Fᵀ, v)` — inherits
  Eval/Solve's honest per-axis bases. `diff` already differentiated `SolveApply` via `Eval(Solve(f),…)`,
  so this also **aligns the lowering with diff** (no `_dd_solve_apply` change needed).
- Tests updated to the new semantics: `test_represent.py::test_solve_apply_is_eval_of_solve`,
  `test_doc_represent.py::test_solve_apply_solves` (`Aᵀ·got = v`), the `SolveApply` doc example.
- Full suite green (1434).

### Phase 3 — Rewriter dividend ✅ DONE
- `rule_eval_compose_fold`: `Eval(f, Eval(h, x)) → Eval(f∘h, x)` (application associativity). With
  `rule_inverse_compose` + `rule_eval_identity` this cancels applied inverse chains:
  `Eval(Solve(g), Eval(g, x)) → x`, `Eval(Pinv(fi), Eval(fi, x)) → x` (inj). Tests in
  `tests/test_rewrite.py`.

### Phase 2 (deeper) — `LLam` storage orientation — ASSESSED, DEFERRED
Two options:
**Assessment — NOT changed here (deliberately).** The `(cod,dom)` LLam storage is **load-bearing**
for the §7 pullback einsum (`plans/41 §0/§7`), and the `Rep` tag/array mismatch only bites a
**non-default-body / concrete-MATCH path that "is not exercised by callers"** (`represent.py:3656`).
Re-tagging or re-orienting risks the working pullback seam for a path nothing currently hits, so
touching it unprompted is the wrong trade. Options when someone has bandwidth to test the seam:
- **(2a, light, recommended)** keep `(cod,dom)`; fix the `Rep` tag to match the array and add a
  loud guard on the non-exercised path (so it fails fast instead of silently mis-pairing).
- **(2b, full)** make `_rep_llam` emit `(dom,cod)` and transpose at the seam — higher risk.
This is a **separate concern** from the apply-orientation fix (it's basis-tagging honesty, not the
apply convention), so it does not block Phases 0/1/3.

### Phase 3 — Rewriter dividend (closes #11's remainder) ✅ DONE — see above.

## Decisions
1. **Canonical orientation** — ✅ DONE: `Eval`'s contraction-view (input = `Dual` slot).
2. **`SolveApply`** — ✅ DONE on this branch: now `Eval(Solve(f),v) = solve(Fᵀ,v)`.
3. **`LLam`** — ⏳ deferred (2a leaning), separate concern; not changed here.

## Cross-branch sync

The fix now lives on **`rewrite-sop-normal-form`** and **intentionally diverges** from the other
worktrees, which still have the old `SolveApply` orientation:

- `grassmann` (`transform-pullback-covariant-diff`), `grassmann-ad` (`ad`), `grassmann-stable`
  (`main`) — **not yet migrated**.
- **Merge will be tricky** (`_rep_solve_apply` + the `SolveApply` tests changed semantics). At the
  sync: port this doc + the `_rep_solve_apply` desugar + the test updates to the shared base; run
  `test_application_convention_invariant` on each branch (it FAILS pre-migration — the gate);
  apply once; then re-run each branch's suite (esp. anything asserting the old `F⁻¹v`).
- **Diff alignment dividend:** because `diff` already differentiated `SolveApply` via
  `Eval(Solve(f),…)`, the old branches' primal lowering disagreed with their own derivative — this
  fix removes that latent inconsistency, so the migration is a correctness win, not just tidiness.

## Risk / blast radius (as executed here)
- The `SolveApply` semantic change broke exactly **3 tests** (all asserting the old `F⁻¹v`); fixed
  to the new `F⁻ᵀv`. Full suite green (1434). The lowering change is contained (`_rep_solve_apply`
  desugar) and well-covered.
- `LLam` Phase 2 deliberately untouched (load-bearing seam, non-exercised mismatch).
