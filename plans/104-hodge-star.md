# Plan 104 — Hodge star ⋆ (status: 2 of 3 engine gaps fixed; construction verified)

Goal: `hodge_star(ω, k) : Λᵏ(V) → Λⁿ⁻ᵏ(V)`, verified, exposed in the ASCII surface.

## VERIFIED CONSTRUCTION (math solved — `pinv(M) ∘ flat`, metric induced by the embedding)
```python
def pair(s, a, b): return G.inner(s).at([0, 1])(a, b)         # ⟨a,b⟩_s (explicit space)
Lk, Lnk, Ln = exterior_power(V,k), exterior_power(V,n-k), exterior_power(V,n)
vol  = G.project(G.tensor(*[e(i) for i in range(n)]), Ln)      # antisym-projection = volume form
vol_unit = G.scale(G.reciprocal(G.sqrt(pair(Ln, vol, vol))), vol)   # ⟨vol,vol⟩ = 1
M = G.llam(Lnk, lambda g: G.llam(Lk, lambda a: pair(Ln, wedge(a, g, k, n-k), vol_unit)))  # CURRIED
star = G.pinv(M)(G.flat(Lk)(beta))                            # ⋆β = M⁻¹(flat β)
```
Notes: `e(i)` = External constants (run fine). `M` MUST be the **curried** nested `llam`
(`Λⁿ⁻ᵏ → Dual(Λᵏ)`); a 2-arg `llam` lifts to `(Λⁿ⁻ᵏ⊗Λᵏ)⊸ℝ` and mis-shapes `pinv`.

## The three engine gaps this surfaced
1. **FFS α-equivalence** — FIXED (`infer/unify.py`). Independently-built `Img(alt(V,k))` now unify.
2. **Scalar ⊗-unit** — FIXED (`infer/unify.py`). `OProd(X, Scalar) ≅ X`, so `A⊸ℝ` (≡ `A*⊗Scalar`)
   unifies with the bare dual `A*` from `flat`.
   → With (1)+(2) the construction PASSES INFERENCE. Full suite green (2320 passed).
3. **Static dimension of exterior powers in lowering** — OPEN. `dim_of(Img(alt(V,k)))` is a
   runtime SVD atom (`DimAtomRef`), so `pinv`'s lowering raises
   `RepresentError: …has a runtime/symbolic dimension; dynamic-shape lowering …not yet wired`
   (`lower/represent.py:5262`, via `rank_of` → `DimAtomRef` in `infer/dims.py:155`).
   The dimension IS statically `C(n,k)`, but:
   - `exterior_power(V,k,rank=C(n,k))` gives a static dim but uses an **abstract `Var`** projector
     that `wedge` can't recover `V` from AND that has no numeric kernel to run.
   - `exterior_power(V,k)` (real `alt`) runs but has the dynamic dim.

### Fix options for gap 3 (pick one — its own focused task)
- (a) **Static rank for the real antisymmetrizer**: teach `rank_of` (or `exterior_power`) that
  `Img(alt(V,k))` has rank `C(dim(V), k)` — e.g. `alt` tags its node so `rank_of` returns the
  binomial instead of `DimAtomRef`. Cleanest; keeps the real (runnable) `alt`.
- (b) **Decouple** dimension-carrier (rank-asserted) from numeric projector (real `alt`) in
  `exterior_power` — small engine design change.
- (c) **Wire dynamic-shape `pinv`/`solve` lowering** for runtime-dimensioned FFS spaces
  (`represent.py`) — deepest.

Recommend (a). Once it lands, the construction above is drop-in → add R³/R⁴ ground-truth tests
(`⋆e₁=e₂∧e₃`, `⋆⋆=(-1)^{k(n-k)}id`, `α∧⋆β=⟨α,β⟩vol`) → expose `wedge`/`alt`/`hodge`/`Lambda(V,k)`
in the ASCII surface → merge.

## Status
Gaps 1+2 fixed and committed (general improvements, suite green). Construction verified to pass
inference. Blocked only on gap 3 (static exterior-power dimension). NOT merged as a runnable ⋆ yet.
