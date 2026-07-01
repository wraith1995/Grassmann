# Plan 105 — derivative operators in the expression language (exterior d, covariant ∇, D)

Status: **DESIGN for review.** Feasible today — the pieces exist (`diff`, `wedge`, `alt`,
`hodge`, `exterior_power`, the immersion-gauge covariant machinery, and the exterior-algebra
codegen just wired). Question: which operators + how fields depend on a point.

## The reframe
The surface is currently ALGEBRAIC: an expression is a fixed value in its variables. Derivative
operators need FIELDS — expressions that vary over a **base point** `x ∈ V` (or a manifold via a
parameterization). So we add: a distinguished point, and operators that differentiate w.r.t. it.

Feasibility shown: for a 0-form `f(x) = ½⟨x,x⟩`, `diff(f, x) = x♭` — its differential 1-form.
For a k-form, `d = alt(diff)` (antisymmetrize the new covariant index into the form).

## Operators to add (all inline, composable)
- **`D(f, x)` / `diff(f, x)`** — the total derivative / Jacobian of `f` w.r.t. the point `x`
  (directly the engine's `diff`). The primitive everything else builds on. SMALL.
- **`d(ω)`** — exterior derivative on a **k-form field** `ω : V → Λᵏ(V*)`:
  `dω = antisymmetrize(∂ω/∂x) ∈ Λᵏ⁺¹(V*)` (= `project(alt(flatten(diff(ω, x))), Λᵏ⁺¹(V*))`).
  Nilpotent (`d∘d = 0`) — a good correctness test.
- **`grad(f) = d(f)`** (0-form → 1-form) · **`curl(V) = ⋆ d V`** (3-D) · **`div(V) = ⋆ d ⋆ V`** —
  derived from `d` + `hodge`, so they come almost for free once `d` lands.
- **`covariant(v)` / `∇(v)`** — covariant derivative of a tangent field along the manifold; reuse
  the immersion-gauge connection (`covariant_diff`), exposed inline instead of only via `wrt`.

## Key design decisions (☐)
1. **Forms vs vectors.** Differential forms live in `Λᵏ(V*)` (exterior powers of the DUAL);
   today `Lambda(V,k)` is `Λᵏ(V)`. Add **`Omega(V,k)` = Λᵏ(V*)** (k-forms) so `d` has the right
   target. ☐ Spelling: `Omega(V,k)` / `Λ(V*,k)` / `Form(V,k)`?
2. **How a field depends on the point.** Two ways, both supported:
   - (a) **opaque field** `omega(x)` — a black-box function of the point; `d`/`∇` differentiate it
     via the ④ required-derivative mechanism (already downloadable).
   - (b) **explicit field** — an expression written in the point variable `x` (e.g. `½*x'*x`),
     differentiated symbolically.
   ☐ Support both from the start, or opaque-only first?
3. **The point.** A distinguished variable (`x : R^n`, the manifold coordinate) or a
   parameterization `phi` (curved manifold → use `∇`, `d` pulls back). ☐ Flat `R^n` first, add
   parameterization support after?

## Why it's tractable now
- `d`, `grad/div/curl` reduce to `diff` + `alt`/`wedge`/`hodge` + `project` — all present.
- `∇` reuses `covariant_diff` (validated to 1e-10) — just needs an inline surface form.
- Download: `d`/`∇` of opaque fields emit the ④ inputs + the exterior codegen that now renders.
- Great built-in tests: `d∘d = 0`, `⋆d⋆ = div`, Stokes-flavor identities, `∇g = 0`.

## Scope options
- **A — minimal:** inline `D(f, x)` / `diff(f, x)` (total derivative in expressions). Small,
  immediately useful (Jacobians/gradients inline), no new spaces. Ship first.
- **B — exterior calculus:** `Omega(V,k)` + `d` + `grad/div/curl`. Medium; the headline ask.
- **C — covariant:** inline `∇` on parameterizations. Medium; reuses existing machinery.

Recommend **A → B → C**, each verified (`d∘d=0`, etc.) and downloadable.

## Decisions (answered) + implemented
- `D(f, x)` (total/covariant derivative) + `d(g, x)` (exterior); top-level `wrt` is now OPTIONAL
  (leave it empty → the primal IS the inline-derivative expression). Forms in `Λᵏ(V*)` via
  **`Omega(V,k)`**. Fields are opaque functions evaluated at a point var. NO grad/div/curl
  primitives — recover them in examples.

### Done ✅
- **`D(f, x)`** — total derivative, any expression: `D(½|x|²,x)=x` (gradient), `D(A*x,x)=A`
  (Jacobian). Builds/runs/**downloads**.
- **`Omega(V,k)` = Λᵏ(V*)** space/type; grade-tracked.
- **`d(f, x)` for a 0-form** — the differential 1-form (via `diff` + inject into `Λ¹(V*)`).
- **Optional `wrt`** end to end (render + download with inline `D`/`d`).
- Grammar "Derivatives" section + a `D(f,x): gradient` example.

### Remaining — `d` for k ≥ 1 (the general exterior derivative)
`dω = alt(∂ω/∂x)` over `(V*)⊗ᵏ⁺¹`. Two real subtleties found:
1. **Currying** — `diff(diff(ω,x),x)` is a NESTED `LLam` (`λdx.λdy.…`); needs uncurrying
   (`lift`) into a single `(V*)⊗ᵏ⁺¹` value before `alt`. For an OPAQUE form field the first
   `diff` is a single clean map, so `d(opaque_form)` should work directly; `d∘d` is where the
   nesting bites.
2. **Constant injection** into `Λᵏ(V*)` hits the same subspace/canonicalization friction seen
   with the Hodge basis vectors.
Once this lands, the textbook **`grad = d`(0-form), `curl = ⋆d`, `div = ⋆d⋆`** examples drop in
(they need `d` on 1-/2-forms + `hodge` on `Λᵏ(V*)`), plus `d∘d=0` as the correctness test.

Recommended next: a focused pass wiring `d` for k≥1 (uncurry + the `Λᵏ(V*)` injection), then the
grad/div/curl example table + `d∘d=0`.
