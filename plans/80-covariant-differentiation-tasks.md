# Covariant differentiation — task backlog

The first cut (`src/grassmann/transform/diff.py`) gives the total derivative `X⊸space(e)`
for closed terms, with a Levi-Civita connection correction for a **rank-1 section in a
tangent space** (`Remetric(sec, g)` with a marked metric). This doc lists the work to make
covariant differentiation general. Ordered roughly by value × independence.

## Current scope (done)
- Total/Fréchet derivative: product/chain/quotient over Eval/Compose/OProd/Scale/Inner +
  nonlinear scalar External; applied inverses (`Solve`/`Pinv` through the vector), `Dagger`.
- **Covariant connection, BASIS-FREE**, for sections `Remetric(sec, g)` (marked `g`/`Dg`,
  `X = M`) via the tensor-product/Leibniz rule — `∇sec = d(sec) + Γ(h)·sec` applying the
  connection map `Γ(h):M⊸M` per slot (`+Γ` contravariant, `−Γ` covariant):
  - **vector** `v:M`, **covector** `ω:M*`, **(1,1) tensor** `T:M⊸M` (`∇T = ΓT − TΓ`).
  - The connection map `Γ(h) = Pinv(g) ∘ ½(T1 + T2 − Permute(T2,(1,0)))` uses the inverse
    metric as the **bare map `Pinv(g)`** (`g` decodes as the flat map `M⊸M*`), so nothing
    realizes a metric basis — the derivative is coordinate-free end to end.
  - Verified: vs analytic Christoffel (tangent/cotangent/(1,1)), `∇I = 0`, and the
    metric-pairing invariant `⟨∇ω,v⟩+⟨ω,∇v⟩ = d⟨ω,v⟩`.
- Flat/constant metric ⇒ not a covariant section ⇒ the plain total derivative.

### The earlier `Sharp`/`WithInner`/`cob` trouble was self-inflicted
The first cut raised the index with `Sharp(WithInner(M,g))`, which tagged the result with a
`WithInner` *basis*; summing that against the flat derivative dragged in `cob` and abstract
metricized bases. A coordinate-free derivative must never realize a basis — switching the
raise to the bare `Pinv(g)` map removed the basis entirely and unblocked `(1,1)`. Tasks #17
(cob abstract bases) and the old `cob._numeric_dim` mitigation are **no longer needed** for
covariant diff.

## Covariant-tensor coverage so far
**Done, basis-free:** vector `v:M`, covector `ω:M*`, **all-covariant forms** `(0,p)` (incl.
**the metric** — `∇g=0` verified to machine precision), and **`(1,1)` maps** `T:M⊸M`
(`∇T=ΓT−TΓ`, `∇I=0`). All via per-slot `Eval`/`Compose` of the bare `Γ(h)`.

**Remaining — pure contravariant slots (`(2,0)`/`(p,0)`):** Kron free-label threading is now
done (`_rep_kron`/`_KronFreeOp`), but applying `Γ` to a primal slot via
`Eval(Kron(I,…,Γ,…,I),(0,),v)` hits the **nested-⊗ codomain** issue (task #16/#18): `Kron`'s
codomain is a nested `OProd`, so `Eval` returns the *fused* `(|M|²,)` cod, which mismatches
the flat `(|M|,|M|)` of the covariant terms in the `Sum`. Fix options: (a) unfuse the Eval
result (`InvLift`/reshape, threading the seed), or (b) use `OProd`+`Trace` per slot (keeps
factors flat) — needs `OProdV`/`Trace` free-label threading. Then `(2,0)` + general `(p,q)`
land. Tests: `∇(g⁻¹)=0`, `∇(S⊗T)=∇S⊗T+S⊗∇T`.

## Known limitations to lift (each is a task)
1. **Cotangent / dual sections.** Covariant derivative of a 1-form `ω`:
   `∇_h ω = dω − Γᵀ·ω` (lower index, opposite sign). Detect a metricized `Dual`/covector
   section and apply the dual connection. *Test:* `∇⟨ω, v⟩ = ⟨∇ω, v⟩ + ⟨ω, ∇v⟩` (the metric
   pairing is parallel).
2. **Higher-rank tensor sections.** General `(p,q)`-tensor field: one `+Γ` per contravariant
   slot, one `−Γ` per covariant slot (covariant Leibniz over slots). Today only rank-1.
   *Test:* `∇g = 0` (metric compatibility) and `∇(S⊗T) = ∇S⊗T + S⊗∇T`.
3. **Connection from a frame / chart Jacobian (the FEM case).** Define `∇` by a moving frame
   `e_a(x)` (e.g. a chart immersion `J(x)`); `Γ` from `∂e`. This is the tangent-space case
   that matters most in practice (FEMOracle `baseExt/forms.py` tangent/cotangent frames).
   Likely subsumes the metric route via `g = JᵀJ` (task 4).
4. **Composite x-dependent metrics.** Lift the "metric must be a marked Var" restriction:
   differentiate `g(x) = Jᵀ G J` (a pullback metric) by recursively computing `∂g` from `∂J`,
   reusing the metric-gradient hook. Removes the need to hand-supply `Dg`.
5. **Covariant Leibniz through the expression, not just the top.** Apply `∇` compositionally
   (`∇` distributes through `Eval`/`OProd` of tensor-valued sections) instead of only at the
   outermost value. Needed once sections appear as subterms of a larger contraction.
6. **`X ≠ s0`.** Support a section whose tangent space differs from the differentiation
   variable's space (pull the connection back along the parameter map). Currently asserted equal.

## Higher-rank recipe (task 2) — connection map DONE, blocked on basis-infra
The connection map `Γ(h):M⊸M` is **built and validated** (`diff._connection_map`):
`Γ(h) = Compose(Sharp(MR), (0,), ½(T1 + T2 − Permute(T2,(1,0))))` with
`T1=Eval(Dg,(0,),h)`, `T2=Eval(Dg,(1,),h)` — enabled by the `M⊸M*` decode generalization
(`_decode_map_space`) + `Permute` + Eval f-free-labels. `Eval(Γ(h),(0,),v)` matches the
Christoffel contraction (tested).

The `(1,1)` tensor case `∇_h T = Conn∘T − T∘Conn` (Compose-based, both Composes thread the
seed) is **coded but gated** (`_connection_term` raises): the correction's basis is
`WithInner`-contaminated through `Sharp`, so summing it against the flat derivative needs
**reconciling abstract metricized bases in `cob`** — the real remaining blocker:
- `cob` can't size an identity between two *equal abstract* `BaseBasis` (no explicit dim);
  partially mitigated (`cob._numeric_dim` sizes from a numeric metric) but the endo
  correction's basis differs *structurally* from the flat part (nested `WithInner` via
  `Sharp`), needing realization of abstract bases. Fix: either keep the connection map's
  result in the bare (default) basis (strip `WithInner` from `Sharp`'s output), or make `cob`
  treat `WithInner`-decorated bases as their inner basis end-to-end (incl. realize/size).
- Once unblocked: `(1,1)` lands immediately (`∇I=0`, `∇T=ΓT−TΓ` already validated offline);
  general `(p,q)` follows via per-slot `Kron(I,…,Γ(h),…,I)`+`Eval` (needs Kron free-labels)
  or `Permute`-to-front. Tests: `∇g=0`, `∇(S⊗T)=∇S⊗T+S⊗∇T`.

## New capabilities (build on the connection)
7. **Christoffel symbols as a first-class term** `christoffel(g) : Γ^k_{ij}`. Reusable
   (geodesics, parallel transport, curvature) instead of inlined in the correction.
8. **Second covariant derivative / Hessian** `∇∇` and the **Riemann curvature**
   `R = ∇∇ − ∇∇`. *Tests:* `∇²` of a scalar is symmetric; `R = 0` for a flat metric;
   a known closed form on the 2-sphere / polar coordinates.
9. **Parallel transport & geodesics.** `∇_{γ'} V = 0` (transport ODE) and `∇_{γ'} γ' = 0`
   (geodesic), as derived terms over the connection (task 7).
10. **Exterior covariant derivative** `d_∇` for vector-valued forms — combines `wedge`
    (already pullback-functorial) with `∇`. *Test:* `d_∇ d_∇ ω = R ∧ ω`.
11. **Metric-compatible / torsion-free assertions.** A check that the realized connection
    satisfies the two defining Levi-Civita properties — turns task 2's `∇g=0` into a
    reusable invariant test.

## Differentiation gaps (not strictly covariant, but blocking generality)
12. **Differentiate w.r.t. an inverted/pseudo-inverted map** (`Solve`/`Pinv` with `x` in the
    matrix). Currently `DiffError`. The derivative is a higher tensor (`∂f` carries the seed
    *and* a full map codomain); needs substrate task 15/16.
13. **Full Moore–Penrose derivative.** Only the leading `−f†(df)f†` term is implemented; add
    the rank-deficient Golub–Pereyra terms (`(I−ff†)`, `(I−f†f)`), gated on `inj`/`surj`.
14. **Value-`Lam` chain rule.** A value-producing `Lam` (a `Π`, not a function over spaces)
    could be differentiated through its body; today all `NEval`/`Lam` are rejected as
    higher-order. (SpaceFunctions/`Frame` stay rejected — those are genuinely higher-order.)

## Substrate enablers (prerequisites for 5, 12)
15. **Free-label threading through bulk ops** (`SolveOp`/`PinvOp`/`InvOp`): batched RHS so a
    seed axis rides through, the way `Eval`/`_rep_compose` now do. Unlocks seed-through-bulk
    cases generally (task 12).
16. **Nested-factor codomains in Eval/Compose** (or an `(dom,cod)`-layout LLam-map mode):
    a derivative whose codomain is itself a map space currently collapses to one fused factor
    and can't be re-contracted. Unlocks map-position differentiation (task 12) and a clean
    "full Jacobian as an Eval-able tensor" (helps task 5).

## Suggested order
`2 → 1 → 7 → 8` builds the tensor-calculus core (then `∇g=0`, curvature tests land).
`3 → 4` delivers the practical FEM frame/chart connection in parallel. `15 → 16 → 12 → 13`
is the substrate track that also benefits non-covariant diff. `9, 10, 11` are applications
once `7`/`8` exist.
