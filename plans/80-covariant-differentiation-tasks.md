# Covariant differentiation — task backlog

The first cut (`src/grassmann/transform/diff.py`) gives the total derivative `X⊸space(e)`
for closed terms, with a Levi-Civita connection correction for a **rank-1 section in a
tangent space** (`Remetric(sec, g)` with a marked metric). This doc lists the work to make
covariant differentiation general. Ordered roughly by value × independence.

## Current scope (done)
- Total/Fréchet derivative: product/chain/quotient over Eval/Compose/OProd/Scale/Inner +
  nonlinear scalar External; applied inverses (`Solve`/`Pinv` through the vector), `Dagger`.
- Connection for **rank-1 sections**, both slot types (`Remetric(sec, g)`, marked `g`/`Dg`,
  `X = M`): tangent `∇_h v = dv + ½g⁻¹(Γ·h·v)` and cotangent `∇_h ω = dω − ½(Γ·h·♯ω)`.
  Both built by contracting `Dg` with `h` and the (raised) section as Eval args — no tensor
  transpose. Verified vs analytic Christoffel + the metric-pairing invariant.
- Flat/constant metric ⇒ `Γ = 0` exactly (recovers the total derivative).

## DISCOVERED: substrate gates for higher-rank (task 2 ⇐ task 16)
Higher-rank needs the general `(1,1)` connection *map* `Γ(h):M⊸M`, which needs two things
the IR lacks (so **task 2 is blocked on task 16**):
- **A tensor axis-permute / `(0,2)`-form transpose.** `Dagger` only transposes proper maps
  (1 dual + 1 prim); a `(0,2)` form (2 duals) has no transpose, and raise→dagger→lower fails
  on nested-dual decoding. Add a `Permute(v, perm)` node (lowers to a transpose einsum).
- **`Eval` `f`-position free-label threading.** Applying a seed-carrying `Kron(I,…,Γ(h),…,I)`
  map to one tensor slot puts the seed on the map (`f`) side; `_rep_eval` threads *arg*
  free-labels but assumes `f` is seed-free (`_rep_compose` already threads both — mirror it).

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
