# 44 — Exterior-power `inner` does not decode (blocks k-form metrics)

Found while wiring k-form value spaces in `pointwise` (see `pointwise/plans/39`). Sibling of the
FFS-static-dims work (`43`): the dimension/decode laws don't reach exterior powers.

## Symptom
`inner` / `dot` on a `Λᵏ` (k≥2) cannot be compiled:
```python
import grassmann as G
V  = G.space("Re", 3)
L1 = G.exterior_power(V, 1); a = G.var("a", L1); b = G.var("b", L1)
w  = G.wedge(a, b, 1, 1)                      # a 2-vector in Λ²(V) — VALUE compiles fine
G.compile(G.dot(w, w))                        # EmbedError: G.dot could not infer the operand's space
m  = G.inner(G.exterior_power(V, 2))
G.compile(G.compose(G.compose(m, w), w))      # InferError: cannot decode map space (dom ⊸ cod): Img(alt(V,k))
```
- The **wedge value** compiles and runs (`e₁∧e₂ → [0, -√2, 0]`).
- The **metric** does not: `inner(ΛᵏV) : ΛᵏV ⊗ ΛᵏV ⊸ R` has domain `ΛᵏV = Img(alt(V,k))`, whose dim the
  map-space decoder can't resolve into `dom ⊸ cod`.
- Holds for a **static** base (`Λ²(R³)`, `|Λ²|=3`) AND a dynamic `Λ²(Image(J))` — so it is NOT only the
  FFS dynamic-rank gap; the `Img(alt)` domain itself doesn't decode.

## The rank-pin is mutually exclusive with wedge
`exterior_power(V, k, rank=C(n,k))` makes the dim static (the explicit-`Var` projector path), but then
`wedge` raises *"could not recover the base space V from the operand's Λᵏ; ensure it was built with
exterior_power(V, k)"* — `_base_space_of_exterior` only recovers `V` from the `alt`-built form. So one
cannot both pin the dim (for `inner`) and wedge (which needs the `alt` form). And even the rank-pinned
`Λᵏ` still failed `inner` decode (`Img(Var('Alt_2', …))`).

## Fix (proposed)
1. **Exterior-power dim law** so the `inner`-map / `Img(alt(V,k))` decodes WITHOUT an explicit rank:
   `|Λᵏ Img(alt(V,k))| = C(|V|, k)` (and `= C(rank, k)` when `V = Img(inj: Rᵗ⊸Rᵉ)`, i.e. compose with the
   FFS rank laws from `43`). Thread it into the map-space decoder so `inner(exterior_power(V,k))` resolves.
2. **Reconcile rank-pinning with `wedge`**: let `_base_space_of_exterior` recover `V` from a statically
   dimensioned `Λᵏ` too (or make `exterior_power` always carry `V` retrievably), so pinned `Λᵏ` is wedge-able.
3. `G.dot` could fall back to `inner(space_of(x))` more aggressively, but (1) is the real fix.

## Why it matters
Blocks FEEC / k-forms / moment DOFs end to end in `pointwise` (the Grassmann backend's payoff over the
dumb shape-tuple backend — exterior algebra). Once `inner(Λᵏ)` compiles, the pointwise k-form slice
(a 2-form mass, then a moment DOF) lands immediately; the value-space wiring is already staged
(`pointwise single_compile.types.Ext`).

Status: NOTED, not fixed. UNCOMMITTED working-tree note (the user pushes grassmann manually).

## RESOLVED (2026-06-26) — `dot(Λᵏ)` compiles; the `k!` is a convention, handled downstream

The "cannot decode map space" in the symptom above was a RED HERRING — it's `compose`-into-a-map-slot
misuse (fails for plain `R³` too), not the real path. The real path is `G.dot(w, w)`, which failed only
because `_space_of` couldn't infer a *derived* operand's space.

**Fix (one change, `embed/builders.py::_space_of`):** synthesize the operand's space via `synth_space`
when it has no explicit `.space`. `dot` then keys `Inner` on the operand's OWN `Λᵏ = Img(alt(V,k))` node,
so it shares the `alt` term (no alpha-mismatch, no memoization). The metric lowers through the EXISTING
dynamic lane (`dim_of(Img(alt))` defers → `DynamicDim` → `_dyn_identity` → FFS realization of the alt
projector at runtime). No static dim law, no metric code. `tests/test_exterior_dot.py`.

**The remaining `k!` is a convention, NOT a bug.** Native `dot(Λᵏ)` returns the **tensor** inner
`k!·det(Gram)` (because `wedge` builds the un-normalized antisymmetric tensor; its `ProjectedBasis` is
tensor-orthonormal). The **geometric/determinant** inner `det[⟨αᵢ,βⱼ⟩]` (the standard k-form/FEEC inner)
is `(1/k!)·` that. Per the consumer's decision (`pointwise/plans/40`, option 3), Grassmann keeps the
tensor convention unchanged and the `1/k!` is applied at the pointwise boundary. So no Grassmann metric
change is needed here.
