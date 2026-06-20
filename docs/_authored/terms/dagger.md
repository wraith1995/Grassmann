---
status: review
---

**Metric adjoint** `Dagger(f)` — `f†`, the adjoint of `f` defined by `⟨f† a, b⟩ = ⟨a, f b⟩`. In
coordinates `f† = G_cod·fᵀ·G_dom⁻¹` with per-slot metrics in the active bases. In the orthonormal
default both metrics are the identity, so it collapses to a plain transpose (the min-sym win); a
non-orthonormal (`ImageBasis`) endpoint routes through the metric formula so the adjoint identity
stays basis-invariant (`43 §2c`). Surface: `f.dagger` / `f.T`.
