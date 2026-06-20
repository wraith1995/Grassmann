---
status: review
---

A space **re-metricized** `WithInner(s, g)` — the same underlying space `s` carrying an explicit
inner product `g : s* ⊗ s*` (an SPD metric), per `20` point 9. The metric is metadata: it does
not change dimension or coordinates, only how lengths/angles and the adjoint are computed.

This is how a non-Euclidean metric enters: `inner`/`flat`/`sharp`/`dagger`/`pinv` over a
`WithInner` space read `g` instead of the identity (an SPD assertion guards it). The metric `g`
may be a numeric matrix or a *term-valued* field (e.g. an x-dependent metric). The **default**
basis is the `g`-orthonormal [`WithInnerBasis`](../bases/with-inner-basis.md) — in those
coordinates `g` becomes the identity (the min-sym fast path); the un-whitened standard basis is
kept as a candidate, where `g` is materialized. Created at the surface by `s.with_inner(g)` /
the [`Remetric`](../terms/remetric.md) term on values.
