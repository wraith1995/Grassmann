---
status: review
---

The **kernel** `Ker(f)` of a linear map `f : dom ⊸ cod` — the subspace of `dom` that `f`
annihilates (`f·x = 0`), a fundamental subspace. Dimension `|dom| − rank(f)`:
static when props pin the rank, else a runtime `δ_f`.

`Ker` types null spaces, constraint manifolds, and gauge freedoms. Its basis is
[`ProjectedBasis(f,"ker")`](../bases/projected-basis.md) — the trailing right singular vectors
`V` of `f`, realized by the SVD seam; the columns lie in `dom` and satisfy `f·col = 0`.
