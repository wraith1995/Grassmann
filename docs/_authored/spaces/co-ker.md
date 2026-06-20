---
status: review
---

The **cokernel** `CoKer(f)` of `f : dom ⊸ cod` — the orthogonal complement of `Img(f)` inside
`cod`. Dimension `|cod| − rank(f)`. It is what `f` *fails* to reach in the
codomain, so it captures the obstruction/residual directions of a least-squares problem.

Basis [`ProjectedBasis(f,"coker")`](../bases/projected-basis.md) — the trailing left singular
vectors `U` of `f` (in `cod`), realized by the SVD seam.
