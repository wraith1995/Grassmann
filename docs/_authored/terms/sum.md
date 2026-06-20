---
status: review
---

**Vector addition** `Sum(v₀,…,vₙ)` — n-ary `v₀ + … + vₙ`; all summands share one space. The
abelian-group operation of a vector space. At lowering, summands are first **change-of-based to a
common basis** (`_coerce_basis`, now FFS-total) and their free (basis-index) axes aligned, then
added — so a sum of differently-pinned operands is reconciled, not miscomputed.
