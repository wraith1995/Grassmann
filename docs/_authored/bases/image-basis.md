---
status: review
---

The **literal map-application basis** `ImageBasis(f)` — the *raw* columns `f(eᵢ)` of `f`'s range
(`73 T2`), i.e. the operator matrix itself, **not** orthonormalized. Distinct from the
orthonormal [`ProjectedBasis`](projected-basis.md): in `ImageBasis` the immersed metric is the
non-identity Gram `JᵀJ` (and its dual carries `(JᵀJ)⁻¹`). This is the basis to pin when modeling a
tangent space in *source/pushforward* coordinates (the pointwise substrate). Realized as the
literal columns (no SVD); change-of-basis to/from it routes through the FFS seam
(`_basis_cob_matrix`).
