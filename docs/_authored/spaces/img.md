---
status: review
---

The **image (range)** `Img(f)` of a linear map `f : dom ⊸ cod` — the subspace of `cod` spanned
by `f`'s columns, one of the four fundamental subspaces (`def:ffs`). Its dimension is
`rank(f)`: **static** (`= |dom|`) when `f` is asserted injective, otherwise a **runtime** rank
`δ_f` resolved at evaluation.

`Img` is the substrate for modeling immersed tangent spaces (a chart Jacobian's image) and for
typing the result of a rank-revealing operation. Its **default** basis is the orthonormal
range basis [`ProjectedBasis(f,"image")`](../bases/projected-basis.md) — the `Q` of `f`'s QR
(injective) or the leading left singular vectors `U` (general). The literal, non-orthonormal
columns `f(eᵢ)` are the separate [`ImageBasis`](../bases/image-basis.md) candidate; in that
basis the metric is the immersed Gram `JᵀJ`. Lowering realizes the columns through the
SVD/QR/GSVD seam (`41 §1`).
