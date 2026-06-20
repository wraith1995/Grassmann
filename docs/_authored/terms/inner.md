---
status: review
---

The **inner-product map** `Inner(s)` — the metric `s ⊗ s ⊸ ℝ`, i.e. `⟨·,·⟩` as a bilinear form.
`dot(x,y)` is `Eval(Inner(s),(0,1),x,y)`. The Gram is the identity in an orthonormal basis (the
min-sym default) and the immersed Gram `JᵀJ` in a non-orthonormal `ImageBasis`; for a `WithInner`
space it is the explicit metric `g`. Built in the operands' active basis at the contraction site.
