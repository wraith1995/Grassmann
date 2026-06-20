---
status: review
---

The field **ℝ** itself — the one-dimensional space of scalars (`Scalar`, dim 1).

It is the codomain of every inner product and trace, the result type of a fully-contracted
tensor, and the multiplier type of `Scale`. As a space it has a single basis vector (the
number `1`), so a value lowers to a polyarray array of shape `()` (a 0-D scalar). There is
nothing to choose about its basis — `ScalarBasis` is canonical and orthonormal.
