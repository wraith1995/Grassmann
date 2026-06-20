---
status: review
---

The **tensor product** `OProd(s₀,…,sₙ)` = `s₀ ⊗ … ⊗ sₙ` — the multilinear product. Dimension
is the **product** of the parts; a value lowers to a multi-axis array of shape
`(|s₀|,…,|sₙ|)`, one axis per factor (axes are kept *unfused* so structure is visible to
downstream nodes).

`⊗` is the workhorse: a rank-k tensor is a k-fold `⊗`, and a **linear map** `s ⊸ t` is the
tensor `s* ⊗ t` (a dual input slot tensored with an output slot). Hence `Eval`/`Compose`
contract `⊗` axes, and `Lift`/`InvLift` reassociate fused vs unfused `⊗` slots.
`linear_map_space(s,t)` is the convenience spelling `~s * t`. Its basis is the Kronecker
[`BasisProd`](../bases/basis-prod.md) of the factor bases.
