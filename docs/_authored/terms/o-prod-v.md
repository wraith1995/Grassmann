---
status: review
---

A **tensor (outer) product of vectors** `OProdV(v₀,…)` — the value `v₀ ⊗ … ⊗ vₙ` of space
`⊗(sᵢ)`, an einsum outer product over the factors' axes. Each factor keeps its own axes
(unfused); any free basis-index labels a factor carries ride through, so the outer product of
`LLam`-bound matrix operands assembles the full map tensor.
