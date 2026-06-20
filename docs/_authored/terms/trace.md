---
status: review
---

**Trace / contraction of two slots** `Trace(f, i, j)` — contract axes `i` and `j` of a tensor
(the diagonal sum), e.g. the matrix trace `Σ Aᵢᵢ`. Lowers to a diagonal-sum einsum over the two
named axes.
