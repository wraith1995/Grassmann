---
status: review
---

The **direct sum** `OPlus(s₀,…,sₙ)` = `s₀ ⊕ … ⊕ sₙ` — independent components stacked side by
side. Dimension is the **sum** of the parts; a value lowers to a single 1-D array of length
`Σ|sᵢ|` formed by concatenating the component coordinate blocks.

Use `⊕` for products of distinct quantities that do not interact multiplicatively — a state
made of a position and a velocity, a block vector, a tuple. The component injections/projections
are [`TupleEmbed`](../terms/tuple-embed.md) / [`TupleProject`](../terms/tuple-project.md) (pad
/ slice the block). `bot()` is the empty sum `⊕()`, the zero-dimensional space. Its basis is the
block-diagonal [`BasisSum`](../bases/basis-sum.md) of the component bases.
