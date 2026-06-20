---
status: review
---

**Applied exact solve** `SolveApply(f, v)` — `f⁻¹ · v`, solving `f·x = v` for a square full-rank
`f` without materializing `f⁻¹`. Lowers to a `SolveOp` (a linear solve), which is cheaper and
better-conditioned than forming the inverse then multiplying.
