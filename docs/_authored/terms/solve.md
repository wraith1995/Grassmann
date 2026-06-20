---
status: review
---

**Exact inverse map** `Solve(f)` — `f⁻¹` for a **square, full-rank** `f`. A guard
(`assert_square_full_rank`) enforces invertibility; lowers to an `InvOp`. Use `Solve` (not
`Pinv`) when the map is genuinely bijective — it is the min-sym, exact path.
