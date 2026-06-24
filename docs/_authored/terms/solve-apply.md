---
status: review
---

**Applied exact solve** `SolveApply(f, v)` — apply the inverse map in `Eval`'s canonical
contraction-view (a map's input is its `Dual` slot): `SolveApply(f, v) ≡ Eval(Solve(f), (0,), v)
= solve(fᵀ, v)` for a square full-rank `f`, without materializing `f⁻¹`. This satisfies the
application invariant `XApply(f, v) ≡ Eval(X(f), v)` shared with `PinvApply` (see
`plans/91-application-conventions.md`). Lowers structurally to `Solve(f)` (an inverse map) applied
by an `Eval` contraction — honest per-axis bases, unlike a bare `SolveOp`.
