---
status: review
---

A **space-function application** `App(g, positions, args)` — applies a space/type-level function
(a `SpaceFunction`, the `BLam`/`Lam` over space/nat domains) to space, dim, or term arguments,
mirroring `Eval`/`NEval` at the type level.

This is the reduction site for space polymorphism: `(λα. α⊗α)[V]` is an `App` that β-reduces to
`V⊗V`. An *unreduced* `App` has no static dimension or default basis — it must be reduced (its
function resolved and substituted) before it can be lowered to coordinates.
